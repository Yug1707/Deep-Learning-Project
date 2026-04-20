"""Train AST for multi-label genre classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.ast_pipeline.common import (
    DEFAULT_CONFIG,
    configure_logger,
    ensure_output_dirs,
    load_config,
    save_json,
    save_run_snapshot,
    set_seed,
)
from training.ast_pipeline.data import ASTBatchCollator, ASTManifestDataset
from training.ast_pipeline.model import (
    create_ast_model,
    freeze_backbone,
    load_feature_extractor,
    save_hf_artifacts,
    unfreeze_backbone,
)
from utils.metrics import MultiLabelMetrics


def _load_class_mapping(manifests_dir: Path) -> Dict:
    class_map_path = manifests_dir / "class_mapping.json"
    if not class_map_path.exists():
        raise FileNotFoundError(
            f"Missing class mapping at {class_map_path}. Run build_dataset stage first."
        )

    with open(class_map_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _move_inputs_to_device(batch_inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch_inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
    return moved


def _run_epoch(
    model,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    grad_accum_steps: int,
    max_grad_norm: float,
    training: bool,
    metrics_calculator: MultiLabelMetrics,
) -> Tuple[float, Dict[str, float]]:
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    num_batches = 0
    probs_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    iterator = tqdm(loader, desc="Train" if training else "Eval", leave=False)
    processed_steps = 0

    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(iterator, start=1):
        if batch is None:
            continue

        processed_steps += 1
        labels = batch["labels"].to(device)
        model_inputs = _move_inputs_to_device(batch["model_inputs"], device)

        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**model_inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)

            if training:
                scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()

                if processed_steps % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        num_batches += 1

        probs = torch.sigmoid(logits.detach()).cpu()
        probs_all.append(probs)
        labels_all.append(labels.detach().cpu())

    if training and processed_steps > 0 and (processed_steps % grad_accum_steps) != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if probs_all:
        pred_tensor = torch.cat(probs_all, dim=0)
        label_tensor = torch.cat(labels_all, dim=0)
        metrics = metrics_calculator.calculate_all_metrics(pred_tensor, label_tensor)
    else:
        num_labels = max(1, int(getattr(loader.dataset, "num_labels", 1)))
        metrics = metrics_calculator.calculate_all_metrics(
            torch.zeros((0, num_labels), dtype=torch.float32),
            torch.zeros((0, num_labels), dtype=torch.float32),
        )

    avg_loss = total_loss / max(1, num_batches)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def _save_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    epoch: int,
    best_f1: float,
    config: Dict,
    history: Dict,
) -> None:
    payload = {
        "epoch": epoch,
        "best_f1": float(best_f1),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "history": history,
    }
    torch.save(payload, checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AST multi-label classifier")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to AST pipeline config")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["seed"])
    set_seed(seed)

    dirs = ensure_output_dirs(config)
    logger = configure_logger(dirs["runtime_logs_dir"] / "train.log", logger_name="ast_train")
    save_run_snapshot(config, dirs["results_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    class_mapping = _load_class_mapping(dirs["manifests_dir"])
    num_labels = int(class_mapping["top_k_genres"])

    ast_cfg = config["ast"]
    audio_cfg = config["audio"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    model_name = ast_cfg["model_name"]
    cache_dir = ast_cfg.get("cache_dir")

    feature_extractor = load_feature_extractor(model_name=model_name, cache_dir=cache_dir)
    model = create_ast_model(model_name=model_name, num_labels=num_labels, cache_dir=cache_dir).to(device)

    freeze_epochs = int(ast_cfg.get("freeze_feature_encoder_epochs", 0))
    if freeze_epochs > 0:
        freeze_backbone(model)
        logger.info("Backbone frozen for first %d epoch(s)", freeze_epochs)

    train_ds = ASTManifestDataset(
        manifest_path=dirs["manifests_dir"] / "train_manifest.csv",
        sample_rate=int(audio_cfg["sample_rate"]),
        clip_seconds=float(audio_cfg["clip_seconds"]),
        res_type=str(audio_cfg.get("res_type", "soxr_hq")),
        mode="train",
        seed=seed,
        min_audio_seconds=float(config["dataset"].get("min_audio_seconds", 0.0)),
    )
    val_ds = ASTManifestDataset(
        manifest_path=dirs["manifests_dir"] / "val_manifest.csv",
        sample_rate=int(audio_cfg["sample_rate"]),
        clip_seconds=float(audio_cfg["clip_seconds"]),
        res_type=str(audio_cfg.get("res_type", "soxr_hq")),
        mode="val",
        seed=seed,
        min_audio_seconds=float(config["dataset"].get("min_audio_seconds", 0.0)),
    )

    collator = ASTBatchCollator(feature_extractor=feature_extractor, sample_rate=int(audio_cfg["sample_rate"]))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        collate_fn=collator,
        drop_last=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(train_cfg["num_epochs"])),
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(device.type, enabled=bool(train_cfg.get("amp", True) and device.type == "cuda"))
    metrics_calculator = MultiLabelMetrics(threshold=float(eval_cfg["threshold"]))

    start_epoch = 1
    best_f1 = -1.0
    history: Dict[str, List[Dict[str, float]]] = {"train": [], "val": []}

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = Path.cwd() / resume_path
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_f1 = float(ckpt.get("best_f1", -1.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        history = ckpt.get("history", history)
        logger.info("Resumed from checkpoint: %s", resume_path)

    num_epochs = int(train_cfg["num_epochs"])
    save_every = int(train_cfg.get("save_every", 1))

    for epoch in range(start_epoch, num_epochs + 1):
        if freeze_epochs > 0 and epoch == (freeze_epochs + 1):
            unfreeze_backbone(model)
            logger.info("Backbone unfrozen at epoch %d", epoch)

        train_ds.set_epoch(epoch)

        logger.info("Epoch %d/%d", epoch, num_epochs)
        _, train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=bool(train_cfg.get("amp", True) and device.type == "cuda"),
            grad_accum_steps=max(1, int(train_cfg.get("gradient_accumulation_steps", 1))),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            training=True,
            metrics_calculator=metrics_calculator,
        )

        with torch.no_grad():
            _, val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                use_amp=bool(train_cfg.get("amp", True) and device.type == "cuda"),
                grad_accum_steps=1,
                max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
                training=False,
                metrics_calculator=metrics_calculator,
            )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        scheduler.step()

        last_ckpt = dirs["checkpoints_dir"] / "last.pt"
        _save_checkpoint(last_ckpt, model, optimizer, epoch, best_f1, config, history)

        if (epoch % save_every) == 0:
            epoch_ckpt = dirs["checkpoints_dir"] / f"epoch_{epoch:03d}.pt"
            _save_checkpoint(epoch_ckpt, model, optimizer, epoch, best_f1, config, history)

        current_f1 = float(val_metrics.get("f1_macro", 0.0))
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_ckpt = dirs["checkpoints_dir"] / "best.pt"
            _save_checkpoint(best_ckpt, model, optimizer, epoch, best_f1, config, history)
            logger.info("New best model at epoch %d with val f1_macro=%.4f", epoch, current_f1)

        logger.info("Train metrics: %s", train_metrics)
        logger.info("Val metrics: %s", val_metrics)

    history_payload = {
        "best_f1_macro": best_f1,
        "num_epochs": num_epochs,
        "history": history,
    }
    save_json(history_payload, dirs["results_dir"] / "training_history.json")

    best_ckpt = dirs["checkpoints_dir"] / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    save_hf_artifacts(model, feature_extractor, dirs["models_dir"])
    logger.info("Training complete. Artifacts saved under %s", dirs["output_root"])


if __name__ == "__main__":
    main()
