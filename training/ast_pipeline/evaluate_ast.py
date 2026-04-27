"""Evaluate trained AST checkpoint on val/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from sklearn.metrics import f1_score, multilabel_confusion_matrix
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
from training.ast_pipeline.model import create_ast_model, load_feature_extractor, apply_classifier_dropout
from utils.metrics import MultiLabelMetrics


def _load_class_map(manifests_dir: Path) -> Dict[str, str]:
    mapping_path = manifests_dir / "class_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("index_to_genre", {})


def _resolve_model_source(models_dir: Path, model_name: str) -> tuple[str, bool]:
    """Prefer local saved Hugging Face artifacts when available."""
    required = ["config.json", "preprocessor_config.json"]
    if models_dir.exists() and all((models_dir / filename).exists() for filename in required):
        return str(models_dir), True
    return model_name, False


def _evaluate_split(
    model,
    loader: DataLoader,
    threshold: float,
    device: torch.device,
    class_map: Dict[str, str],
    split_name: str,
) -> Dict:
    metrics_calc = MultiLabelMetrics(threshold=threshold)
    probs_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
            if batch is None:
                continue

            model_inputs = {
                key: value.to(device)
                for key, value in batch["model_inputs"].items()
                if isinstance(value, torch.Tensor)
            }
            labels = batch["labels"].cpu()
            outputs = model(**model_inputs)
            probs = torch.sigmoid(outputs.logits).cpu()
            probs_all.append(probs)
            labels_all.append(labels)

    if not probs_all:
        num_labels = max(1, len(class_map))
        empty_probs = torch.zeros((0, num_labels), dtype=torch.float32)
        empty_labels = torch.zeros((0, num_labels), dtype=torch.float32)
        return {
            "split": split_name,
            "num_samples": 0,
            "metrics": metrics_calc.calculate_all_metrics(empty_probs, empty_labels),
            "per_class_confusion": [],
        }

    probs_tensor = torch.cat(probs_all, dim=0)
    labels_tensor = torch.cat(labels_all, dim=0)
    metrics = metrics_calc.calculate_all_metrics(probs_tensor, labels_tensor)

    y_true = (labels_tensor.numpy() >= 0.5).astype(int)
    y_pred = (probs_tensor.numpy() >= threshold).astype(int)

    confusion = multilabel_confusion_matrix(y_true, y_pred)
    per_class: List[Dict] = []
    for class_idx, matrix in enumerate(confusion):
        tn, fp, fn, tp = matrix.ravel().tolist()
        class_f1 = f1_score(y_true[:, class_idx], y_pred[:, class_idx], zero_division=0)
        per_class.append(
            {
                "class_index": class_idx,
                "genre_id": class_map.get(str(class_idx), str(class_idx)),
                "support": int(y_true[:, class_idx].sum()),
                "f1": float(class_f1),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return {
        "split": split_name,
        "num_samples": int(y_true.shape[0]),
        "metrics": metrics,
        "per_class_confusion": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AST model")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to AST pipeline config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional explicit checkpoint path")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    dirs = ensure_output_dirs(config)
    logger = configure_logger(dirs["runtime_logs_dir"] / "eval.log", logger_name="ast_eval")
    save_run_snapshot(config, dirs["results_dir"])

    audio_cfg = config["audio"]
    ast_cfg = config["ast"]
    threshold = float(config["evaluation"]["threshold"])
    inference_cfg = config.get("inference", {})
    eval_num_workers = int(inference_cfg.get("num_workers", 0))

    if sys.version_info >= (3, 14) and eval_num_workers > 0:
        logger.warning(
            "Python %s detected; forcing evaluate DataLoader num_workers=0 for stability",
            ".".join(map(str, sys.version_info[:3])),
        )
        eval_num_workers = 0

    class_map = _load_class_map(dirs["manifests_dir"])
    num_labels = len(class_map)

    model_source, local_only = _resolve_model_source(dirs["models_dir"], ast_cfg["model_name"])
    if local_only:
        logger.info("Loading feature extractor/model from local artifacts: %s", model_source)
    else:
        logger.info("Local artifacts not found in %s; loading from %s", dirs["models_dir"], model_source)

    feature_extractor = load_feature_extractor(
        model_source,
        cache_dir=ast_cfg.get("cache_dir"),
        local_files_only=local_only,
    )
    model = create_ast_model(
        model_name=model_source,
        num_labels=num_labels,
        cache_dir=ast_cfg.get("cache_dir"),
        local_files_only=local_only,
        hidden_dropout_prob=ast_cfg.get("hidden_dropout_prob"),
        attention_probs_dropout_prob=ast_cfg.get("attention_probs_dropout_prob"),
        classifier_dropout_prob=ast_cfg.get("classifier_dropout_prob"),
    )
    apply_classifier_dropout(model, dropout_prob=float(ast_cfg.get("classifier_dropout_prob", 0.3)))

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else dirs["checkpoints_dir"] / "best.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    collator = ASTBatchCollator(feature_extractor=feature_extractor, sample_rate=int(audio_cfg["sample_rate"]))

    report: Dict[str, Dict] = {}
    for split in ["val", "test"]:
        ds = ASTManifestDataset(
            manifest_path=dirs["manifests_dir"] / f"{split}_manifest.csv",
            sample_rate=int(audio_cfg["sample_rate"]),
            clip_seconds=float(audio_cfg["clip_seconds"]),
            res_type=str(audio_cfg.get("res_type", "soxr_hq")),
            mode=split,
            seed=int(config["seed"]),
            min_audio_seconds=float(config["dataset"].get("min_audio_seconds", 0.0)),
        )
        loader = DataLoader(
            ds,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=max(0, eval_num_workers),
            pin_memory=bool(config["training"].get("pin_memory", True)),
            collate_fn=collator,
            drop_last=False,
        )
        report[split] = _evaluate_split(
            model=model,
            loader=loader,
            threshold=threshold,
            device=device,
            class_map=class_map,
            split_name=split,
        )

    save_json(report, dirs["results_dir"] / "evaluation_report.json")
    logger.info("Evaluation complete: %s", {k: v["metrics"] for k, v in report.items()})


if __name__ == "__main__":
    main()
