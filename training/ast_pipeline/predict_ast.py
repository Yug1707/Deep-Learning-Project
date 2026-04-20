"""Batch inference for AST genre classifier."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from training.ast_pipeline.common import (
    DEFAULT_CONFIG,
    configure_logger,
    ensure_output_dirs,
    load_config,
    save_json,
    save_run_snapshot,
    set_seed,
)
from training.ast_pipeline.data import (
    chunk_waveform,
    discover_audio_files,
    load_audio_file,
)
from training.ast_pipeline.model import create_ast_model, load_feature_extractor


def _load_class_map(manifests_dir: Path) -> Dict[str, str]:
    mapping_path = manifests_dir / "class_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("index_to_genre", {})


def _gather_inputs(audio: str | None, audio_dir: str | None, manifest: str | None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    if audio:
        rows.append({"track_id": "", "audio_path": str(Path(audio))})

    if audio_dir:
        for path in discover_audio_files(audio_dir):
            rows.append({"track_id": "", "audio_path": str(path)})

    if manifest:
        manifest_path = Path(manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "audio_path" not in reader.fieldnames:
                raise ValueError("Manifest must contain an 'audio_path' column")

            for item in reader:
                rows.append(
                    {
                        "track_id": str(item.get("track_id", "")),
                        "audio_path": str(item["audio_path"]),
                    }
                )

    if not rows:
        raise ValueError("Provide at least one of --audio, --audio-dir, or --manifest")

    return rows


def _batched_probabilities(
    model,
    feature_extractor,
    chunks: List[np.ndarray],
    sample_rate: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            chunk_batch = chunks[i : i + batch_size]
            features = feature_extractor(
                chunk_batch,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            model_inputs = {
                key: value.to(device)
                for key, value in features.items()
                if isinstance(value, torch.Tensor)
            }
            logits = model(**model_inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            outputs.append(probs)

    return np.concatenate(outputs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AST batch inference")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to AST pipeline config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path")
    parser.add_argument("--audio", type=str, default=None, help="Single audio path")
    parser.add_argument("--audio-dir", type=str, default=None, help="Directory of audio files")
    parser.add_argument("--manifest", type=str, default=None, help="CSV with at least audio_path column")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size")
    parser.add_argument("--output", type=str, default=None, help="Optional prediction JSONL path")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    dirs = ensure_output_dirs(config)
    logger = configure_logger(dirs["runtime_logs_dir"] / "predict.log", logger_name="ast_predict")
    save_run_snapshot(config, dirs["results_dir"])

    ast_cfg = config["ast"]
    audio_cfg = config["audio"]
    inf_cfg = config["inference"]
    threshold = float(args.threshold) if args.threshold is not None else float(config["evaluation"]["threshold"])
    batch_size = int(args.batch_size) if args.batch_size is not None else int(inf_cfg.get("batch_size", 8))
    top_k = int(inf_cfg.get("write_top_k", 5))

    class_map = _load_class_map(dirs["manifests_dir"])
    num_labels = len(class_map)

    model = create_ast_model(
        model_name=ast_cfg["model_name"],
        num_labels=num_labels,
        cache_dir=ast_cfg.get("cache_dir"),
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else dirs["checkpoints_dir"] / "best.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    feature_extractor = load_feature_extractor(ast_cfg["model_name"], cache_dir=ast_cfg.get("cache_dir"))

    rows = _gather_inputs(audio=args.audio, audio_dir=args.audio_dir, manifest=args.manifest)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_jsonl = Path(args.output) if args.output else dirs["predictions_dir"] / f"predictions_{ts}.jsonl"
    if not output_jsonl.is_absolute():
        output_jsonl = Path.cwd() / output_jsonl
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0

    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            audio_path = Path(row["audio_path"])
            record: Dict[str, object] = {
                "track_id": row.get("track_id", ""),
                "audio_path": str(audio_path),
            }
            try:
                waveform = load_audio_file(
                    audio_path,
                    sample_rate=int(audio_cfg["sample_rate"]),
                    res_type=str(audio_cfg.get("res_type", "soxr_hq")),
                )
                chunks = chunk_waveform(
                    waveform=waveform,
                    sample_rate=int(audio_cfg["sample_rate"]),
                    chunk_seconds=float(ast_cfg.get("chunk_seconds", audio_cfg.get("clip_seconds", 10.0))),
                    hop_seconds=float(ast_cfg.get("chunk_hop_seconds", 5.0)),
                    max_chunks=int(ast_cfg.get("max_chunks_per_track", 12)),
                )

                probs_per_chunk = _batched_probabilities(
                    model=model,
                    feature_extractor=feature_extractor,
                    chunks=chunks,
                    sample_rate=int(audio_cfg["sample_rate"]),
                    batch_size=batch_size,
                    device=device,
                )
                mean_probs = probs_per_chunk.mean(axis=0)

                pred_mask = mean_probs >= threshold
                pred_indices = np.where(pred_mask)[0].tolist()
                top_indices = np.argsort(mean_probs)[::-1][: min(top_k, mean_probs.shape[0])]

                record.update(
                    {
                        "success": True,
                        "num_chunks": int(len(chunks)),
                        "predicted_indices": pred_indices,
                        "predicted_genre_ids": [class_map.get(str(i), str(i)) for i in pred_indices],
                        "predicted_probabilities": [float(mean_probs[i]) for i in pred_indices],
                        "top_k": [
                            {
                                "class_index": int(i),
                                "genre_id": class_map.get(str(i), str(i)),
                                "probability": float(mean_probs[i]),
                            }
                            for i in top_indices
                        ],
                    }
                )
                successes += 1
            except Exception as exc:
                record.update(
                    {
                        "success": False,
                        "error": str(exc),
                    }
                )
                failures += 1

            handle.write(json.dumps(record) + "\n")

    summary = {
        "total": len(rows),
        "successes": successes,
        "failures": failures,
        "output_jsonl": str(output_jsonl),
        "threshold": threshold,
        "batch_size": batch_size,
    }

    summary_path = dirs["predictions_dir"] / f"prediction_summary_{ts}.json"
    save_json(summary, summary_path)
    logger.info("Inference complete: %s", summary)
    print(summary)


if __name__ == "__main__":
    main()
