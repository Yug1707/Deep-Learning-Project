"""Predict genres using VGGish embeddings and trained XGBoost model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

from training.vggish_xgboost.common import load_config, project_root
from utils.vggish_extractor import VGGishExtractor, pool_embeddings


SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".ogg"}


def _predict_one(
    model,
    extractor: VGGishExtractor,
    audio_path: Path,
    sample_rate: int,
    pooling: str,
    threshold: float,
    class_map: Dict[str, str],
) -> Dict:
    frames = extractor.extract_from_file(audio_path, sample_rate=sample_rate)
    pooled = pool_embeddings(frames, mode=pooling).reshape(1, -1)

    probs = model.predict_proba(pooled)[0]
    pred_mask = probs >= threshold
    indices = np.where(pred_mask)[0].tolist()

    labels = [class_map.get(str(i), str(i)) for i in indices]
    label_probs = [float(probs[i]) for i in indices]

    top_order = np.argsort(probs)[::-1][:5]
    top5 = [
        {
            "class_index": int(i),
            "genre_id": class_map.get(str(i), str(i)),
            "probability": float(probs[i]),
        }
        for i in top_order
    ]

    return {
        "audio_path": str(audio_path),
        "predicted_indices": indices,
        "predicted_genre_ids": labels,
        "predicted_probabilities": label_probs,
        "top5": top5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with VGGish + XGBoost")
    parser.add_argument(
        "--config",
        type=str,
        default="training/vggish_xgboost/config_vggish_xgboost.json",
        help="Path to VGGish pipeline config",
    )
    parser.add_argument("--audio", type=str, help="Single audio file")
    parser.add_argument("--audio-dir", type=str, help="Directory of audio files")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    args = parser.parse_args()

    if not args.audio and not args.audio_dir:
        raise ValueError("Provide --audio or --audio-dir")

    config = load_config(args.config)
    root = project_root()

    paths = config["paths"]
    sample_rate = int(config["audio"]["sample_rate"])
    pooling = config["vggish"].get("pooling", "mean")
    threshold = float(args.threshold) if args.threshold is not None else float(config["evaluation"]["threshold"])

    model = joblib.load(root / paths["models_dir"] / "xgboost_ovr.joblib")
    class_mapping = load_config(root / paths["output_root"] / "manifests" / "class_mapping.json")
    class_map = class_mapping.get("index_to_genre", {})

    extractor = VGGishExtractor(device=config["vggish"].get("device", "auto"))

    audio_paths: List[Path] = []
    if args.audio:
        audio_paths.append(Path(args.audio))
    if args.audio_dir:
        base = Path(args.audio_dir)
        audio_paths.extend([p for p in base.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS])

    results: List[Dict] = []
    for audio_path in audio_paths:
        try:
            pred = _predict_one(
                model=model,
                extractor=extractor,
                audio_path=audio_path,
                sample_rate=sample_rate,
                pooling=pooling,
                threshold=threshold,
                class_map=class_map,
            )
            pred["success"] = True
            results.append(pred)
        except Exception as exc:
            results.append(
                {
                    "audio_path": str(audio_path),
                    "success": False,
                    "error": str(exc),
                }
            )

    for item in results:
        print(item)


if __name__ == "__main__":
    main()
