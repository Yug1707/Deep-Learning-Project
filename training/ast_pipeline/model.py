"""Model helpers for AST-based multi-label classification."""

from __future__ import annotations

from pathlib import Path

from transformers import ASTFeatureExtractor, ASTForAudioClassification


def load_feature_extractor(model_name: str, cache_dir: str | None = None) -> ASTFeatureExtractor:
    """Load AST feature extractor."""
    return ASTFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)


def create_ast_model(
    model_name: str,
    num_labels: int,
    cache_dir: str | None = None,
) -> ASTForAudioClassification:
    """Load pretrained AST model configured for multi-label classification."""
    return ASTForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )


def _resolve_backbone(model: ASTForAudioClassification):
    if hasattr(model, "audio_spectrogram_transformer"):
        return model.audio_spectrogram_transformer
    if hasattr(model, "ast"):
        return model.ast
    return None


def freeze_backbone(model: ASTForAudioClassification) -> None:
    """Freeze AST backbone parameters, keeping classifier trainable."""
    backbone = _resolve_backbone(model)
    if backbone is None:
        return

    for param in backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: ASTForAudioClassification) -> None:
    """Unfreeze AST backbone parameters."""
    backbone = _resolve_backbone(model)
    if backbone is None:
        return

    for param in backbone.parameters():
        param.requires_grad = True


def save_hf_artifacts(model: ASTForAudioClassification, feature_extractor: ASTFeatureExtractor, output_dir: str | Path) -> None:
    """Save Hugging Face-compatible model + feature extractor."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    feature_extractor.save_pretrained(out)
