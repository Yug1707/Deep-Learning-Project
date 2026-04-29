"""Model helpers for AST-based multi-label classification."""

from __future__ import annotations

from pathlib import Path

from transformers import ASTFeatureExtractor, ASTForAudioClassification


def load_feature_extractor(
    model_name: str,
    cache_dir: str | None = None,
    local_files_only: bool | None = None,
) -> ASTFeatureExtractor:
    """Load AST feature extractor from local artifacts or Hugging Face Hub."""
    kwargs = {"cache_dir": cache_dir}
    if local_files_only is not None:
        kwargs["local_files_only"] = bool(local_files_only)
    return ASTFeatureExtractor.from_pretrained(model_name, **kwargs)


def create_ast_model(
    model_name: str,
    num_labels: int,
    cache_dir: str | None = None,
    local_files_only: bool | None = None,
    hidden_dropout_prob: float | None = None,
    attention_probs_dropout_prob: float | None = None,
    classifier_dropout_prob: float | None = None,
) -> ASTForAudioClassification:
    """Load pretrained AST model configured for multi-label classification with dropout regularization."""
    kwargs = {
        "num_labels": num_labels,
        "cache_dir": cache_dir,
        "problem_type": "multi_label_classification",
        "ignore_mismatched_sizes": True,
    }
    if local_files_only is not None:
        kwargs["local_files_only"] = bool(local_files_only)
    if hidden_dropout_prob is not None:
        kwargs["hidden_dropout_prob"] = float(hidden_dropout_prob)
    if attention_probs_dropout_prob is not None:
        kwargs["attention_probs_dropout_prob"] = float(attention_probs_dropout_prob)

    model = ASTForAudioClassification.from_pretrained(model_name, **kwargs)
    if classifier_dropout_prob is not None:
        apply_classifier_dropout(model, dropout_prob=float(classifier_dropout_prob))
    return model


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


def apply_classifier_dropout(
    model: ASTForAudioClassification,
    dropout_prob: float = 0.3,
) -> None:
    """Add dropout layer before classifier to regularize predictions."""
    try:
        import torch
        import torch.nn as nn

        if not hasattr(model, "classifier"):
            return

        classifier = model.classifier
        if not hasattr(classifier, "dense"):
            return

        # Insert dropout before final dense layer
        old_dense = classifier.dense
        classifier.dropout = nn.Dropout(dropout_prob)
        classifier.dense = old_dense
    except Exception:
        pass


def save_hf_artifacts(model: ASTForAudioClassification, feature_extractor: ASTFeatureExtractor, output_dir: str | Path) -> None:
    """Save Hugging Face-compatible model + feature extractor."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    feature_extractor.save_pretrained(out)
