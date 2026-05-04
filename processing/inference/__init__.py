"""Training-specific inference utilities — batch decode and parity inference."""

from __future__ import annotations

import numpy as np

from core.inference.model import (
    COARSE_LABEL_ORDER,
    FINE_TO_COARSE,
    InferenceArtifacts,
    apply_static_specialist_refinement,
    causal_hmm_decode,
    predict_hierarchical_proba,
)


def decode_predictions(
    decoder: str,
    y_pred_raw: np.ndarray,
    y_proba_raw: np.ndarray,
    sequence_groups: list[str],
    log_init_probs: np.ndarray,
    log_trans_probs: np.ndarray,
) -> np.ndarray:
    """Decode raw predictions with a supported causal strategy."""
    if decoder == "none":
        return y_pred_raw.copy()
    if decoder == "causal-hmm":
        return causal_hmm_decode(
            proba=y_proba_raw,
            sequence_groups=sequence_groups,
            log_init_probs=log_init_probs,
            log_trans_probs=log_trans_probs,
        )
    raise ValueError(f"Unsupported temporal decoder: {decoder}")


def predict_with_artifacts(
    artifacts: InferenceArtifacts,
    x_values: np.ndarray,
    sequence_groups: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run parity inference on window features for live or uploaded sessions.

    Returns raw argmax predictions, decoded predictions, and refined probabilities.
    """
    if artifacts.use_hierarchical:
        if artifacts.coarse_model is None:
            raise ValueError("coarse_model is required when use_hierarchical is enabled.")
        fine_models = artifacts.fine_models or {}
        y_proba_raw = predict_hierarchical_proba(
            x_values=x_values,
            label_order=artifacts.label_order,
            coarse_model=artifacts.coarse_model,
            fine_models=fine_models,
        )
        static_idx = [
            idx for idx, label in enumerate(artifacts.label_order)
            if label in set(artifacts.static_specialist_labels or [])
        ]
        gate_scores = np.sum(y_proba_raw[:, static_idx], axis=1) if static_idx else np.zeros(y_proba_raw.shape[0])
    else:
        y_proba_raw = np.asarray(artifacts.base_model.predict_proba(x_values), dtype=np.float64)
        static_idx = [
            idx for idx, label in enumerate(artifacts.label_order)
            if label in set(artifacts.static_specialist_labels or [])
        ]
        gate_scores = np.sum(y_proba_raw[:, static_idx], axis=1) if static_idx else np.zeros(y_proba_raw.shape[0])

    y_proba_refined = apply_static_specialist_refinement(
        y_proba=y_proba_raw,
        x_values=x_values,
        label_order=artifacts.label_order,
        static_model=artifacts.static_specialist_model,
        static_labels=artifacts.static_specialist_labels or [],
        gate_scores=gate_scores,
        gate_threshold=artifacts.static_specialist_threshold,
        blend=artifacts.static_specialist_blend,
    )
    y_pred_raw = np.argmax(y_proba_refined, axis=1).astype(np.int32)
    y_pred = decode_predictions(
        decoder=artifacts.temporal_decoder,
        y_pred_raw=y_pred_raw,
        y_proba_raw=y_proba_refined,
        sequence_groups=sequence_groups,
        log_init_probs=artifacts.log_init_probs,
        log_trans_probs=artifacts.log_trans_probs,
    )
    return y_pred_raw, y_pred, y_proba_refined
