"""Reusable MotionLens inference and post-processing utilities for training and live apps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COARSE_LABEL_ORDER = ("locomotion", "static", "transition", "other")
FINE_TO_COARSE = {
    "walk": "locomotion",
    "run": "locomotion",
    "stairs": "locomotion",
    "sit": "static",
    "stand": "static",
    "lay": "static",
    "transitions": "transition",
    "locomotion-other": "other",
}


@dataclass
class InferenceArtifacts:
    """Container for everything needed to run parity inference on new windows."""

    base_model: object
    label_order: list[str]
    temporal_decoder: str
    log_init_probs: np.ndarray
    log_trans_probs: np.ndarray
    use_hierarchical: bool = False
    coarse_model: object | None = None
    fine_models: dict[str, object] | None = None
    static_specialist_model: object | None = None
    static_specialist_labels: list[str] | None = None
    static_specialist_threshold: float = 0.35
    static_specialist_blend: float = 0.6


def apply_static_specialist_refinement(
    y_proba: np.ndarray,
    x_values: np.ndarray,
    label_order: list[str],
    static_model,
    static_labels: list[str],
    gate_scores: np.ndarray,
    gate_threshold: float,
    blend: float,
) -> np.ndarray:
    """Refine static-class probabilities while preserving total static mass."""
    if static_model is None or not static_labels:
        return y_proba

    index_by_label = {label: idx for idx, label in enumerate(label_order)}
    specialist_indices = [index_by_label[label] for label in static_labels if label in index_by_label]
    if not specialist_indices:
        return y_proba

    gate_threshold = float(np.clip(gate_threshold, 0.0, 1.0))
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0.0:
        return y_proba

    out = y_proba.copy()
    apply_mask = gate_scores >= gate_threshold
    if not np.any(apply_mask):
        return out

    specialist_proba = np.asarray(static_model.predict_proba(x_values[apply_mask]), dtype=np.float64)
    apply_rows = np.flatnonzero(apply_mask)

    for row_local, row_idx in enumerate(apply_rows):
        base_static = out[row_idx, specialist_indices]
        spec_static = specialist_proba[row_local, : len(specialist_indices)]
        mixed_static = (1.0 - blend) * base_static + blend * spec_static

        static_mass_before = float(np.sum(base_static))
        static_mass_after = float(np.sum(mixed_static))
        if static_mass_after > 1e-12 and static_mass_before > 1e-12:
            mixed_static *= static_mass_before / static_mass_after

        out[row_idx, specialist_indices] = mixed_static
        row_sum = float(np.sum(out[row_idx]))
        if row_sum > 0.0:
            out[row_idx] /= row_sum

    return out


def causal_hmm_decode(
    proba: np.ndarray,
    sequence_groups: list[str],
    log_init_probs: np.ndarray,
    log_trans_probs: np.ndarray,
) -> np.ndarray:
    """Causal decoder suitable for online windows without lookahead."""
    if proba.shape[0] != len(sequence_groups):
        raise ValueError("Probability rows and sequence metadata lengths do not match.")

    n_rows, num_classes = proba.shape
    out = np.zeros(n_rows, dtype=np.int32)
    init_probs = np.exp(log_init_probs)
    trans_probs = np.exp(log_trans_probs)

    prev_group = None
    posterior = np.zeros(num_classes, dtype=np.float64)

    for idx in range(n_rows):
        emission = np.clip(proba[idx], 1e-12, 1.0)
        group = sequence_groups[idx]

        if group != prev_group:
            posterior = init_probs * emission
            prev_group = group
        else:
            prior = posterior @ trans_probs
            posterior = prior * emission

        norm = float(np.sum(posterior))
        if norm <= 0.0:
            posterior = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
        else:
            posterior = posterior / norm

        out[idx] = int(np.argmax(posterior))

    return out


def predict_hierarchical_proba(
    x_values: np.ndarray,
    label_order: list[str],
    coarse_model,
    fine_models: dict[str, object],
) -> np.ndarray:
    """Compute full-class probabilities from coarse and optional fine experts."""
    n_rows = x_values.shape[0]
    full = np.zeros((n_rows, len(label_order)), dtype=np.float64)
    coarse_proba = np.asarray(coarse_model.predict_proba(x_values), dtype=np.float64)

    for coarse_idx, coarse_label in enumerate(COARSE_LABEL_ORDER):
        fine_indices = [idx for idx, label in enumerate(label_order) if FINE_TO_COARSE.get(label) == coarse_label]
        if not fine_indices:
            continue
        coarse_slice = coarse_proba[:, coarse_idx]
        fine_model = fine_models.get(coarse_label)
        if len(fine_indices) == 1:
            full[:, fine_indices[0]] = coarse_slice
            continue
        if fine_model is None:
            uniform = coarse_slice / len(fine_indices)
            for fine_idx in fine_indices:
                full[:, fine_idx] = uniform
            continue

        fine_proba = np.asarray(fine_model.predict_proba(x_values), dtype=np.float64)
        for local_idx, fine_idx in enumerate(fine_indices):
            full[:, fine_idx] = coarse_slice * fine_proba[:, local_idx]

    row_sum = np.sum(full, axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
    return full / row_sum


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
