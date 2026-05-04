"""MotionLens inference, decoding, and decoder-state utilities for training and live apps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    feature_names: list[str] = field(default_factory=list)
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


def causal_hmm_step(
    emission: np.ndarray,
    *,
    log_init_probs: np.ndarray,
    log_trans_probs: np.ndarray,
    previous_posterior: np.ndarray | None,
) -> tuple[int, np.ndarray, str]:
    """Run one causal HMM step and return (argmax_idx, posterior, state_source)."""

    emission_arr = np.asarray(emission, dtype=np.float64)
    emission_arr = np.clip(emission_arr, 1e-12, 1.0)
    emission_sum = float(np.sum(emission_arr))
    if emission_sum <= 0.0:
        emission_arr = np.full(emission_arr.size, 1.0 / max(emission_arr.size, 1), dtype=np.float64)
    else:
        emission_arr = emission_arr / emission_sum

    num_classes = emission_arr.size
    init_probs = np.exp(np.asarray(log_init_probs, dtype=np.float64))
    trans_probs = np.exp(np.asarray(log_trans_probs, dtype=np.float64))

    prev_norm: np.ndarray | None = None
    if previous_posterior is not None:
        candidate = np.asarray(previous_posterior, dtype=np.float64)
        if candidate.shape == (num_classes,) and np.all(np.isfinite(candidate)):
            candidate_sum = float(np.sum(candidate))
            if candidate_sum > 0.0:
                prev_norm = candidate / candidate_sum

    if prev_norm is None:
        prior = init_probs
        state_source = "initialized"
    else:
        prior = prev_norm @ trans_probs
        state_source = "continued"

    posterior = prior * emission_arr
    norm = float(np.sum(posterior))
    if norm <= 0.0:
        posterior = np.full(num_classes, 1.0 / max(num_classes, 1), dtype=np.float64)
    else:
        posterior = posterior / norm

    idx = int(np.argmax(posterior))
    return idx, posterior, state_source


def causal_hmm_decode(
    proba: np.ndarray,
    sequence_groups: list[str],
    log_init_probs: np.ndarray,
    log_trans_probs: np.ndarray,
) -> np.ndarray:
    """Causal decoder suitable for online windows without lookahead."""
    if proba.shape[0] != len(sequence_groups):
        raise ValueError("Probability rows and sequence metadata lengths do not match.")

    n_rows, _ = proba.shape
    out = np.zeros(n_rows, dtype=np.int32)

    prev_group = None
    posterior: np.ndarray | None = None

    for idx in range(n_rows):
        group = sequence_groups[idx]
        prev_for_step = None if group != prev_group else posterior
        pred_idx, posterior, _ = causal_hmm_step(
            proba[idx],
            log_init_probs=log_init_probs,
            log_trans_probs=log_trans_probs,
            previous_posterior=prev_for_step,
        )
        out[idx] = pred_idx
        prev_group = group

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


def extract_decoder_posterior(
    previous_state: dict[str, Any] | None,
    *,
    num_classes: int,
) -> np.ndarray | None:
    """Extract and validate previous posterior from persisted decoder state."""
    if not isinstance(previous_state, dict):
        return None
    raw_prev = previous_state.get("posterior")
    if not isinstance(raw_prev, list) or len(raw_prev) != num_classes:
        return None
    candidate = np.asarray(raw_prev, dtype=np.float64)
    if not np.all(np.isfinite(candidate)):
        return None
    candidate_sum = float(np.sum(candidate))
    if candidate_sum <= 0.0:
        return None
    return candidate / candidate_sum


def decoder_boundary_gap_ns(
    previous_state: dict[str, Any] | None,
    *,
    current_window_start_ns: int | None,
) -> int | None:
    """Return positive boundary gap in ns between previous and current windows, if available."""
    if not isinstance(previous_state, dict) or current_window_start_ns is None:
        return None
    previous_end_ns = previous_state.get("window_end_ns")
    if not isinstance(previous_end_ns, int):
        return None
    gap_ns = int(current_window_start_ns) - int(previous_end_ns)
    if gap_ns <= 0:
        return None
    return gap_ns


def should_reset_decoder_state(
    previous_state: dict[str, Any] | None,
    *,
    current_window_start_ns: int | None,
    boundary_gap_seconds: float,
) -> tuple[bool, float | None]:
    """Return (should_reset, gap_ms) based on a window gap boundary threshold."""
    threshold_seconds = max(0.0, float(boundary_gap_seconds))
    if threshold_seconds <= 0.0:
        return False, None
    gap_ns = decoder_boundary_gap_ns(
        previous_state,
        current_window_start_ns=current_window_start_ns,
    )
    if gap_ns is None:
        return False, None
    threshold_ns = int(round(threshold_seconds * 1_000_000_000.0))
    return gap_ns > threshold_ns, (gap_ns / 1_000_000.0)


def resolve_live_decoder_reset_mode(bundle_meta: dict[str, Any]) -> str:
    """Return supported live decoder reset mode with safe fallback."""
    raw_mode = bundle_meta.get("live_decoder_reset_mode", "session")
    mode = str(raw_mode).strip().lower()
    if mode in {"per-window", "per-gap", "session"}:
        return mode
    return "session"
