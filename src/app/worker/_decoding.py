"""HMM decoding and decoder state management for inference pipeline.

This module handles temporal decoding and state persistence,
extracted from pipeline.py.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from core.inference import (
    InferenceArtifacts,
    causal_hmm_step,
    extract_decoder_posterior,
    should_reset_on_low_motion,
)


def _ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Coerce array to float64, avoiding redundant dtype conversions."""
    if arr.dtype == np.float64:
        return arr
    return np.asarray(arr, dtype=np.float64)


def _normalize_emission(emission: np.ndarray) -> np.ndarray:
    """Normalize emission probabilities to sum to 1.0."""
    emission = _ensure_float64(emission)
    emission = np.clip(emission, 1e-12, 1.0)
    emission_sum = float(np.sum(emission))
    if emission_sum <= 0.0:
        return np.full(emission.size, 1.0 / max(emission.size, 1), dtype=np.float64)
    return emission / emission_sum


def should_reset(
    previous_state: dict[str, Any] | None,
    label_order: list[str],
    low_motion_gate_fired: bool,
) -> tuple[bool, str | None]:
    """Determine if decoder state should be reset.

    Returns (sequence_reset, reset_reason).
    """
    sequence_reset = False
    reset_reason: str | None = None

    if low_motion_gate_fired:
        if should_reset_on_low_motion(previous_state, label_order=label_order):
            sequence_reset = True
            reset_reason = "low-motion-gate"

    return sequence_reset, reset_reason


def run_hmm_decode(
    emission: np.ndarray,
    artifacts: InferenceArtifacts,
    previous_state: dict[str, Any] | None,
    sequence_reset: bool,
    reset_reason: str | None,
) -> tuple[np.ndarray, int, str]:
    """Run HMM decoding step.

    Returns (decoded_proba, pred_idx, decoder_state_source).
    """
    emission = _normalize_emission(emission)

    if artifacts.temporal_decoder == "none":
        decoded_proba = emission
        pred_idx = int(np.argmax(decoded_proba))
        decoder_state_source = "disabled"
    elif artifacts.temporal_decoder == "causal-hmm":
        prev_posterior = extract_decoder_posterior(
            previous_state,
            num_classes=emission.size,
        )
        pred_idx, decoded_proba, decoder_state_source = causal_hmm_step(
            emission,
            log_init_probs=_ensure_float64(np.asarray(artifacts.log_init_probs)),
            log_trans_probs=_ensure_float64(np.asarray(artifacts.log_trans_probs)),
            previous_posterior=prev_posterior,
        )
        if sequence_reset and decoder_state_source == "initialized":
            decoder_state_source = "reset-low-motion"
    else:
        decoded_proba = emission
        pred_idx = int(np.argmax(decoded_proba))
        decoder_state_source = "unsupported-decoder"

    return decoded_proba, pred_idx, decoder_state_source


def build_decoder_state_source_suffix(
    decoder_state_source: str,
    stairs_failsafe_applied: bool,
    locomotion_transitions_guard_applied: bool,
    artifact_confidence_calibrated: bool,
    pocket_sanity_corrected: bool,
) -> str:
    """Build the decoder_state_source suffix with applied corrections."""
    source = decoder_state_source
    if stairs_failsafe_applied:
        source = f"{source}+stairs-failsafe"
    if locomotion_transitions_guard_applied:
        source = f"{source}+locomotion-transitions"
    if artifact_confidence_calibrated:
        source = f"{source}+artifact-calibration"
    if pocket_sanity_corrected:
        source = f"{source}+pocket-sanity"
    return source


def save_decoder_state(
    store: Any,
    session_id: str,
    decoded_proba: np.ndarray,
    diagnostics: dict[str, Any],
    metric_history: dict[str, Any],
) -> None:
    """Persist decoder state to store after successful inference."""
    if session_id:
        store.set_decoder_state(
            session_id,
            {
                "posterior": decoded_proba.tolist(),
                "window_end_ns": diagnostics.get("window_end_ns"),
                "metric_history": metric_history,
                "updated_at_ns": int(time.time_ns()),
            },
        )


