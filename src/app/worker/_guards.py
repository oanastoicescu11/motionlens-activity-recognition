"""Guard application orchestration for inference pipeline.

This module provides structured guard application with consistent
type handling and result accumulation, extracted from pipeline.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.inference import (
    apply_low_motion_gate,
    apply_stairs_static_guard,
    apply_stairs_moving_window_guard,
    apply_front_pocket_smooth_walk_guard,
    apply_stairs_hallucination_guard,
    apply_overconfident_stairs_failsafe,
    apply_locomotion_transitions_guard,
    apply_artifact_confidence_calibration,
    apply_pocket_activity_sanity_correction,
)
from core.features.cadence import extract_cadence_and_periodicity
from core.inference.artifact_scoring import compute_generic_artifact_score


def _ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Coerce array to float64, avoiding redundant dtype conversions."""
    if arr.dtype == np.float64:
        return arr
    return np.asarray(arr, dtype=np.float64)


@dataclass
class GuardResult:
    """Result from applying a guard with diagnostic info."""

    emission: np.ndarray
    applied: bool
    info: dict[str, Any] = field(default_factory=dict)

    def merge_info(self, other: dict[str, Any]) -> None:
        """Merge additional info dict into this result."""
        self.info.update(other)


@dataclass
class GuardChainResult:
    """Accumulated results from the full guard chain."""

    emission: np.ndarray
    low_motion_gate_fired: bool
    stairs_guard_applied: bool
    stairs_guard_info: dict[str, Any]
    stairs_moving_guard_applied: bool
    stairs_moving_guard_info: dict[str, Any]
    walk_guard_applied: bool
    walk_guard_info: dict[str, Any]
    hallucination_guard_applied: bool
    hallucination_guard_info: dict[str, Any]
    stairs_failsafe_applied: bool
    stairs_failsafe_info: dict[str, Any]
    locomotion_transitions_guard_applied: bool
    locomotion_transitions_guard_reason: str
    artifact_confidence_calibrated: bool
    artifact_confidence_blend: float
    pocket_sanity_corrected: bool
    pocket_sanity_reason: str
    cadence_spm: float
    periodicity_strength: float
    artifact_score: float
    artifact_info: dict[str, Any]
    metric_history: dict[str, Any]


def apply_pre_decode_guards(
    feature_row: np.ndarray,
    emission: np.ndarray,
    label_order: list[str],
    placement_labels: list[str],
    placement_label: str,
    previous_state: dict[str, Any] | None,
) -> tuple[np.ndarray, bool, dict[str, Any], bool, dict[str, Any], bool, dict[str, Any], bool, dict[str, Any]]:
    """Apply guards that run BEFORE HMM decoding.

    Returns (emission, low_motion_gate_fired, low_motion_info,
            stairs_guard_applied, stairs_guard_info,
            stairs_moving_guard_applied, stairs_moving_guard_info,
            walk_guard_applied, walk_guard_info).
    """
    emission = _ensure_float64(emission)

    # 1. Low motion gate
    emission, low_motion_gate_fired = apply_low_motion_gate(
        feature_row=feature_row,
        emission=emission,
        label_order=label_order,
        placement_labels=placement_labels,
        placement_label=placement_label,
    )

    # 2. Stairs static guard
    emission, stairs_guard_applied, stairs_guard_info = apply_stairs_static_guard(
        feature_row=feature_row,
        emission=emission,
        label_order=label_order,
        placement_labels=placement_labels,
    )

    # 3. Stairs moving window guard
    emission, stairs_moving_guard_applied, stairs_moving_guard_info = apply_stairs_moving_window_guard(
        feature_row=feature_row,
        emission=emission,
        label_order=label_order,
        placement_labels=placement_labels,
        placement_label=placement_label,
        previous_state=previous_state,
    )

    # 4. Walk guard (front pocket smooth walk bias)
    emission, walk_guard_applied, walk_guard_info = apply_front_pocket_smooth_walk_guard(
        feature_row=feature_row,
        emission=emission,
        label_order=label_order,
        placement_labels=placement_labels,
        placement_label=placement_label,
    )

    # 5. Hallucination guard
    emission, hallucination_guard_applied, hallucination_guard_info = apply_stairs_hallucination_guard(
        feature_row=feature_row,
        emission=emission,
        label_order=label_order,
        placement_labels=placement_labels,
        placement_label=placement_label,
    )

    return (
        emission,
        low_motion_gate_fired,
        {},
        stairs_guard_applied,
        stairs_guard_info,
        stairs_moving_guard_applied,
        stairs_moving_guard_info,
        walk_guard_applied,
        walk_guard_info,
        hallucination_guard_applied,
        hallucination_guard_info,
    )


def apply_post_decode_guards(
    emission: np.ndarray,
    decoded_proba: np.ndarray,
    feature_row: np.ndarray,
    label_order: list[str],
    placement_labels: list[str],
    placement_label: str,
    raw_label: str,
    raw_confidence: float,
    cadence_spm: float,
    periodicity_strength: float,
    artifact_score: float,
    artifact_info: dict[str, Any],
    previous_state: dict[str, Any] | None,
) -> tuple[
    np.ndarray,
    bool,
    dict[str, Any],
    bool,
    str,
    bool,
    float,
    bool,
    str,
]:
    """Apply guards that run AFTER HMM decoding.

    Returns (decoded_proba, stairs_failsafe_applied, stairs_failsafe_info,
            locomotion_transitions_guard_applied, locomotion_transitions_reason,
            artifact_confidence_calibrated, artifact_confidence_blend,
            pocket_sanity_corrected, pocket_sanity_reason).
    """
    emission = _ensure_float64(emission)
    decoded_proba = _ensure_float64(decoded_proba)

    # 1. Stairs failsafe
    decoded_proba, stairs_failsafe_applied, stairs_failsafe_info = apply_overconfident_stairs_failsafe(
        emission=emission,
        decoded_proba=decoded_proba,
        label_order=label_order,
        placement_label=placement_label,
    )

    # 2. Locomotion transitions guard
    decoded_proba, locomotion_transitions_guard_applied, locomotion_transitions_reason = apply_locomotion_transitions_guard(
        feature_row=feature_row,
        emission=decoded_proba,
        label_order=label_order,
        placement_labels=placement_labels,
        previous_state=previous_state,
    )

    # 3. Artifact confidence calibration
    decoded_proba, artifact_confidence_calibrated, artifact_confidence_blend = apply_artifact_confidence_calibration(
        decoded_proba,
        artifact_score=artifact_score,
        placement_label=placement_label,
    )

    # 4. Pocket sanity correction
    decoded_proba, pocket_sanity_corrected, pocket_sanity_reason = apply_pocket_activity_sanity_correction(
        decoded_proba,
        label_order=label_order,
        placement_label=placement_label,
        cadence_spm=cadence_spm,
        periodicity_strength=periodicity_strength,
        artifact_score=artifact_score,
        raw_label=raw_label,
        raw_confidence=raw_confidence,
    )

    return (
        decoded_proba,
        stairs_failsafe_applied,
        stairs_failsafe_info,
        locomotion_transitions_guard_applied,
        locomotion_transitions_reason,
        artifact_confidence_calibrated,
        artifact_confidence_blend,
        pocket_sanity_corrected,
        pocket_sanity_reason,
    )


def compute_guard_auxiliary(
    feature_row: np.ndarray,
    placement_labels: list[str],
    previous_state: dict[str, Any] | None,
) -> tuple[float, float, float, dict[str, Any], dict[str, Any]]:
    """Compute auxiliary values needed for guards: cadence, artifact score, etc.

    Returns (cadence_spm, periodicity_strength, artifact_score, artifact_info, metric_history).
    """
    cadence_spm, periodicity_strength = extract_cadence_and_periodicity(
        feature_row=feature_row,
        placement_labels=placement_labels,
    )
    artifact_score, artifact_info, metric_history = compute_generic_artifact_score(
        feature_row=feature_row,
        placement_labels=placement_labels,
        previous_state=previous_state,
    )
    return cadence_spm, periodicity_strength, artifact_score, artifact_info, metric_history
