"""Response serialization for inference pipeline using Pydantic models.

This module defines typed snapshot models for inference results,
replacing raw dict returns with validated Pydantic objects.

Use .model_dump(mode="json") to serialize for API/storage boundaries.

Example usage:
    snapshot = SuccessSnapshot(session_id="s1", message_id=1, activity="walk", ...)
    data = snapshot.model_dump(mode="json")

    error = ModelLoadErrorSnapshot.from_error(
        session_id="s1", message_id=1, error="file not found"
    )
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Self, Union

from pydantic import BaseModel, Field


# Model window size (used as default)
MIN_WINDOW_SAMPLES = 128


# ----------------------------------------------------------------------
# Base Snapshot
# ----------------------------------------------------------------------


class BaseSnapshot(BaseModel):
    """Base class for all inference snapshots."""

    session_id: str
    message_id: int
    updated_at_ns: int = Field(default_factory=lambda: int(time.time_ns()))
    snapshot_type: str = "base"

    model_config = {"extra": "allow"}


# ----------------------------------------------------------------------
# Top Prediction
# ----------------------------------------------------------------------


class TopPrediction(BaseModel):
    """One of up to 3 top predicted activities with confidence."""

    label: str
    confidence: float


# ----------------------------------------------------------------------
# Success Snapshot
# ----------------------------------------------------------------------


class SuccessSnapshot(BaseSnapshot):
    """Full inference result with all guard diagnostics and features."""

    snapshot_type: Literal["success"] = "success"
    status: Literal["success"] = "success"
    activity: str
    confidence: float
    raw_activity: str
    raw_confidence: float
    top_predictions: list[TopPrediction] = Field(default_factory=list)
    sample_count: int = 0
    window_raw_samples: int | None = None
    window_samples_used: int = MIN_WINDOW_SAMPLES
    window_start_ns: int | None = None
    window_end_ns: int | None = None
    hampel_replaced_points: int = 0
    placement_label: str = "front_pocket"
    placement_source: str = "pocket_only"
    cadence_spm: float = 0.0
    periodicity_strength: float = 0.0
    loose_pocket_artifact_score: float = 0.0
    artifact_confidence_calibrated: bool = False
    artifact_confidence_blend: float = 0.0
    pocket_sanity_corrected: bool = False
    pocket_sanity_reason: str = "none"
    # Artifact sub-scores
    artifact_hf_ratio: float = 0.0
    artifact_adaptive_enabled: bool = False
    artifact_base_score: float = 0.0
    artifact_adaptive_score: float = 0.0
    artifact_z_jerk: float = 0.0
    artifact_z_hf: float = 0.0
    artifact_z_entropy: float = 0.0
    artifact_z_gravity: float = 0.0
    artifact_z_periodicity: float = 0.0
    artifact_z_dom_jitter: float = 0.0
    artifact_jerk_term: float = 0.0
    artifact_hf_term: float = 0.0
    artifact_entropy_term: float = 0.0
    artifact_gravity_term: float = 0.0
    artifact_periodicity_term: float = 0.0
    artifact_dom_jitter_term: float = 0.0
    # Decoder
    decoder: str = "none"
    decoder_state_source: str = "disabled"
    live_decoder_reset_mode: str = "session"
    live_sequence_boundary_gap_seconds: float = 3.0
    decoder_sequence_reset: bool = False
    decoder_sequence_gap_ms: float | None = None
    # Guard flags
    low_motion_gate_fired: bool = False
    locomotion_transitions_guard_applied: bool = False
    locomotion_transitions_guard_reason: str = "none"
    stairs_guard_applied: bool = False
    stairs_guard_energy: float = 0.0
    stairs_guard_jerk: float = 0.0
    stairs_guard_dom_freq: float = 0.0
    stairs_moving_guard_applied: bool = False
    stairs_moving_guard_energy: float = 0.0
    stairs_moving_guard_dom_freq: float = 0.0
    stairs_moving_guard_prev_stairs: float = 0.0
    stairs_moving_guard_smoothed_stairs: float = 0.0
    stairs_moving_guard_walkrun_mass: float = 0.0
    walk_guard_applied: bool = False
    walk_guard_energy: float = 0.0
    walk_guard_dom_freq: float = 0.0
    walk_guard_jerk: float = 0.0
    walk_guard_walk: float = 0.0
    walk_guard_run: float = 0.0
    walk_guard_stairs: float = 0.0
    hallucination_guard_applied: bool = False
    hallucination_guard_energy: float = 0.0
    hallucination_guard_jerk: float = 0.0
    hallucination_guard_stairs: float = 0.0
    stairs_failsafe_applied: bool = False
    stairs_failsafe_raw_stairs: float = 0.0
    stairs_failsafe_raw_walkrun: float = 0.0
    stairs_failsafe_decoded_stairs: float = 0.0
    stairs_failsafe_decoded_walkrun: float = 0.0


# ----------------------------------------------------------------------
# No-Data Snapshot
# ----------------------------------------------------------------------


class NoDataSnapshot(BaseSnapshot):
    """No raw accelerometer points available for inference."""

    snapshot_type: Literal["no_data"] = "no_data"
    status: Literal["success"] = "success"
    activity: Literal["no-data"] = "no-data"
    confidence: Literal[0.0] = 0.0


# ----------------------------------------------------------------------
# Queued Snapshot
# ----------------------------------------------------------------------


class QueuedSnapshot(BaseSnapshot):
    """Window not ready — insufficient samples or time span."""

    snapshot_type: Literal["queued"] = "queued"
    status: Literal["queued"] = "queued"
    reason: str = "insufficient-data"
    sample_count: int = 0
    samples_needed: int = MIN_WINDOW_SAMPLES
    window_span_ms: float | None = None
    window_required_ms: float | None = None

    @classmethod
    def from_diagnostics(
        cls,
        *,
        session_id: str,
        message_id: int,
        diagnostics: dict[str, Any],
        points: list[Any],
    ) -> Self:
        """Build from diagnostics dict extracted during resampling."""
        return cls(
            session_id=session_id,
            message_id=message_id,
            reason=diagnostics.get("reason", "insufficient-data"),
            sample_count=diagnostics.get("sample_count", len(points)),
            samples_needed=diagnostics.get("samples_needed", MIN_WINDOW_SAMPLES),
            window_span_ms=diagnostics.get("window_span_ms"),
            window_required_ms=diagnostics.get("window_required_ms"),
        )


# ----------------------------------------------------------------------
# Queued Processed Snapshot
# ----------------------------------------------------------------------


class QueuedProcessedSnapshot(BaseSnapshot):
    """Task was processed but store was not available — placeholder response."""

    snapshot_type: Literal["queued_processed"] = "queued_processed"
    status: Literal["queued-processed"] = "queued-processed"


# ----------------------------------------------------------------------
# No-Features Snapshot
# ----------------------------------------------------------------------


class NoFeaturesSnapshot(BaseSnapshot):
    """Feature extraction returned empty result."""

    snapshot_type: Literal["no_features"] = "no_features"
    status: Literal["success"] = "success"
    activity: Literal["no-features"] = "no-features"
    confidence: Literal[0.0] = 0.0


# ----------------------------------------------------------------------
# Error Snapshots
# ----------------------------------------------------------------------


class ErrorSnapshot(BaseSnapshot):
    """Base class for all error snapshots."""

    status: Literal["error"] = "error"
    error: str


class PredictionErrorSnapshot(ErrorSnapshot):
    """Generic prediction pipeline failure."""

    snapshot_type: Literal["prediction_error"] = "prediction_error"
    error_code: Literal["prediction_failed"] = "prediction_failed"


class ModelLoadErrorSnapshot(ErrorSnapshot):
    """Model artifact loading failed."""

    snapshot_type: Literal["model_load_error"] = "model_load_error"
    error_code: Literal["model_load_failed"] = "model_load_failed"

    @classmethod
    def from_error(
        cls,
        *,
        session_id: str,
        message_id: int,
        error: Exception | str,
    ) -> Self:
        """Build from an exception or error string."""
        error_str = str(error)
        return cls(
            session_id=session_id,
            message_id=message_id,
            error=f"Failed to load model artifacts: {error_str}",
        )


class FeatureErrorSnapshot(ErrorSnapshot):
    """Feature extraction raised an exception."""

    snapshot_type: Literal["feature_error"] = "feature_error"
    error_code: Literal["feature_extraction_failed"] = "feature_extraction_failed"

    @classmethod
    def from_error(
        cls,
        *,
        session_id: str,
        message_id: int,
        error: Exception | str,
    ) -> Self:
        """Build from an exception or error string."""
        error_str = str(error)
        return cls(
            session_id=session_id,
            message_id=message_id,
            error=f"Feature extraction failed: {error_str}",
        )


# ----------------------------------------------------------------------
# Union Type with Discriminator
# ----------------------------------------------------------------------


InferenceSnapshot = Annotated[
    Union[
        SuccessSnapshot,
        NoDataSnapshot,
        QueuedSnapshot,
        QueuedProcessedSnapshot,
        NoFeaturesSnapshot,
        PredictionErrorSnapshot,
        ModelLoadErrorSnapshot,
        FeatureErrorSnapshot,
    ],
    Field(discriminator="snapshot_type"),
]


# ----------------------------------------------------------------------
# Legacy Builder Functions (for backward compatibility)
# Kept for gradual migration — prefer direct constructors or class methods
# ----------------------------------------------------------------------


def build_success_snapshot(
    *,
    session_id: str,
    message_id: int,
    diagnostics: dict[str, Any],
    points: list,
    # Raw prediction
    raw_label: str,
    raw_confidence: float,
    # Decoded prediction
    pred_label: str,
    pred_confidence: float,
    decoded_proba: np.ndarray,
    label_order: list[str],
    # Guard results
    low_motion_gate_fired: bool,
    stairs_guard_applied: bool,
    stairs_guard_info: dict[str, Any],
    stairs_moving_guard_applied: bool,
    stairs_moving_guard_info: dict[str, Any],
    walk_guard_applied: bool,
    walk_guard_info: dict[str, Any],
    hallucination_guard_applied: bool,
    hallucination_guard_info: dict[str, Any],
    stairs_failsafe_applied: bool,
    stairs_failsafe_info: dict[str, Any],
    locomotion_transitions_guard_applied: bool,
    locomotion_transitions_guard_reason: str,
    artifact_confidence_calibrated: bool,
    artifact_confidence_blend: float,
    pocket_sanity_corrected: bool,
    pocket_sanity_reason: str,
    # Auxiliary
    cadence_spm: float,
    periodicity_strength: float,
    artifact_score: float,
    artifact_info: dict[str, Any],
    hampel_replaced: int,
    # Placement
    placement_label: str,
    placement_source: str,
    # Decoder
    decoder: str,
    decoder_state_source: str,
    reset_mode: str,
    boundary_gap_seconds: float,
    sequence_reset: bool,
    sequence_gap_ms: float | None,
) -> SuccessSnapshot:
    """Build a SuccessSnapshot Pydantic model (legacy builder)."""
    import numpy as np

    top_3_indices = np.argsort(decoded_proba)[-3:][::-1]
    top_predictions = [
        TopPrediction(
            label=label_order[int(idx)],
            confidence=float(decoded_proba[int(idx)]),
        )
        for idx in top_3_indices
    ]

    return SuccessSnapshot(
        session_id=session_id,
        message_id=message_id,
        activity=pred_label,
        confidence=pred_confidence,
        raw_activity=raw_label,
        raw_confidence=raw_confidence,
        top_predictions=top_predictions,
        sample_count=diagnostics.get("sample_count", len(points)),
        window_raw_samples=diagnostics.get("window_raw_samples"),
        window_samples_used=diagnostics.get("window_samples_used", MIN_WINDOW_SAMPLES),
        window_start_ns=diagnostics.get("window_start_ns"),
        window_end_ns=diagnostics.get("window_end_ns"),
        hampel_replaced_points=hampel_replaced,
        placement_label=placement_label,
        placement_source=placement_source,
        cadence_spm=cadence_spm,
        periodicity_strength=periodicity_strength,
        loose_pocket_artifact_score=artifact_score,
        artifact_confidence_calibrated=artifact_confidence_calibrated,
        artifact_confidence_blend=artifact_confidence_blend,
        pocket_sanity_corrected=pocket_sanity_corrected,
        pocket_sanity_reason=pocket_sanity_reason,
        artifact_hf_ratio=artifact_info.get("artifact_hf_ratio", 0.0),
        artifact_adaptive_enabled=bool(artifact_info.get("artifact_adaptive_enabled", 0.0) >= 0.5),
        artifact_base_score=artifact_info.get("artifact_base_score", 0.0),
        artifact_adaptive_score=artifact_info.get("artifact_adaptive_score", 0.0),
        artifact_z_jerk=artifact_info.get("artifact_z_jerk", 0.0),
        artifact_z_hf=artifact_info.get("artifact_z_hf", 0.0),
        artifact_z_entropy=artifact_info.get("artifact_z_entropy", 0.0),
        artifact_z_gravity=artifact_info.get("artifact_z_gravity", 0.0),
        artifact_z_periodicity=artifact_info.get("artifact_z_periodicity", 0.0),
        artifact_z_dom_jitter=artifact_info.get("artifact_z_dom_jitter", 0.0),
        artifact_jerk_term=artifact_info.get("artifact_jerk_term", 0.0),
        artifact_hf_term=artifact_info.get("artifact_hf_term", 0.0),
        artifact_entropy_term=artifact_info.get("artifact_entropy_term", 0.0),
        artifact_gravity_term=artifact_info.get("artifact_gravity_term", 0.0),
        artifact_periodicity_term=artifact_info.get("artifact_periodicity_term", 0.0),
        artifact_dom_jitter_term=artifact_info.get("artifact_dom_jitter_term", 0.0),
        decoder=decoder,
        decoder_state_source=decoder_state_source,
        live_decoder_reset_mode=reset_mode,
        live_sequence_boundary_gap_seconds=boundary_gap_seconds,
        decoder_sequence_reset=sequence_reset,
        decoder_sequence_gap_ms=sequence_gap_ms,
        low_motion_gate_fired=low_motion_gate_fired,
        locomotion_transitions_guard_applied=locomotion_transitions_guard_applied,
        locomotion_transitions_guard_reason=locomotion_transitions_guard_reason,
        stairs_guard_applied=stairs_guard_applied,
        stairs_guard_energy=stairs_guard_info.get("stairs_guard_energy", 0.0),
        stairs_guard_jerk=stairs_guard_info.get("stairs_guard_jerk", 0.0),
        stairs_guard_dom_freq=stairs_guard_info.get("stairs_guard_dom_freq", 0.0),
        stairs_moving_guard_applied=stairs_moving_guard_applied,
        stairs_moving_guard_energy=stairs_moving_guard_info.get("stairs_moving_guard_energy", 0.0),
        stairs_moving_guard_dom_freq=stairs_moving_guard_info.get("stairs_moving_guard_dom_freq", 0.0),
        stairs_moving_guard_prev_stairs=stairs_moving_guard_info.get("stairs_moving_guard_prev_stairs", 0.0),
        stairs_moving_guard_smoothed_stairs=stairs_moving_guard_info.get("stairs_moving_guard_smoothed_stairs", 0.0),
        stairs_moving_guard_walkrun_mass=stairs_moving_guard_info.get("stairs_moving_guard_walkrun_mass", 0.0),
        walk_guard_applied=walk_guard_applied,
        walk_guard_energy=walk_guard_info.get("walk_guard_energy", 0.0),
        walk_guard_dom_freq=walk_guard_info.get("walk_guard_dom_freq", 0.0),
        walk_guard_jerk=walk_guard_info.get("walk_guard_jerk", 0.0),
        walk_guard_walk=walk_guard_info.get("walk_guard_walk", 0.0),
        walk_guard_run=walk_guard_info.get("walk_guard_run", 0.0),
        walk_guard_stairs=walk_guard_info.get("walk_guard_stairs", 0.0),
        hallucination_guard_applied=hallucination_guard_applied,
        hallucination_guard_energy=hallucination_guard_info.get("hallucination_guard_energy", 0.0),
        hallucination_guard_jerk=hallucination_guard_info.get("hallucination_guard_jerk", 0.0),
        hallucination_guard_stairs=hallucination_guard_info.get("hallucination_guard_stairs", 0.0),
        stairs_failsafe_applied=stairs_failsafe_applied,
        stairs_failsafe_raw_stairs=stairs_failsafe_info.get("stairs_failsafe_raw_stairs", 0.0),
        stairs_failsafe_raw_walkrun=stairs_failsafe_info.get("stairs_failsafe_raw_walkrun", 0.0),
        stairs_failsafe_decoded_stairs=stairs_failsafe_info.get("stairs_failsafe_decoded_stairs", 0.0),
        stairs_failsafe_decoded_walkrun=stairs_failsafe_info.get("stairs_failsafe_decoded_walkrun", 0.0),
    )
