"""Shared inference utilities for training-time evaluation and live app predictions."""

from .model import (
    InferenceArtifacts,
    apply_static_specialist_refinement,
    causal_hmm_step,
    causal_hmm_decode,
    predict_hierarchical_proba,
    extract_decoder_posterior,
    decoder_boundary_gap_ns,
    should_reset_decoder_state,
    resolve_live_decoder_reset_mode,
)
from .artifact_scoring import (
    FRONT_POCKET_LABEL,
    POCKET_ARTIFACT_SCORE_THRESHOLD,
    compute_loose_pocket_artifact_score,
    compute_generic_artifact_score,
    apply_artifact_confidence_calibration,
)
from .guards import (
    LOCOMOTION_LABELS,
    STATIC_LABELS,
    LOW_MOTION_ENERGY_THRESHOLD,
    apply_low_motion_gate,
    apply_locomotion_transitions_guard,
    should_reset_on_low_motion,
    apply_stairs_static_guard,
    apply_stairs_moving_window_guard,
    apply_overconfident_stairs_failsafe,
    apply_front_pocket_smooth_walk_guard,
    apply_stairs_hallucination_guard,
    apply_pocket_activity_sanity_correction,
)

__all__ = [
    # model
    "InferenceArtifacts",
    "apply_static_specialist_refinement",
    "causal_hmm_step",
    "causal_hmm_decode",
    "predict_hierarchical_proba",
    "extract_decoder_posterior",
    "decoder_boundary_gap_ns",
    "should_reset_decoder_state",
    "resolve_live_decoder_reset_mode",
    # artifact_scoring
    "FRONT_POCKET_LABEL",
    "POCKET_ARTIFACT_SCORE_THRESHOLD",
    "compute_loose_pocket_artifact_score",
    "compute_generic_artifact_score",
    "apply_artifact_confidence_calibration",
    # guards
    "LOCOMOTION_LABELS",
    "STATIC_LABELS",
    "LOW_MOTION_ENERGY_THRESHOLD",
    "apply_low_motion_gate",
    "apply_locomotion_transitions_guard",
    "should_reset_on_low_motion",
    "apply_stairs_static_guard",
    "apply_stairs_moving_window_guard",
    "apply_overconfident_stairs_failsafe",
    "apply_front_pocket_smooth_walk_guard",
    "apply_stairs_hallucination_guard",
    "apply_pocket_activity_sanity_correction",
]
