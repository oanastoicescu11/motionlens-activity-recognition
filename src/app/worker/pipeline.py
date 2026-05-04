"""Inference worker pipeline.

Loads the FeaturePack2 model or pocket_only model and orchestrates live feature extraction,
guard application, HMM decoding (if applicable), and result serialization for one queued task.

The main process_task() function delegates to specialized modules:
- _steps.py: Resampling, feature extraction, model inference
- _guards.py: Guard application orchestration
- _decoding.py: HMM decoding and decoder state management
- _serialization.py: Response building
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from app.insights import compute_live_window_insights

# Import from extracted modules (relative imports within the worker package)
from ._steps import (
    MIN_WINDOW_SAMPLES,
    DEFAULT_PLACEMENT_LABEL,
    _ensure_artifacts_loaded,
    _resample_recent_window,
    _compute_refined_proba_row,
    extract_features,
)
from ._guards import (
    apply_pre_decode_guards,
    apply_post_decode_guards,
    compute_guard_auxiliary,
)
from ._decoding import (
    should_reset,
    run_hmm_decode,
    build_decoder_state_source_suffix,
    save_decoder_state,
)
from ._serialization import (
    InferenceSnapshot,
    SuccessSnapshot,
    NoDataSnapshot,
    NoFeaturesSnapshot,
    QueuedSnapshot,
    QueuedProcessedSnapshot,
    ModelLoadErrorSnapshot,
    FeatureErrorSnapshot,
    PredictionErrorSnapshot,
    build_success_snapshot,
)

# Module-level logger
LOGGER = logging.getLogger(__name__)


def process_task(task: dict[str, Any], *, store: Any = None) -> InferenceSnapshot:
    """Process one queued inference task.

    Extracts features from a rolling timestamp window, resamples to model rate,
    runs prediction, and returns activity classification with confidence scores.

    Parameters
    ----------
    task : dict
        Queued task containing session_id, message_id, points_added, received_at_ns.
    store : SessionStore, optional
        Store instance to fetch raw points. Required for full inference.
        If None, returns placeholder snapshot.

    Returns
    -------
    InferenceSnapshot
        Pydantic model with activity, confidence, and metadata.
        Use .model_dump(mode="json") for serialization.
    """
    session_id = task.get("session_id")
    message_id = task.get("message_id")

    # Fast path: no store available
    if store is None:
        return QueuedProcessedSnapshot(session_id=session_id, message_id=message_id)

    # Load model artifacts (lazy)
    try:
        artifacts, bundle_meta = _ensure_artifacts_loaded()
    except Exception as e:
        return ModelLoadErrorSnapshot.from_error(
            session_id=session_id,
            message_id=message_id,
            error=e,
        )

    try:
        # Read raw points from store
        points = store.read_raw_points(session_id, limit=6000)
        if not points:
            return NoDataSnapshot(session_id=session_id, message_id=message_id)

        # Setup
        placement_labels = bundle_meta.get("placement_labels", [])
        target_sample_rate = float(bundle_meta.get("target_sample_rate_hz", 50.0))

        # Resample to fixed window
        acc_x, acc_y, acc_z, diagnostics = _resample_recent_window(
            points,
            target_sample_rate_hz=target_sample_rate,
            window_samples=MIN_WINDOW_SAMPLES,
        )
        if acc_x is None or acc_y is None or acc_z is None:
            return QueuedSnapshot.from_diagnostics(
                session_id=session_id,
                message_id=message_id,
                diagnostics=diagnostics,
                points=points,
            )

        placement_label = DEFAULT_PLACEMENT_LABEL

        # Extract features
        try:
            feature_row = extract_features(
                acc_x=acc_x,
                acc_y=acc_y,
                acc_z=acc_z,
                placement_labels=placement_labels,
                target_sample_rate_hz=target_sample_rate,
            )
        except Exception as e:
            return FeatureErrorSnapshot.from_error(
                session_id=session_id,
                message_id=message_id,
                error=e,
            )

        if feature_row is None or len(feature_row) == 0:
            return NoFeaturesSnapshot(session_id=session_id, message_id=message_id)

        # Model inference
        feature_matrix = np.array([feature_row], dtype=np.float64)
        emission_row = _compute_refined_proba_row(artifacts, feature_matrix)

        # Get previous decoder state
        previous_state = store.get_decoder_state(session_id) if session_id else None

        # --- Pre-decode guards ---
        (
            emission_row,
            low_motion_gate_fired,
            _,
            stairs_guard_applied,
            stairs_guard_info,
            stairs_moving_guard_applied,
            stairs_moving_guard_info,
            walk_guard_applied,
            walk_guard_info,
            hallucination_guard_applied,
            hallucination_guard_info,
        ) = apply_pre_decode_guards(
            feature_row=feature_row,
            emission=emission_row,
            label_order=artifacts.label_order,
            placement_labels=placement_labels,
            placement_label=placement_label,
            previous_state=previous_state,
        )

        # --- Decoder reset logic ---
        sequence_reset, reset_reason = should_reset(
            previous_state=previous_state,
            label_order=artifacts.label_order,
            low_motion_gate_fired=low_motion_gate_fired,
        )

        # Reset previous_state if needed
        if sequence_reset and reset_reason == "low-motion-gate":
            previous_state = None  # Will trigger re-init

        # --- HMM decoding ---
        decoded_proba, pred_idx, decoder_state_source = run_hmm_decode(
            emission=emission_row,
            artifacts=artifacts,
            previous_state=previous_state if not sequence_reset else None,
            sequence_reset=sequence_reset,
            reset_reason=reset_reason,
        )

        # --- Auxiliary computations needed for post-decode guards ---
        cadence_spm, periodicity_strength, artifact_score, artifact_info, metric_history = compute_guard_auxiliary(
            feature_row=feature_row,
            placement_labels=placement_labels,
            previous_state=previous_state,
        )

        # Raw prediction info (before post-decode guards)
        raw_idx = int(np.argmax(emission_row))
        raw_label = artifacts.label_order[raw_idx]
        raw_confidence = float(emission_row[raw_idx])

        # --- Post-decode guards ---
        (
            decoded_proba,
            stairs_failsafe_applied,
            stairs_failsafe_info,
            locomotion_transitions_guard_applied,
            locomotion_transitions_reason,
            artifact_confidence_calibrated,
            artifact_confidence_blend,
            pocket_sanity_corrected,
            pocket_sanity_reason,
        ) = apply_post_decode_guards(
            emission=emission_row,
            decoded_proba=decoded_proba,
            feature_row=feature_row,
            label_order=artifacts.label_order,
            placement_labels=placement_labels,
            placement_label=placement_label,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            cadence_spm=cadence_spm,
            periodicity_strength=periodicity_strength,
            artifact_score=artifact_score,
            artifact_info=artifact_info,
            previous_state=previous_state,
        )

        # Build decoder state source suffix with all applied corrections
        decoder_state_source = build_decoder_state_source_suffix(
            decoder_state_source=decoder_state_source,
            stairs_failsafe_applied=stairs_failsafe_applied,
            locomotion_transitions_guard_applied=locomotion_transitions_guard_applied,
            artifact_confidence_calibrated=artifact_confidence_calibrated,
            pocket_sanity_corrected=pocket_sanity_corrected,
        )

        # Final prediction
        pred_idx = int(np.argmax(decoded_proba))
        pred_label = artifacts.label_order[pred_idx]
        pred_confidence = float(decoded_proba[pred_idx])

        # Compute live insights from the feature row
        try:
            insights_obj = compute_live_window_insights(
                feature_row=feature_row,
                feature_names=artifacts.feature_names,
                baseline_body_mag_energy=0.123,  # RealWorld2016 lying baseline
                decoded_activity=pred_label,
                confidence=pred_confidence,
                artifact_score=artifact_score,
            )
            insights_dict = {
                "intensity_ratio": insights_obj.intensity_ratio,
                "intensity_level": insights_obj.intensity_level,
                "cadence_spm": insights_obj.cadence_spm,
                "cadence_consistency_pct": insights_obj.cadence_consistency_pct,
                "smoothness_score": insights_obj.smoothness_score,
                "burst_detected": insights_obj.burst_detected,
                "orientation_change_deg": insights_obj.orientation_change_deg,
                "posture_change_detected": insights_obj.posture_change_detected,
                "phone_orientation": insights_obj.phone_orientation,
                "signal_quality_score": insights_obj.signal_quality_score,
                "prediction_confidence_tier": insights_obj.prediction_confidence_tier,
                "activity_specific_insight": insights_obj.activity_specific_insight,
                "applicable_insights": insights_obj.applicable_insights,
                "stride_regularity_score": insights_obj.stride_regularity_score,
                "vertical_oscillation_mm": insights_obj.vertical_oscillation_mm,
                "ground_impact_score": insights_obj.ground_impact_score,
                "posture_stability_score": insights_obj.posture_stability_score,
                "sway_amplitude_deg": insights_obj.sway_amplitude_deg,
            }
        except Exception:
            LOGGER.exception(
                "Live insight computation failed for session_id=%s message_id=%s",
                session_id,
                message_id,
            )
            insights_dict = {}

        # Save decoder state
        save_decoder_state(
            store=store,
            session_id=session_id,
            decoded_proba=decoded_proba,
            diagnostics=diagnostics,
            metric_history=metric_history,
        )

        # Build and return success response
        return build_success_snapshot(
            session_id=session_id,
            message_id=message_id,
            diagnostics=diagnostics,
            points=points,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            pred_label=pred_label,
            pred_confidence=pred_confidence,
            decoded_proba=decoded_proba,
            label_order=artifacts.label_order,
            low_motion_gate_fired=low_motion_gate_fired,
            stairs_guard_applied=stairs_guard_applied,
            stairs_guard_info=stairs_guard_info,
            stairs_moving_guard_applied=stairs_moving_guard_applied,
            stairs_moving_guard_info=stairs_moving_guard_info,
            walk_guard_applied=walk_guard_applied,
            walk_guard_info=walk_guard_info,
            hallucination_guard_applied=hallucination_guard_applied,
            hallucination_guard_info=hallucination_guard_info,
            stairs_failsafe_applied=stairs_failsafe_applied,
            stairs_failsafe_info=stairs_failsafe_info,
            locomotion_transitions_guard_applied=locomotion_transitions_guard_applied,
            locomotion_transitions_guard_reason=locomotion_transitions_reason,
            artifact_confidence_calibrated=artifact_confidence_calibrated,
            artifact_confidence_blend=artifact_confidence_blend,
            pocket_sanity_corrected=pocket_sanity_corrected,
            pocket_sanity_reason=pocket_sanity_reason,
            cadence_spm=cadence_spm,
            periodicity_strength=periodicity_strength,
            artifact_score=artifact_score,
            artifact_info=artifact_info,
            hampel_replaced=0,  # Hampel not yet implemented in features
            decoder=artifacts.temporal_decoder,
            decoder_state_source=decoder_state_source,
            reset_reason=reset_reason,
            sequence_reset=sequence_reset,
            insights=insights_dict,
        )

    except Exception as e:
        return PredictionErrorSnapshot(
            session_id=session_id,
            message_id=message_id,
            error=f"Prediction failed: {str(e)}",
        )
