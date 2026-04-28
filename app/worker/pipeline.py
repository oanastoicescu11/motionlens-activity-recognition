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

# Import from extracted modules (relative imports within the worker package)
from ._steps import (
    MIN_WINDOW_SAMPLES,
    _ensure_artifacts_loaded,
    _resample_recent_window,
    _compute_refined_proba_row,
    _resolve_placement_label,
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
    resolve_reset_mode_and_gap,
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

        # Resolve placement
        placement_label, placement_source = _resolve_placement_label(task, bundle_meta)
        is_pocket_only = len(placement_labels) == 0

        # Extract features
        try:
            # For pocket_only mode, pass empty placement_labels so no one-hot
            # placement features are added (matching training with placement_labels=[]).
            feature_row = extract_features(
                acc_x=acc_x,
                acc_y=acc_y,
                acc_z=acc_z,
                placement_label=placement_label if not is_pocket_only else "front_pocket",
                placement_labels=() if is_pocket_only else placement_labels,
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
        reset_mode, boundary_gap_seconds = resolve_reset_mode_and_gap(bundle_meta)
        sequence_reset, reset_reason, sequence_gap_ms = should_reset(
            previous_state=previous_state,
            label_order=artifacts.label_order,
            low_motion_gate_fired=low_motion_gate_fired,
            reset_mode=reset_mode,
            boundary_gap_seconds=boundary_gap_seconds,
            current_window_start_ns=diagnostics.get("window_start_ns"),
        )

        # Reset previous_state if needed
        if sequence_reset and reset_reason in ("low-motion-gate", "per-window", "gap-boundary"):
            if reset_reason == "low-motion-gate":
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
            placement_label=placement_label,
            placement_source=placement_source,
            decoder=artifacts.temporal_decoder,
            decoder_state_source=decoder_state_source,
            reset_mode=reset_mode,
            boundary_gap_seconds=boundary_gap_seconds,
            sequence_reset=sequence_reset,
            sequence_gap_ms=sequence_gap_ms,
        )

    except Exception as e:
        return PredictionErrorSnapshot(
            session_id=session_id,
            message_id=message_id,
            error=f"Prediction failed: {str(e)}",
        )
