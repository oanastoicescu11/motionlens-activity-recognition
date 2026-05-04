"""Inference pipeline steps: resampling, feature extraction, and model inference.

This module contains the core data transformation steps that were extracted
from the monolithic process_task() function in pipeline.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

from core.features import compute_window_features, window_feature_names
from core.inference import (
    InferenceArtifacts,
    apply_static_specialist_refinement,
    predict_hierarchical_proba,
)

# Module-level logger and model singleton
LOGGER = logging.getLogger(__name__)
_artifacts: InferenceArtifacts | None = None
_bundle_meta: dict[str, Any] | None = None

# Model was trained on 128-sample windows (2.56 seconds at 50 Hz)
MIN_WINDOW_SAMPLES = 128
DEFAULT_PLACEMENT_LABEL = "front_pocket"


def _ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Coerce array to float64, avoiding redundant dtype conversions."""
    if arr.dtype == np.float64:
        return arr
    return np.asarray(arr, dtype=np.float64)


def _load_model() -> tuple[InferenceArtifacts, dict[str, Any]]:
    """Load pocket_only inference bundle from src/artifacts/model/.

    Loads inference_bundle.joblib which contains the XGBoost base model, optional
    static specialist, causal-HMM temporal decoder transition probs, and configuration.
    """
    artifacts_dir = Path(__file__).resolve().parents[2] / "artifacts" / "model"
    bundle_path = artifacts_dir / "inference_bundle.joblib"

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Runtime model bundle not found at {bundle_path}. "
            "Deploy src/artifacts/model/inference_bundle.joblib for live inference. "
            "Training and evaluation artifacts should stay under output/."
        )

    bundle: dict[str, Any] = joblib.load(bundle_path)
    LOGGER.info("Loaded pocket_only inference bundle from: %s", bundle_path)

    feature_names = list(bundle.get("feature_names") or window_feature_names(bundle.get("placement_labels", [])))

    artifacts = InferenceArtifacts(
        base_model=bundle["base_model"],
        label_order=bundle["label_order"],
        temporal_decoder=bundle.get("temporal_decoder", "none"),
        log_init_probs=bundle.get("log_init_probs"),
        log_trans_probs=bundle.get("log_trans_probs"),
        feature_names=feature_names,
        use_hierarchical=bool(bundle.get("use_hierarchical", False)),
        coarse_model=bundle.get("coarse_model"),
        fine_models=bundle.get("fine_models"),
        static_specialist_model=bundle.get("static_specialist_model"),
        static_specialist_labels=bundle.get("static_specialist_labels"),
        static_specialist_threshold=float(bundle.get("static_specialist_threshold", 0.35)),
        static_specialist_blend=float(bundle.get("static_specialist_blend", 0.6)),
    )

    meta = {
        "placement_labels": bundle.get("placement_labels", []),
        "target_sample_rate_hz": float(bundle.get("target_sample_rate_hz", 50.0)),
        "feature_names": feature_names,
        "model_type": "pocket_only",
    }

    return artifacts, meta


def _ensure_artifacts_loaded() -> tuple[InferenceArtifacts, dict[str, Any]]:
    """Load artifacts on first use (lazy initialization)."""
    global _artifacts, _bundle_meta
    if _artifacts is None:
        _artifacts, _bundle_meta = _load_model()
    assert _artifacts is not None and _bundle_meta is not None
    return _artifacts, _bundle_meta


def _resample_recent_window(
    points: list[tuple[int, float, float, float]],
    *,
    target_sample_rate_hz: float,
    window_samples: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Build a fixed-size recent window by timestamp and resample to the model rate."""
    if not points:
        return None, None, None, {
            "reason": "no-data",
            "sample_count": 0,
            "samples_needed": window_samples,
        }

    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 4:
        return None, None, None, {
            "reason": "invalid-points",
            "sample_count": len(points),
            "samples_needed": window_samples,
        }

    times_ns = data[:, 0].astype(np.int64)
    acc_x = data[:, 1].astype(np.float64)
    acc_y = data[:, 2].astype(np.float64)
    acc_z = data[:, 3].astype(np.float64)

    order = np.argsort(times_ns)
    times_ns = times_ns[order]
    acc_x = acc_x[order]
    acc_y = acc_y[order]
    acc_z = acc_z[order]

    unique_times_ns, unique_idx = np.unique(times_ns, return_index=True)
    times_ns = unique_times_ns
    acc_x = acc_x[unique_idx]
    acc_y = acc_y[unique_idx]
    acc_z = acc_z[unique_idx]

    if times_ns.size < 2:
        return None, None, None, {
            "reason": "insufficient-samples",
            "sample_count": int(times_ns.size),
            "samples_needed": window_samples,
        }

    window_duration_ns = int(round((window_samples / target_sample_rate_hz) * 1_000_000_000.0))
    end_ns = int(times_ns[-1])
    start_ns = end_ns - window_duration_ns

    in_window = times_ns >= start_ns
    win_t = times_ns[in_window]
    win_x = acc_x[in_window]
    win_y = acc_y[in_window]
    win_z = acc_z[in_window]

    if win_t.size < 2:
        return None, None, None, {
            "reason": "insufficient-samples",
            "sample_count": int(win_t.size),
            "samples_needed": window_samples,
        }

    span_ns = int(win_t[-1] - win_t[0])
    if span_ns <= 0 or span_ns < int(0.8 * window_duration_ns):
        return None, None, None, {
            "reason": "insufficient-time-span",
            "sample_count": int(win_t.size),
            "samples_needed": window_samples,
            "window_span_ms": span_ns / 1_000_000.0,
            "window_required_ms": window_duration_ns / 1_000_000.0,
        }

    target_t = np.linspace(start_ns, end_ns, num=window_samples, endpoint=False, dtype=np.float64)
    resampled_x = np.interp(target_t, win_t.astype(np.float64), win_x)
    resampled_y = np.interp(target_t, win_t.astype(np.float64), win_y)
    resampled_z = np.interp(target_t, win_t.astype(np.float64), win_z)

    diagnostics = {
        "reason": "ok",
        "sample_count": int(times_ns.size),
        "window_raw_samples": int(win_t.size),
        "window_samples_used": int(window_samples),
        "window_span_ms": span_ns / 1_000_000.0,
        "window_required_ms": window_duration_ns / 1_000_000.0,
        "window_start_ns": start_ns,
        "window_end_ns": end_ns,
    }
    return resampled_x, resampled_y, resampled_z, diagnostics


def _compute_refined_proba_row(artifacts: InferenceArtifacts, feature_matrix: np.ndarray) -> np.ndarray:
    """Compute one-row class probabilities with specialist refinement applied."""
    feature_matrix = _ensure_float64(feature_matrix)

    if artifacts.use_hierarchical:
        if artifacts.coarse_model is None:
            raise ValueError("coarse_model is required when use_hierarchical is enabled.")
        fine_models = artifacts.fine_models or {}
        y_proba_raw = predict_hierarchical_proba(
            x_values=feature_matrix,
            label_order=artifacts.label_order,
            coarse_model=artifacts.coarse_model,
            fine_models=fine_models,
        )
    else:
        predict_proba = getattr(artifacts.base_model, "predict_proba")
        y_proba_raw = predict_proba(feature_matrix)

    y_proba_raw = _ensure_float64(y_proba_raw)

    static_idx = [
        idx
        for idx, label in enumerate(artifacts.label_order)
        if label in set(artifacts.static_specialist_labels or [])
    ]
    gate_scores = np.sum(y_proba_raw[:, static_idx], axis=1) if static_idx else np.zeros(y_proba_raw.shape[0])

    y_proba_refined = apply_static_specialist_refinement(
        y_proba=y_proba_raw,
        x_values=feature_matrix,
        label_order=artifacts.label_order,
        static_model=artifacts.static_specialist_model,
        static_labels=artifacts.static_specialist_labels or [],
        gate_scores=gate_scores,
        gate_threshold=artifacts.static_specialist_threshold,
        blend=artifacts.static_specialist_blend,
    )
    return _ensure_float64(y_proba_refined[0])


def extract_features(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    *,
    placement_labels: list[str],
    target_sample_rate_hz: float,
) -> np.ndarray | None:
    """Extract features from resampled accelerometer data.

    Returns None if feature extraction fails or produces empty results.
    For pocket_only mode (empty placement_labels), no placement one-hot features
    are added and no stripping is needed since compute_window_features respects
    an empty placement_labels tuple.
    """
    is_pocket_only = len(placement_labels) == 0

    # For pocket_only, pass empty placement_labels so compute_window_features
    # doesn't add any placement one-hot features. This matches the training
    # config where placement_labels=[] means no placement features.
    effective_placement_labels: tuple[str, ...] = () if is_pocket_only else tuple(placement_labels)

    feature_row = compute_window_features(
        acc_x=acc_x,
        acc_y=acc_y,
        acc_z=acc_z,
        placement_label=DEFAULT_PLACEMENT_LABEL,
        placement_labels=effective_placement_labels,
        target_sample_rate_hz=target_sample_rate_hz,
    )

    if feature_row is None or len(feature_row) == 0:
        return None

    return _ensure_float64(feature_row)
