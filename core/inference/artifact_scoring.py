"""Pocket artifact scoring, session history tracking, and confidence calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.features.window_features import window_feature_names

FRONT_POCKET_LABEL = "front_pocket"

POCKET_ARTIFACT_SCORE_THRESHOLD = 1.5
POCKET_ARTIFACT_HIGH_THRESHOLD = 2.2
POCKET_CONFIDENCE_BLEND = 0.35
POCKET_CONFIDENCE_BLEND_HIGH = 0.50

ADAPTIVE_HISTORY_LEN = 90
ADAPTIVE_MIN_HISTORY = 20


def compute_loose_pocket_artifact_score(
    feature_row: np.ndarray,
    placement_labels: list[str],
) -> tuple[float, dict[str, float]]:
    """Estimate loose-pocket artifact severity from high-frequency and instability features."""
    feature_names = window_feature_names(placement_labels)
    info: dict[str, float] = {
        "artifact_jerk_term": 0.0,
        "artifact_hf_term": 0.0,
        "artifact_entropy_term": 0.0,
        "artifact_gravity_term": 0.0,
        "artifact_periodicity_term": 0.0,
        "artifact_dom_jitter_term": 0.0,
        "artifact_hf_ratio": 0.0,
    }

    try:
        jerk_idx = feature_names.index("jerk_mag_p90")
        e8_20_idx = feature_names.index("body_mag_energy_8_20_hz")
        e0p5_4_idx = feature_names.index("body_mag_energy_0p5_4_hz")
        entropy_idx = feature_names.index("body_mag_spectral_entropy")
        gravity_inst_idx = feature_names.index("gravity_angle_stability_std")
        periodicity_idx = feature_names.index("body_mag_autocorr_periodicity_ratio")
        dom_jitter_idx = feature_names.index("stft_dom_freq_std")
    except ValueError:
        return 0.0, info

    jerk_p90 = abs(float(feature_row[jerk_idx]))
    energy_8_20 = max(float(feature_row[e8_20_idx]), 0.0)
    energy_0p5_4 = max(float(feature_row[e0p5_4_idx]), 0.0)
    entropy = abs(float(feature_row[entropy_idx]))
    gravity_instability = abs(float(feature_row[gravity_inst_idx]))
    periodicity = abs(float(feature_row[periodicity_idx]))
    dom_jitter = abs(float(feature_row[dom_jitter_idx]))

    hf_ratio = energy_8_20 / max(energy_0p5_4, 1e-8)

    jerk_term = float(np.clip((jerk_p90 - 1.5) / 2.0, 0.0, 2.5))
    hf_term = float(np.clip((hf_ratio - 0.35) / 0.40, 0.0, 2.5))
    entropy_term = float(np.clip((entropy - 3.2) / 1.2, 0.0, 2.5))
    gravity_term = float(np.clip((gravity_instability - 0.10) / 0.20, 0.0, 2.5))
    periodicity_term = float(np.clip((periodicity - 1.3) / 1.0, 0.0, 2.5))
    dom_jitter_term = float(np.clip((dom_jitter - 0.18) / 0.22, 0.0, 2.5))

    info.update({
        "artifact_jerk_term": jerk_term,
        "artifact_hf_term": hf_term,
        "artifact_entropy_term": entropy_term,
        "artifact_gravity_term": gravity_term,
        "artifact_periodicity_term": periodicity_term,
        "artifact_dom_jitter_term": dom_jitter_term,
        "artifact_hf_ratio": hf_ratio,
    })

    score = (
        0.30 * jerk_term
        + 0.30 * hf_term
        + 0.15 * entropy_term
        + 0.15 * gravity_term
        + 0.10 * dom_jitter_term
        - 0.35 * periodicity_term
    )
    return float(max(0.0, score)), info


def extract_metric_history(previous_state: dict[str, Any] | None) -> dict[str, list[float]]:
    """Read per-session artifact metric history from decoder state."""
    if not isinstance(previous_state, dict):
        return {}
    raw = previous_state.get("metric_history")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[float]] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            cleaned = [float(v) for v in value if isinstance(v, (int, float)) and np.isfinite(v)]
            if cleaned:
                out[str(key)] = cleaned[-ADAPTIVE_HISTORY_LEN:]
    return out


def append_metric_history(
    history: dict[str, list[float]],
    *,
    values: dict[str, float],
) -> dict[str, list[float]]:
    """Append current metrics into bounded history."""
    out = {key: list(seq[-ADAPTIVE_HISTORY_LEN:]) for key, seq in history.items()}
    for key, value in values.items():
        seq = out.setdefault(key, [])
        seq.append(float(value))
        if len(seq) > ADAPTIVE_HISTORY_LEN:
            del seq[: len(seq) - ADAPTIVE_HISTORY_LEN]
    return out


def robust_center_scale(values: list[float]) -> tuple[float, float]:
    """Return robust center/scale using median and MAD."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 1.0
    center = float(np.median(arr))
    mad = float(np.median(np.abs(arr - center))) * 1.4826
    scale = max(mad, 1e-3)
    return center, scale


def robust_z(value: float, history_values: list[float]) -> float:
    """Compute robust z-score relative to history; 0.0 if insufficient history."""
    if len(history_values) < ADAPTIVE_MIN_HISTORY:
        return 0.0
    center, scale = robust_center_scale(history_values)
    return float(np.clip((float(value) - center) / scale, -4.0, 4.0))


def compute_generic_artifact_score(
    feature_row: np.ndarray,
    placement_labels: list[str],
    *,
    previous_state: dict[str, Any] | None,
) -> tuple[float, dict[str, float], dict[str, list[float]]]:
    """Compute clothing-agnostic pocket artifact score with per-session adaptation."""
    base_score, base_info = compute_loose_pocket_artifact_score(feature_row, placement_labels)
    feature_names = window_feature_names(placement_labels)

    try:
        jerk_idx = feature_names.index("jerk_mag_p90")
        e8_20_idx = feature_names.index("body_mag_energy_8_20_hz")
        e0p5_4_idx = feature_names.index("body_mag_energy_0p5_4_hz")
        entropy_idx = feature_names.index("body_mag_spectral_entropy")
        gravity_inst_idx = feature_names.index("gravity_angle_stability_std")
        periodicity_idx = feature_names.index("body_mag_autocorr_periodicity_ratio")
        dom_jitter_idx = feature_names.index("stft_dom_freq_std")
    except ValueError:
        return base_score, base_info, extract_metric_history(previous_state)

    raw_metrics = {
        "jerk_p90": abs(float(feature_row[jerk_idx])),
        "hf_ratio": max(float(feature_row[e8_20_idx]), 0.0) / max(float(feature_row[e0p5_4_idx]), 1e-8),
        "entropy": abs(float(feature_row[entropy_idx])),
        "gravity_instability": abs(float(feature_row[gravity_inst_idx])),
        "periodicity_ratio": abs(float(feature_row[periodicity_idx])),
        "dom_jitter": abs(float(feature_row[dom_jitter_idx])),
    }

    history = extract_metric_history(previous_state)
    z_jerk = robust_z(raw_metrics["jerk_p90"], history.get("jerk_p90", []))
    z_hf = robust_z(raw_metrics["hf_ratio"], history.get("hf_ratio", []))
    z_entropy = robust_z(raw_metrics["entropy"], history.get("entropy", []))
    z_gravity = robust_z(raw_metrics["gravity_instability"], history.get("gravity_instability", []))
    z_periodicity = robust_z(raw_metrics["periodicity_ratio"], history.get("periodicity_ratio", []))
    z_dom_jitter = robust_z(raw_metrics["dom_jitter"], history.get("dom_jitter", []))

    adaptive_score = (
        0.30 * max(0.0, z_jerk)
        + 0.30 * max(0.0, z_hf)
        + 0.15 * max(0.0, z_entropy)
        + 0.15 * max(0.0, z_gravity)
        + 0.10 * max(0.0, z_dom_jitter)
        - 0.35 * max(0.0, z_periodicity)
    )

    has_adaptive = len(history.get("jerk_p90", [])) >= ADAPTIVE_MIN_HISTORY
    if has_adaptive:
        score = 0.35 * base_score + 0.65 * float(max(0.0, adaptive_score))
    else:
        score = base_score

    info = dict(base_info)
    info.update({
        "artifact_adaptive_enabled": 1.0 if has_adaptive else 0.0,
        "artifact_z_jerk": z_jerk,
        "artifact_z_hf": z_hf,
        "artifact_z_entropy": z_entropy,
        "artifact_z_gravity": z_gravity,
        "artifact_z_periodicity": z_periodicity,
        "artifact_z_dom_jitter": z_dom_jitter,
        "artifact_base_score": base_score,
        "artifact_adaptive_score": float(max(0.0, adaptive_score)),
    })

    updated_history = append_metric_history(history, values=raw_metrics)
    return float(max(0.0, score)), info, updated_history


def apply_artifact_confidence_calibration(
    decoded_proba: np.ndarray,
    *,
    artifact_score: float,
    placement_label: str,
) -> tuple[np.ndarray, bool, float]:
    """Reduce overconfidence under high pocket artifact by blending with uniform prior."""
    if placement_label != FRONT_POCKET_LABEL:
        return decoded_proba, False, 0.0
    if artifact_score <= POCKET_ARTIFACT_SCORE_THRESHOLD:
        return decoded_proba, False, 0.0

    blend = POCKET_CONFIDENCE_BLEND
    if artifact_score >= POCKET_ARTIFACT_HIGH_THRESHOLD:
        blend = POCKET_CONFIDENCE_BLEND_HIGH

    uniform = np.full(decoded_proba.size, 1.0 / max(decoded_proba.size, 1), dtype=np.float64)
    out = (1.0 - blend) * decoded_proba + blend * uniform
    total = float(np.sum(out))
    if total > 0.0:
        out /= total
    return out, True, float(blend)
