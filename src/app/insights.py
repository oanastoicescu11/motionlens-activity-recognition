"""Live demo insights computed from already-extracted feature windows.

This module intentionally consumes feature rows produced by the shared
FeaturePack2 extractor so live inference and insights share one processing path.

Threshold evidence:
- Intensity levels: RealWorld2016 (Sztyler & Stuckenschmidt, 15 probands, 50 Hz, waist):
  lying std_mag = 0.123 m/s² (still), standing = 0.191 m/s² (light),
  walking std_mag = 2.191 m/s² (moderate), running = 6.063 m/s² (high).
  [ANALYSIS_REPORT F4, F5; DOI placeholder]
- Cadence range: iPhone placement sweep trimmed walking 1.77–1.91 Hz,
  running 2.76–2.79 Hz across front-center, front-side, lateral zones.
  [ANALYSIS_REPORT F19]
- Postural sway ratio: WoW dataset (4–6 subjects per patch) shows standing std_mag
  ~3–5× sitting; used to calibrate posture_change_threshold_deg = 15.0.
  [ANALYSIS_REPORT F3]
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Activity groupings for conditional insight display
# ---------------------------------------------------------------------------

LOCOMOTION_ACTIVITIES = {"walk", "run", "stairs"}
STATIC_ACTIVITIES = {"sit/lay", "stand"}
ALL_ACTIVITIES = {"walk", "run", "stairs", "sit/lay", "stand", "transitions", "locomotion-other"}


def _activity_group(activity: str) -> str:
    """Map an activity label to its insight group."""
    if activity in LOCOMOTION_ACTIVITIES:
        return "locomotion"
    if activity in STATIC_ACTIVITIES:
        return "static"
    if activity == "transitions":
        return "transition"
    return "other"


# ---------------------------------------------------------------------------
# Per-window insight dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveWindowInsights:
    """One-window insight outputs used by the UI cards."""

    # Universal insights (always present)
    intensity_ratio: float
    intensity_level: str
    burst_detected: bool
    orientation_change_deg: float
    posture_change_detected: bool
    phone_orientation: str
    signal_quality_score: float
    prediction_confidence_tier: str
    activity_specific_insight: str
    applicable_insights: list[str]

    # Locomotion insights (meaningful for walk/run/stairs)
    cadence_spm: float
    cadence_consistency_pct: float
    smoothness_score: float
    stride_regularity_score: float
    vertical_oscillation_mm: float
    ground_impact_score: float

    # Static insights (meaningful for sit/lay/stand)
    posture_stability_score: float
    sway_amplitude_deg: float


# ---------------------------------------------------------------------------
# Session recap dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveSessionRecap:
    """Aggregated summary over a short live session."""

    active_seconds: float
    still_seconds: float
    moderate_or_higher_seconds: float
    high_seconds: float
    locomotion_seconds: float
    static_seconds: float
    peak_intensity_ratio: float
    cadence_mean_spm: float
    cadence_consistency_mean_pct: float
    smoothness_mean_score: float
    transition_count: int
    burst_count: int
    posture_change_count: int
    estimated_steps: int
    estimated_distance_m: float
    signal_quality_mean: float
    signal_quality_min: float
    peak_cadence_spm: float
    peak_impact_score: float
    avg_posture_stability: float
    phone_reposition_count: int
    most_common_activity: str
    time_by_activity: dict[str, float] = field(default_factory=dict)
    activity_transition_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature index helper
# ---------------------------------------------------------------------------


def _feature_index(feature_names: list[str]) -> dict[str, int]:
    index = {name: idx for idx, name in enumerate(feature_names)}
    required = {
        "body_mag_energy",
        "body_mag_dom_freq_hz",
        "body_mag_autocorr_peak_freq_hz",
        "body_mag_autocorr_periodicity_ratio",
        "body_mag_interpeak_interval_cv",
        "stft_dom_freq_std",
        "jerk_mag_std",
        "jerk_mag_p90",
        "dwt_high_low_ratio",
        "gravity_tilt_start_end_delta",
        # New required features for extended insights
        "body_vertical_energy",
        "gravity_tilt_abs_z_std",
        "body_mag_coeff_var",
        "gravity_x_mean",
        "gravity_y_mean",
        "gravity_z_mean",
        "gravity_angle_stability_std",
    }
    missing = sorted(required.difference(index))
    if missing:
        raise ValueError(f"Feature schema missing required names: {missing}")
    return index


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _intensity_level(intensity_ratio: float) -> str:
    """Classify intensity_ratio into activity level.

    Thresholds calibrated from RealWorld2016 (Sztyler & Stuckenschmidt, 15 probands, 50 Hz):
    - lying (baseline): std_mag = 0.123 m/s² → intensity_ratio ≈ 1.0
    - standing: std_mag = 0.191 m/s² → intensity_ratio ≈ 1.24
    - sitting: std_mag = 0.259 m/s² → intensity_ratio ≈ 1.45
    - walking: std_mag = 2.191 m/s² → intensity_ratio ≈ 4.22
    - running: std_mag = 6.063 m/s² → intensity_ratio ≈ 7.01

    Thresholds chosen to place sitting/standing in [light, moderate] band,
    walking clearly in moderate, running in high.

    Evidence: ANALYSIS_REPORT F4, F5; DESIGN.md MVP gait scope decision.
    """
    if intensity_ratio < 1.2:
        return "still"
    if intensity_ratio < 1.8:
        return "light"
    if intensity_ratio < 2.8:
        return "moderate"
    return "high"


# ---------------------------------------------------------------------------
# Cadence extraction (existing)
# ---------------------------------------------------------------------------


def _cadence_hz(feature_row: np.ndarray, idx: dict[str, int]) -> float:
    """Extract cadence frequency in Hz, with physiological range guard.

    Physiological range for human bipedal locomotion:
    - Walking: ~1.5–2.5 Hz (typical 1.77–1.91 Hz per iPhone placement sweep)
    - Running: ~2.5–3.5 Hz (typical 2.76–2.79 Hz per iPhone placement sweep)
    - Constraint: 0.5 Hz lower bound rules out postural sway (~0.2 Hz),
      4.0 Hz upper bound rules out unrealistic cadence artifacts.

    Evidence: ANALYSIS_REPORT F19 (iPhone sweep trimmed walking/running cadences),
    Mannini & Sabatini 2010 DOI:10.3390/s100201154 (gait biomechanics).
    """
    auto_hz = float(feature_row[idx["body_mag_autocorr_peak_freq_hz"]])
    fft_hz = float(feature_row[idx["body_mag_dom_freq_hz"]])
    if 0.5 <= auto_hz <= 4.0:
        return auto_hz
    if 0.5 <= fft_hz <= 4.0:
        return fft_hz
    return 0.0


# ---------------------------------------------------------------------------
# New insight helper functions
# ---------------------------------------------------------------------------


def _stride_regularity(periodicity: float, interval_cv: float, dom_freq_std: float) -> float:
    """Gait cycle consistency score (0–100).

    High regularity = tight inter-peak intervals, strong periodicity,
    stable dominant frequency.
    """
    score = 100.0 * (
        0.5 * _clamp((periodicity - 1.0) / 2.0, 0.0, 1.0)
        + 0.3 * (1.0 - _clamp(interval_cv / 0.5, 0.0, 1.0))
        + 0.2 * (1.0 - _clamp(dom_freq_std / 0.75, 0.0, 1.0))
    )
    return float(_clamp(score, 0.0, 100.0))


def _vertical_oscillation(body_vertical_energy: float, gravity_tilt_std: float) -> float:
    """Estimated vertical phone displacement in mm per step.

    Calibration: body_vertical_energy ~0.5 m/s² → ~10 mm bounce.
    gravity_tilt_std adds a small correction for pocket swing.
    """
    base_mm = math.sqrt(max(body_vertical_energy, 0.0)) * 14.0
    tilt_correction = gravity_tilt_std * 20.0  # radians → mm-ish
    return float(_clamp(base_mm + tilt_correction, 0.0, 80.0))


def _ground_impact(jerk_p90: float, jerk_std: float, body_energy: float) -> float:
    """Foot-strike harshness score (0–100).

    High jerk + high body energy = hard impact (running, heel-strike).
    Low jerk + low energy = soft impact (walking, forefoot).
    """
    normalized_jerk = _clamp(jerk_p90 / 4.0, 0.0, 1.0)
    normalized_std = _clamp(jerk_std / 2.5, 0.0, 1.0)
    energy_weight = _clamp(body_energy / 6.0, 0.0, 1.0)
    score = 100.0 * (0.5 * normalized_jerk + 0.3 * normalized_std + 0.2 * energy_weight)
    return float(_clamp(score, 0.0, 100.0))


def _posture_stability(body_energy: float, body_std: float, gravity_stability: float) -> float:
    """How still the body is during static posture (0–100).

    High score = very stable (meditation-quality stillness).
    Low score = fidgeting, tremor, or unstable posture.
    """
    energy_norm = _clamp(body_energy / 0.5, 0.0, 1.0)
    std_norm = _clamp(body_std / 0.3, 0.0, 1.0)
    gravity_norm = _clamp(gravity_stability / 0.2, 0.0, 1.0)
    score = 100.0 * (
        0.4 * (1.0 - energy_norm)
        + 0.35 * (1.0 - std_norm)
        + 0.25 * (1.0 - gravity_norm)
    )
    return float(_clamp(score, 0.0, 100.0))


def _sway_amplitude(gravity_tilt_std: float) -> float:
    """Estimated angular sway in degrees.

    gravity_tilt_abs_z_std is in radians; convert to degrees.
    """
    return float(abs(gravity_tilt_std) * (180.0 / math.pi))


def _phone_orientation(gx: float, gy: float, gz: float) -> str:
    """Human-readable phone orientation from gravity vector mean.

    DeviceMotion portrait frame: gravity aligns with ±Y when the phone is
    vertical (pocket carry) and with ±Z when the phone is flat (horizontal).
    """
    abs_x, abs_y, abs_z = abs(gx), abs(gy), abs(gz)
    dominant = max(abs_x, abs_y, abs_z)

    if dominant < 2.0:
        return "tilted"

    if abs_z >= 8.5 and abs_z >= abs_y and abs_z >= abs_x:
        return "flat"

    if abs_y >= 8.5 and abs_y >= abs_z and abs_y >= abs_x:
        return "upright" if gy < 0 else "upside-down"

    return "tilted"


def _signal_quality(artifact_score: float, dwt_ratio: float, coeff_var: float) -> float:
    """Composite signal quality score (0–100).

    High artifact score or high frequency noise = poor quality.
    Low variation and clean DWT ratio = good quality.
    """
    artifact_norm = _clamp(artifact_score / 1.0, 0.0, 1.0)
    dwt_norm = _clamp((dwt_ratio - 0.5) / 2.0, 0.0, 1.0)
    cv_norm = _clamp(coeff_var / 1.5, 0.0, 1.0)
    score = 100.0 * (
        0.5 * (1.0 - artifact_norm)
        + 0.3 * (1.0 - dwt_norm)
        + 0.2 * (1.0 - cv_norm)
    )
    return float(_clamp(score, 0.0, 100.0))


def _confidence_tier(confidence: float, artifact_score: float) -> str:
    """Qualitative confidence tier for user trust calibration.

    Thresholds align with the artifact-scoring module:
    - artifact_score < 1.5  → no confidence calibration is triggered
    - artifact_score >= 1.5 → calibration begins (blend = 0.35)
    - artifact_score >= 2.2 → stronger calibration (blend = 0.50)

    Therefore "high" tier is allowed when the signal is clean enough
    that the pipeline does not apply artifact-based blending.
    """
    if confidence >= 0.8 and artifact_score < 1.5:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def _activity_insight_text(
    activity: str,
    cadence_spm: float,
    impact_score: float,
    stability_score: float,
    regularity_score: float,
    sway_deg: float,
    signal_quality: float,
    confidence_tier: str,
) -> str:
    """Generate a human-readable one-liner for the current activity."""
    if confidence_tier == "low":
        return "Uncertain — try adjusting phone position"

    if activity == "walk":
        if regularity_score >= 80:
            return f"Steady walking at {cadence_spm:.0f} SPM"
        if impact_score >= 60:
            return f"Walking with firm steps at {cadence_spm:.0f} SPM"
        return f"Walking at {cadence_spm:.0f} SPM"

    if activity == "run":
        if impact_score >= 70:
            return f"High-impact running — consider softer landing"
        if regularity_score >= 80:
            return f"Smooth running at {cadence_spm:.0f} SPM"
        return f"Running at {cadence_spm:.0f} SPM"

    if activity == "stairs":
        return f"Climbing at {cadence_spm:.0f} SPM"

    if activity == "sit/lay":
        if stability_score >= 80:
            return "Stable seated posture"
        if sway_deg >= 5.0:
            return f"Slight sway while seated — {sway_deg:.1f}°"
        return "Relaxed seated posture"

    if activity == "stand":
        if stability_score >= 80:
            return "Standing still"
        if sway_deg >= 3.0:
            return f"Slight sway while standing — {sway_deg:.1f}°"
        return "Standing"

    if activity == "transitions":
        return "Transitioning..."

    return "Activity detected"


def _applicable_insights(activity: str) -> list[str]:
    """Return the list of insight keys that are meaningful for this activity."""
    group = _activity_group(activity)
    universal = [
        "intensity_ratio",
        "intensity_level",
        "burst_detected",
        "orientation_change_deg",
        "posture_change_detected",
        "phone_orientation",
        "signal_quality_score",
        "prediction_confidence_tier",
        "activity_specific_insight",
    ]

    if group == "locomotion":
        return universal + [
            "cadence_spm",
            "cadence_consistency_pct",
            "smoothness_score",
            "stride_regularity_score",
            "vertical_oscillation_mm",
            "ground_impact_score",
        ]

    if group == "static":
        return universal + [
            "posture_stability_score",
            "sway_amplitude_deg",
        ]

    if group == "transition":
        return universal + [
            "smoothness_score",
        ]

    # "other" — universal only
    return universal


# ---------------------------------------------------------------------------
# Main per-window insight computation
# ---------------------------------------------------------------------------


def compute_live_window_insights(
    feature_row: np.ndarray,
    feature_names: list[str],
    *,
    baseline_body_mag_energy: float,
    decoded_activity: str = "unknown",
    confidence: float = 0.0,
    artifact_score: float = 0.0,
    posture_change_threshold_deg: float = 15.0,
    burst_jerk_p90_threshold: float = 2.5,
    burst_ground_impact_threshold: float = 55.0,
    burst_intensity_ratio_threshold: float = 2.0,
) -> LiveWindowInsights:
    """Compute one-window insights from a precomputed feature row.

    Parameters
    ----------
    feature_row:
        One row from the shared feature extractor used for model inference.
    feature_names:
        Feature schema aligned with ``feature_row``.
    baseline_body_mag_energy:
        Resting baseline for intensity ratio normalization.
        Typically aligned to RealWorld2016 lying std_mag = 0.123 m/s².
        [ANALYSIS_REPORT F4, F5]
    decoded_activity:
        The activity label predicted by the model for this window.
        Used to determine which insights are applicable.
    confidence:
        Model confidence for the decoded activity.
    artifact_score:
        Loose-pocket artifact score from the inference pipeline.
    posture_change_threshold_deg:
        Minimum gravity tilt change (degrees) to flag posture transition.
        15° chosen based on WoW standing-vs-sitting postural sway amplitude
        (~3–5× ratio). [ANALYSIS_REPORT F3, F19]
    burst_jerk_p90_threshold:
        90th percentile jerk magnitude threshold for impact burst detection.
        Calibrated empirically against typical window feature distributions.
    burst_ground_impact_threshold:
        Minimum ground-impact score required before a burst alert is shown.
        This suppresses false positives from smaller posture adjustments.
    burst_intensity_ratio_threshold:
        Minimum normalized intensity ratio required before a burst alert is shown.
    """
    if baseline_body_mag_energy <= 0.0:
        raise ValueError("baseline_body_mag_energy must be positive.")

    idx = _feature_index(feature_names)
    values = np.asarray(feature_row, dtype=np.float64)
    if values.ndim != 1 or values.size != len(feature_names):
        raise ValueError("feature_row must be a 1D vector matching feature_names length.")

    # --- Universal computations ---
    body_mag_energy = float(values[idx["body_mag_energy"]])
    intensity_ratio = math.sqrt(max(body_mag_energy, 0.0) / baseline_body_mag_energy)
    intensity_level = _intensity_level(intensity_ratio)

    jerk_std = abs(float(values[idx["jerk_mag_std"]]))
    jerk_p90 = abs(float(values[idx["jerk_mag_p90"]]))
    dwt_ratio = abs(float(values[idx["dwt_high_low_ratio"]]))
    coeff_var = abs(float(values[idx["body_mag_coeff_var"]]))

    orientation_change_deg = abs(float(values[idx["gravity_tilt_start_end_delta"]])) * (180.0 / math.pi)
    posture_change_detected = orientation_change_deg >= posture_change_threshold_deg

    gx = float(values[idx["gravity_x_mean"]])
    gy = float(values[idx["gravity_y_mean"]])
    gz = float(values[idx["gravity_z_mean"]])
    phone_orientation = _phone_orientation(gx, gy, gz)

    signal_quality = _signal_quality(artifact_score, dwt_ratio, coeff_var)
    confidence_tier = _confidence_tier(confidence, artifact_score)

    # --- Locomotion computations ---
    cadence_hz = _cadence_hz(values, idx)
    cadence_spm = cadence_hz * 60.0

    periodicity = float(values[idx["body_mag_autocorr_periodicity_ratio"]])
    interval_cv = abs(float(values[idx["body_mag_interpeak_interval_cv"]]))
    dom_freq_std = abs(float(values[idx["stft_dom_freq_std"]]))
    cadence_consistency = _stride_regularity(periodicity, interval_cv, dom_freq_std)
    stride_regularity = cadence_consistency  # Same formula, renamed for clarity

    if cadence_hz <= 0.0:
        cadence_consistency = 0.0
        stride_regularity = 0.0

    smoothness = 100.0 * (
        0.45 * (1.0 - _clamp(jerk_std / 2.5, 0.0, 1.0))
        + 0.35 * (1.0 - _clamp(jerk_p90 / 4.0, 0.0, 1.0))
        + 0.20 * (1.0 - _clamp((dwt_ratio - 0.5) / 2.0, 0.0, 1.0))
    )
    smoothness = float(_clamp(smoothness, 0.0, 100.0))

    body_vertical_energy = float(values[idx["body_vertical_energy"]])
    gravity_tilt_std = float(values[idx["gravity_tilt_abs_z_std"]])
    vertical_oscillation = _vertical_oscillation(body_vertical_energy, gravity_tilt_std)
    ground_impact = _ground_impact(jerk_p90, jerk_std, body_mag_energy)
    burst_detected = (
        decoded_activity in LOCOMOTION_ACTIVITIES
        and jerk_p90 >= burst_jerk_p90_threshold
        and ground_impact >= burst_ground_impact_threshold
        and intensity_ratio >= burst_intensity_ratio_threshold
    )

    # --- Static computations ---
    # body_mag_std is not in the current model feature schema (153 features).
    # Derive a proxy from body_mag_energy since std ≈ sqrt(energy) for
    # zero-mean body acceleration magnitude.
    body_std = math.sqrt(max(body_mag_energy, 0.0))
    gravity_stability = abs(float(values[idx["gravity_angle_stability_std"]]))
    posture_stability = _posture_stability(body_mag_energy, body_std, gravity_stability)
    sway_amplitude = _sway_amplitude(gravity_tilt_std)

    # --- Activity-specific insight text ---
    activity_text = _activity_insight_text(
        activity=decoded_activity,
        cadence_spm=cadence_spm,
        impact_score=ground_impact,
        stability_score=posture_stability,
        regularity_score=stride_regularity,
        sway_deg=sway_amplitude,
        signal_quality=signal_quality,
        confidence_tier=confidence_tier,
    )

    applicable = _applicable_insights(decoded_activity)

    return LiveWindowInsights(
        intensity_ratio=float(intensity_ratio),
        intensity_level=intensity_level,
        burst_detected=bool(burst_detected),
        orientation_change_deg=float(orientation_change_deg),
        posture_change_detected=bool(posture_change_detected),
        phone_orientation=phone_orientation,
        signal_quality_score=float(signal_quality),
        prediction_confidence_tier=confidence_tier,
        activity_specific_insight=activity_text,
        applicable_insights=applicable,
        cadence_spm=float(cadence_spm),
        cadence_consistency_pct=float(_clamp(cadence_consistency, 0.0, 100.0)),
        smoothness_score=smoothness,
        stride_regularity_score=float(_clamp(stride_regularity, 0.0, 100.0)),
        vertical_oscillation_mm=float(vertical_oscillation),
        ground_impact_score=float(ground_impact),
        posture_stability_score=float(posture_stability),
        sway_amplitude_deg=float(sway_amplitude),
    )


# ---------------------------------------------------------------------------
# Rolling session aggregator
# ---------------------------------------------------------------------------


class LiveInsightsAggregator:
    """Rolling recap aggregator for 60–120 second live demo sessions.

    Tracks aggregated session metrics: total active/still/moderate-or-higher/high seconds,
    peak intensity, mean cadence and consistency, mean smoothness, and event counts
    (transitions, bursts, posture changes).

    Intensity tiers align with RealWorld2016 evidence (ANALYSIS_REPORT F4, F5):
    - still: < 1.2× baseline (lying/standing at rest)
    - light: 1.2–1.8× baseline (sitting, light postural activity)
    - moderate: 1.8–2.8× baseline (walking)
    - high: > 2.8× baseline (running, jumping, high-intensity locomotion)
    """

    def __init__(
        self,
        feature_names: list[str],
        *,
        baseline_body_mag_energy: float,
        window_hop_seconds: float = 1.28,
    ) -> None:
        if window_hop_seconds <= 0.0:
            raise ValueError("window_hop_seconds must be positive.")
        self._feature_names = list(feature_names)
        self._baseline_energy = float(baseline_body_mag_energy)
        self._window_hop_seconds = float(window_hop_seconds)
        self._window_count = 0

        # Time buckets
        self._active_seconds = 0.0
        self._still_seconds = 0.0
        self._moderate_or_higher_seconds = 0.0
        self._high_seconds = 0.0
        self._locomotion_seconds = 0.0
        self._static_seconds = 0.0

        # Per-activity time tracking
        self._time_by_activity: dict[str, float] = {}

        # Peak tracking
        self._peak_intensity_ratio = 0.0
        self._peak_cadence_spm = 0.0
        self._peak_impact_score = 0.0
        self._signal_quality_min = 100.0
        self._signal_quality_values: list[float] = []

        # Cadence tracking (locomotion only)
        self._cadence_values: list[float] = []
        self._cadence_consistency_values: list[float] = []

        # Smoothness tracking
        self._smoothness_values: list[float] = []

        # Posture stability tracking (static only)
        self._posture_stability_values: list[float] = []

        # Event counts
        self._transition_count = 0
        self._burst_count = 0
        self._posture_change_count = 0
        self._phone_reposition_count = 0

        # Activity transition log
        self._activity_transition_log: list[dict] = []
        self._last_activity: str | None = None
        self._last_level: str | None = None
        self._last_orientation: str | None = None

    def update(
        self,
        feature_row: np.ndarray,
        *,
        decoded_activity: str = "unknown",
        confidence: float = 0.0,
        artifact_score: float = 0.0,
    ) -> LiveWindowInsights:
        """Update rolling recap with one precomputed feature window."""
        insight = compute_live_window_insights(
            feature_row=feature_row,
            feature_names=self._feature_names,
            baseline_body_mag_energy=self._baseline_energy,
            decoded_activity=decoded_activity,
            confidence=confidence,
            artifact_score=artifact_score,
        )
        self._window_count += 1
        hop = self._window_hop_seconds

        # Intensity-based time buckets
        if insight.intensity_level == "still":
            self._still_seconds += hop
        else:
            self._active_seconds += hop

        if insight.intensity_level in {"moderate", "high"}:
            self._moderate_or_higher_seconds += hop
        if insight.intensity_level == "high":
            self._high_seconds += hop

        # Activity-group time buckets
        group = _activity_group(decoded_activity)
        if group == "locomotion":
            self._locomotion_seconds += hop
        elif group == "static":
            self._static_seconds += hop

        # Per-activity time
        self._time_by_activity[decoded_activity] = self._time_by_activity.get(decoded_activity, 0.0) + hop

        # Peak tracking
        self._peak_intensity_ratio = max(self._peak_intensity_ratio, insight.intensity_ratio)
        self._peak_cadence_spm = max(self._peak_cadence_spm, insight.cadence_spm)
        self._peak_impact_score = max(self._peak_impact_score, insight.ground_impact_score)
        self._signal_quality_min = min(self._signal_quality_min, insight.signal_quality_score)
        self._signal_quality_values.append(insight.signal_quality_score)

        # Locomotion-specific tracking
        if insight.cadence_spm > 0.0:
            self._cadence_values.append(insight.cadence_spm)
            self._cadence_consistency_values.append(insight.cadence_consistency_pct)

        self._smoothness_values.append(insight.smoothness_score)

        # Static-specific tracking
        if group == "static":
            self._posture_stability_values.append(insight.posture_stability_score)

        # Intensity-level transition count
        if self._last_level is not None and self._last_level != insight.intensity_level:
            self._transition_count += 1
        self._last_level = insight.intensity_level

        # Activity transition log
        if self._last_activity is not None and self._last_activity != decoded_activity:
            self._activity_transition_log.append({
                "from": self._last_activity,
                "to": decoded_activity,
                "window_index": self._window_count,
            })
        self._last_activity = decoded_activity

        # Event counts
        if insight.burst_detected:
            self._burst_count += 1
        if insight.posture_change_detected:
            self._posture_change_count += 1

        # Phone reposition detection (orientation change > 45° is handled by posture_change,
        # but we also track orientation label changes)
        if self._last_orientation is not None and self._last_orientation != insight.phone_orientation:
            self._phone_reposition_count += 1
        self._last_orientation = insight.phone_orientation

        return insight

    def recap(self) -> LiveSessionRecap:
        """Return aggregated recap values for the final demo card."""
        cadence_mean = float(np.mean(self._cadence_values)) if self._cadence_values else 0.0
        cadence_consistency_mean = (
            float(np.mean(self._cadence_consistency_values))
            if self._cadence_consistency_values
            else 0.0
        )
        smoothness_mean = float(np.mean(self._smoothness_values)) if self._smoothness_values else 0.0
        signal_quality_mean = (
            float(np.mean(self._signal_quality_values))
            if self._signal_quality_values
            else 0.0
        )
        posture_stability_mean = (
            float(np.mean(self._posture_stability_values))
            if self._posture_stability_values
            else 0.0
        )

        # Estimated steps: sum over walk windows (cadence_spm * seconds / 60)
        estimated_steps = 0
        walk_time = self._time_by_activity.get("walk", 0.0)
        if walk_time > 0 and self._cadence_values:
            # Use average cadence during locomotion periods
            avg_cadence = float(np.mean(self._cadence_values))
            estimated_steps = int(round(avg_cadence * self._locomotion_seconds / 60.0))

        # Estimated distance: ~0.7m per step (average stride)
        estimated_distance_m = estimated_steps * 0.7

        # Most common activity
        most_common = "unknown"
        if self._time_by_activity:
            most_common = max(self._time_by_activity, key=self._time_by_activity.get)  # type: ignore[arg-type]

        return LiveSessionRecap(
            active_seconds=float(self._active_seconds),
            still_seconds=float(self._still_seconds),
            moderate_or_higher_seconds=float(self._moderate_or_higher_seconds),
            high_seconds=float(self._high_seconds),
            locomotion_seconds=float(self._locomotion_seconds),
            static_seconds=float(self._static_seconds),
            peak_intensity_ratio=float(self._peak_intensity_ratio),
            cadence_mean_spm=cadence_mean,
            cadence_consistency_mean_pct=cadence_consistency_mean,
            smoothness_mean_score=smoothness_mean,
            transition_count=int(self._transition_count),
            burst_count=int(self._burst_count),
            posture_change_count=int(self._posture_change_count),
            estimated_steps=estimated_steps,
            estimated_distance_m=float(estimated_distance_m),
            signal_quality_mean=signal_quality_mean,
            signal_quality_min=float(self._signal_quality_min),
            peak_cadence_spm=float(self._peak_cadence_spm),
            peak_impact_score=float(self._peak_impact_score),
            avg_posture_stability=posture_stability_mean,
            phone_reposition_count=int(self._phone_reposition_count),
            most_common_activity=most_common,
            time_by_activity=dict(self._time_by_activity),
            activity_transition_log=list(self._activity_transition_log),
        )


_SUMMARY_DEFAULT_HOP_SECONDS = 0.7
_SUMMARY_MIN_STEP_SECONDS = 0.35
_SUMMARY_MAX_STEP_SECONDS = 2.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_history_entries(history: list[dict[str, object]]) -> list[dict[str, object]]:
    """Filter and normalize persisted live inference history records."""
    entries: list[dict[str, object]] = []
    for raw in history:
        if not isinstance(raw, dict):
            continue
        if raw.get("status") != "success":
            continue
        activity = str(raw.get("activity", "unknown"))
        if activity in {"no-data", "no-features", "unknown", "?"}:
            continue
        entries.append(
            {
                "message_id": _safe_int(raw.get("message_id")),
                "received_at_ns": _safe_int(raw.get("received_at_ns") or raw.get("updated_at_ns")),
                "window_start_ns": raw.get("window_start_ns"),
                "window_end_ns": raw.get("window_end_ns"),
                "activity": activity,
                "confidence": _safe_float(raw.get("confidence")),
                "cadence_spm": _safe_float(raw.get("cadence_spm")),
                "signal_quality_score": _safe_float(raw.get("signal_quality_score")),
                "intensity_level": str(raw.get("intensity_level", "unknown")),
                "intensity_ratio": _safe_float(raw.get("intensity_ratio")),
                "posture_change_detected": bool(raw.get("posture_change_detected", False)),
                "burst_detected": bool(raw.get("burst_detected", False)),
            }
        )

    entries.sort(key=lambda item: (_safe_int(item["received_at_ns"]), _safe_int(item["message_id"])))
    if not entries:
        return []

    deltas_seconds = [
        (_safe_int(curr["received_at_ns"]) - _safe_int(prev["received_at_ns"])) / 1_000_000_000.0
        for prev, curr in zip(entries, entries[1:])
        if _safe_int(curr["received_at_ns"]) > _safe_int(prev["received_at_ns"])
    ]
    default_hop_seconds = float(np.median(deltas_seconds)) if deltas_seconds else _SUMMARY_DEFAULT_HOP_SECONDS
    default_hop_seconds = _clamp(default_hop_seconds, _SUMMARY_MIN_STEP_SECONDS, _SUMMARY_MAX_STEP_SECONDS)

    normalized: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        if index + 1 < len(entries):
            next_ns = _safe_int(entries[index + 1]["received_at_ns"])
            current_ns = _safe_int(entry["received_at_ns"])
            step_seconds = (next_ns - current_ns) / 1_000_000_000.0 if next_ns > current_ns else default_hop_seconds
        else:
            step_seconds = default_hop_seconds
        normalized.append(
            {
                **entry,
                "duration_seconds": _clamp(step_seconds, _SUMMARY_MIN_STEP_SECONDS, _SUMMARY_MAX_STEP_SECONDS),
            }
        )
    return normalized


def _activity_noise_threshold_seconds(activity: str) -> float:
    if activity in {"run", "stairs"}:
        return 4.0
    if activity in {"walk", "locomotion-other"}:
        return 2.5
    if activity == "transitions":
        return 1.4
    return 1.2


def _new_bout(entry: dict[str, object]) -> dict[str, object]:
    duration_seconds = _safe_float(entry.get("duration_seconds"), _SUMMARY_DEFAULT_HOP_SECONDS)
    cadence_spm = _safe_float(entry.get("cadence_spm"))
    cadence_weight = duration_seconds if cadence_spm > 0.0 else 0.0
    return {
        "activity": str(entry["activity"]),
        "duration_seconds": duration_seconds,
        "confidence_seconds": _safe_float(entry.get("confidence")) * duration_seconds,
        "signal_quality_seconds": _safe_float(entry.get("signal_quality_score")) * duration_seconds,
        "cadence_seconds": cadence_spm * cadence_weight,
        "cadence_duration_seconds": cadence_weight,
        "burst_count": int(bool(entry.get("burst_detected", False))),
        "posture_change_count": int(bool(entry.get("posture_change_detected", False))),
        "source_count": 1,
    }


def _merge_bout_into(target: dict[str, object], source: dict[str, object]) -> None:
    target["duration_seconds"] = _safe_float(target.get("duration_seconds")) + _safe_float(source.get("duration_seconds"))
    target["confidence_seconds"] = _safe_float(target.get("confidence_seconds")) + _safe_float(source.get("confidence_seconds"))
    target["signal_quality_seconds"] = _safe_float(target.get("signal_quality_seconds")) + _safe_float(source.get("signal_quality_seconds"))
    target["cadence_seconds"] = _safe_float(target.get("cadence_seconds")) + _safe_float(source.get("cadence_seconds"))
    target["cadence_duration_seconds"] = _safe_float(target.get("cadence_duration_seconds")) + _safe_float(source.get("cadence_duration_seconds"))
    target["burst_count"] = _safe_int(target.get("burst_count")) + _safe_int(source.get("burst_count"))
    target["posture_change_count"] = _safe_int(target.get("posture_change_count")) + _safe_int(source.get("posture_change_count"))
    target["source_count"] = _safe_int(target.get("source_count")) + _safe_int(source.get("source_count"))


def _compress_bouts(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    bouts: list[dict[str, object]] = []
    for entry in entries:
        bout = _new_bout(entry)
        if bouts and bouts[-1]["activity"] == bout["activity"]:
            _merge_bout_into(bouts[-1], bout)
        else:
            bouts.append(bout)
    return bouts


def _fold_target_activity(
    previous_bout: dict[str, object] | None,
    current_bout: dict[str, object],
    next_bout: dict[str, object] | None,
) -> str | None:
    duration_seconds = _safe_float(current_bout.get("duration_seconds"))
    activity = str(current_bout["activity"])
    current_group = _activity_group(activity)
    previous_activity = str(previous_bout["activity"]) if previous_bout else None
    next_activity = str(next_bout["activity"]) if next_bout else None
    previous_group = _activity_group(previous_activity) if previous_activity else None
    next_group = _activity_group(next_activity) if next_activity else None

    if previous_activity and next_activity and previous_activity == next_activity:
        if duration_seconds <= _activity_noise_threshold_seconds(activity):
            return previous_activity

    if current_group == "locomotion" and duration_seconds <= _activity_noise_threshold_seconds(activity):
        if previous_activity == "walk" and next_activity == "walk":
            return "walk"
        if activity in {"run", "stairs"} and previous_activity == "walk" and next_bout is None:
            return "walk"
        if activity in {"run", "stairs"} and next_activity == "walk" and previous_bout is None:
            return "walk"

    if previous_bout and next_bout:
        confidence_mean = _safe_float(current_bout.get("confidence_seconds")) / max(duration_seconds, 1e-9)
        signal_quality_mean = _safe_float(current_bout.get("signal_quality_seconds")) / max(duration_seconds, 1e-9)
        if (
            current_group == "locomotion"
            and previous_group == next_group == "static"
            and duration_seconds <= 2.1
            and (confidence_mean < 0.7 or signal_quality_mean < 70.0)
        ):
            previous_duration = _safe_float(previous_bout.get("duration_seconds"))
            next_duration = _safe_float(next_bout.get("duration_seconds"))
            return previous_activity if previous_duration >= next_duration else next_activity

    return None


def _denoise_bouts(bouts: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    """Fold very short noisy bouts into surrounding dominant activity."""
    folded_bouts = 0
    working = [dict(bout) for bout in bouts]

    changed = True
    while changed and len(working) >= 2:
        changed = False
        candidate_indices = list(range(1, len(working) - 1))
        if len(working) == 2:
            candidate_indices.extend([1])
        candidate_indices.extend([0, len(working) - 1])

        seen_indices: set[int] = set()
        ordered_indices: list[int] = []
        for index in candidate_indices:
            if index in seen_indices:
                continue
            seen_indices.add(index)
            ordered_indices.append(index)

        for index in ordered_indices:
            bout = working[index]
            previous_bout = working[index - 1] if index > 0 else None
            next_bout = working[index + 1] if index + 1 < len(working) else None
            replacement = _fold_target_activity(previous_bout, bout, next_bout)
            if replacement is None:
                continue
            if replacement == str(bout["activity"]):
                continue
            working[index]["activity"] = replacement
            folded_bouts += 1
            recompressed: list[dict[str, object]] = []
            for bout in working:
                if recompressed and recompressed[-1]["activity"] == bout["activity"]:
                    _merge_bout_into(recompressed[-1], bout)
                else:
                    recompressed.append(dict(bout))
            working = recompressed
            changed = True
            break

    return working, folded_bouts


def _round_activity_seconds(time_by_activity: dict[str, float]) -> dict[str, float]:
    return {activity: round(seconds, 1) for activity, seconds in time_by_activity.items() if seconds > 0.0}


def build_live_session_summary(
    history: list[dict[str, object]],
    *,
    finalized_at_ns: int | None = None,
) -> dict[str, object]:
    """Generate a denoised end-of-stream summary from compact inference history."""
    finalized_ns = int(time.time_ns() if finalized_at_ns is None else finalized_at_ns)
    entries = _normalize_history_entries(history)
    if not entries:
        return {
            "status": "finalized",
            "finalized_at_ns": finalized_ns,
            "total_duration_seconds": 0.0,
            "dominant_activity": "unknown",
            "walking_seconds": 0.0,
            "stationary_seconds": 0.0,
            "locomotion_seconds": 0.0,
            "estimated_steps": 0,
            "estimated_distance_m": 0.0,
            "average_cadence_spm": 0.0,
            "peak_cadence_spm": 0.0,
            "signal_quality_mean": 0.0,
            "signal_quality_min": 0.0,
            "posture_change_count": 0,
            "noise_bouts_folded": 0,
            "time_by_activity": {},
            "activity_bouts": [],
            "notes": ["No valid inference windows were available to summarize."],
            "summary_text": "No usable live activity data was captured.",
        }

    raw_bouts = _compress_bouts(entries)
    denoised_bouts, folded_bouts = _denoise_bouts(raw_bouts)

    time_by_activity: dict[str, float] = {}
    locomotion_seconds = 0.0
    stationary_seconds = 0.0
    walking_seconds = 0.0
    cadence_weighted_seconds = 0.0
    cadence_duration_seconds = 0.0
    peak_cadence_spm = 0.0
    posture_change_count = 0
    longest_walk_bout_seconds = 0.0
    total_duration_seconds = 0.0

    activity_bouts: list[dict[str, object]] = []
    for bout in denoised_bouts:
        activity = str(bout["activity"])
        duration_seconds = _safe_float(bout.get("duration_seconds"))
        total_duration_seconds += duration_seconds
        time_by_activity[activity] = time_by_activity.get(activity, 0.0) + duration_seconds

        cadence_duration = _safe_float(bout.get("cadence_duration_seconds"))
        cadence_mean = (
            _safe_float(bout.get("cadence_seconds")) / cadence_duration
            if cadence_duration > 0.0
            else 0.0
        )
        confidence_mean = _safe_float(bout.get("confidence_seconds")) / max(duration_seconds, 1e-9)
        signal_quality_mean = _safe_float(bout.get("signal_quality_seconds")) / max(duration_seconds, 1e-9)

        if _activity_group(activity) == "locomotion":
            locomotion_seconds += duration_seconds
            cadence_weighted_seconds += _safe_float(bout.get("cadence_seconds"))
            cadence_duration_seconds += cadence_duration
            peak_cadence_spm = max(peak_cadence_spm, cadence_mean)
        if _activity_group(activity) == "static":
            stationary_seconds += duration_seconds
        if activity == "walk":
            walking_seconds += duration_seconds
            longest_walk_bout_seconds = max(longest_walk_bout_seconds, duration_seconds)

        posture_change_count += _safe_int(bout.get("posture_change_count"))
        activity_bouts.append(
            {
                "activity": activity,
                "duration_seconds": round(duration_seconds, 1),
                "confidence_mean": round(confidence_mean, 2),
                "signal_quality_mean": round(signal_quality_mean, 1),
                "cadence_spm": round(cadence_mean, 1),
            }
        )

    dominant_activity = max(time_by_activity, key=time_by_activity.get) if time_by_activity else "unknown"
    average_cadence_spm = cadence_weighted_seconds / cadence_duration_seconds if cadence_duration_seconds > 0.0 else 0.0
    signal_quality_values = [_safe_float(entry.get("signal_quality_score")) for entry in entries]
    signal_quality_mean = float(np.mean(signal_quality_values)) if signal_quality_values else 0.0
    signal_quality_min = min(signal_quality_values) if signal_quality_values else 0.0
    estimated_steps = int(round(sum(_safe_float(bout.get("cadence_seconds")) for bout in denoised_bouts) / 60.0))
    estimated_distance_m = round(estimated_steps * 0.7, 1)

    notes: list[str] = []
    if folded_bouts > 0:
        notes.append(
            f"Folded {folded_bouts} brief noisy activity bout{'s' if folded_bouts != 1 else ''} into neighboring activity."
        )
    if stationary_seconds > 0.0 and locomotion_seconds > 0.0:
        notes.append("Short locomotion spikes inside longer stationary periods were suppressed when they were not sustained.")
    if walking_seconds > 0.0:
        notes.append("Brief locomotion label blips, including short run or stairs segments inside or at the edges of a longer walk pattern, were merged when too short to be meaningful.")

    summary_text = (
        f"Mostly {dominant_activity.replace('-', ' ')} over {total_duration_seconds:.0f}s. "
        f"Walking {walking_seconds:.0f}s, stationary {stationary_seconds:.0f}s, "
        f"about {estimated_steps} steps at {average_cadence_spm:.0f} SPM average."
    )

    return {
        "status": "finalized",
        "finalized_at_ns": finalized_ns,
        "total_duration_seconds": round(total_duration_seconds, 1),
        "dominant_activity": dominant_activity,
        "walking_seconds": round(walking_seconds, 1),
        "stationary_seconds": round(stationary_seconds, 1),
        "locomotion_seconds": round(locomotion_seconds, 1),
        "estimated_steps": estimated_steps,
        "estimated_distance_m": estimated_distance_m,
        "average_cadence_spm": round(average_cadence_spm, 1),
        "peak_cadence_spm": round(peak_cadence_spm, 1),
        "signal_quality_mean": round(signal_quality_mean, 1),
        "signal_quality_min": round(signal_quality_min, 1),
        "posture_change_count": posture_change_count,
        "longest_walk_bout_seconds": round(longest_walk_bout_seconds, 1),
        "noise_bouts_folded": folded_bouts,
        "time_by_activity": _round_activity_seconds(time_by_activity),
        "activity_bouts": activity_bouts,
        "notes": notes,
        "summary_text": summary_text,
    }
