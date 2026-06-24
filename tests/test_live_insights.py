import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.insights import (
    LiveInsightsAggregator,
    build_live_session_summary,
    compute_live_window_insights,
    _stride_regularity,
    _vertical_oscillation,
    _ground_impact,
    _posture_stability,
    _sway_amplitude,
    _phone_orientation,
    _signal_quality,
    _confidence_tier,
    _activity_insight_text,
    _applicable_insights,
    _activity_group,
)


def _schema() -> list[str]:
    return [
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
        "body_mag_std",
    ]


def _row(**values: float) -> np.ndarray:
    names = _schema()
    out = np.zeros(len(names), dtype=np.float64)
    idx = {name: i for i, name in enumerate(names)}
    for key, value in values.items():
        out[idx[key]] = float(value)
    return out


class LiveInsightsTests(unittest.TestCase):
    def test_window_insights_uses_feature_vector_inputs(self) -> None:
        features = _row(
            body_mag_energy=6.25,
            body_mag_dom_freq_hz=1.8,
            body_mag_autocorr_peak_freq_hz=1.75,
            body_mag_autocorr_periodicity_ratio=2.8,
            body_mag_interpeak_interval_cv=0.12,
            stft_dom_freq_std=0.08,
            jerk_mag_std=0.6,
            jerk_mag_p90=1.2,
            dwt_high_low_ratio=0.9,
            gravity_tilt_start_end_delta=np.deg2rad(4.0),
            body_vertical_energy=0.5,
            gravity_tilt_abs_z_std=0.05,
            body_mag_coeff_var=0.3,
            gravity_x_mean=0.5,
            gravity_y_mean=-9.5,
            gravity_z_mean=0.3,
            gravity_angle_stability_std=0.02,
            body_mag_std=1.2,
        )

        insight = compute_live_window_insights(
            feature_row=features,
            feature_names=_schema(),
            baseline_body_mag_energy=1.0,
            decoded_activity="walk",
            confidence=0.92,
            artifact_score=0.1,
        )

        self.assertEqual(insight.intensity_level, "moderate")
        self.assertGreater(insight.cadence_spm, 100.0)
        self.assertGreater(insight.cadence_consistency_pct, 75.0)
        self.assertFalse(insight.burst_detected)
        self.assertFalse(insight.posture_change_detected)
        self.assertEqual(insight.prediction_confidence_tier, "high")
        self.assertEqual(insight.phone_orientation, "upright")
        self.assertIn("cadence_spm", insight.applicable_insights)
        self.assertIn("stride_regularity_score", insight.applicable_insights)
        self.assertNotIn("posture_stability_score", insight.applicable_insights)

    def test_orientation_change_detected_when_tilt_delta_large(self) -> None:
        features = _row(
            body_mag_energy=1.0,
            body_mag_dom_freq_hz=0.0,
            body_mag_autocorr_peak_freq_hz=0.0,
            body_mag_autocorr_periodicity_ratio=0.0,
            body_mag_interpeak_interval_cv=0.0,
            stft_dom_freq_std=0.0,
            jerk_mag_std=0.3,
            jerk_mag_p90=0.8,
            dwt_high_low_ratio=0.6,
            gravity_tilt_start_end_delta=np.deg2rad(24.0),
            body_vertical_energy=0.1,
            gravity_tilt_abs_z_std=0.02,
            body_mag_coeff_var=0.1,
            gravity_x_mean=0.2,
            gravity_y_mean=0.1,
            gravity_z_mean=-9.7,
            gravity_angle_stability_std=0.01,
            body_mag_std=0.2,
        )
        insight = compute_live_window_insights(
            feature_row=features,
            feature_names=_schema(),
            baseline_body_mag_energy=1.0,
            decoded_activity="stand",
            confidence=0.85,
            artifact_score=0.1,
        )
        self.assertTrue(insight.posture_change_detected)
        self.assertGreaterEqual(insight.orientation_change_deg, 24.0 - 1e-6)
        self.assertIn("posture_stability_score", insight.applicable_insights)
        self.assertIn("sway_amplitude_deg", insight.applicable_insights)
        self.assertNotIn("cadence_spm", insight.applicable_insights)

    def test_burst_detection_ignores_small_non_locomotion_adjustments(self) -> None:
        features = _row(
            body_mag_energy=0.9,
            body_mag_dom_freq_hz=0.0,
            body_mag_autocorr_peak_freq_hz=0.0,
            body_mag_autocorr_periodicity_ratio=0.0,
            body_mag_interpeak_interval_cv=0.0,
            stft_dom_freq_std=0.1,
            jerk_mag_std=1.0,
            jerk_mag_p90=3.2,
            dwt_high_low_ratio=0.7,
            gravity_tilt_start_end_delta=np.deg2rad(12.0),
            body_vertical_energy=0.1,
            gravity_tilt_abs_z_std=0.03,
            body_mag_coeff_var=0.2,
            gravity_x_mean=0.2,
            gravity_y_mean=0.1,
            gravity_z_mean=-9.7,
            gravity_angle_stability_std=0.02,
            body_mag_std=0.3,
        )

        insight = compute_live_window_insights(
            feature_row=features,
            feature_names=_schema(),
            baseline_body_mag_energy=1.0,
            decoded_activity="transitions",
            confidence=0.8,
            artifact_score=0.1,
        )

        self.assertFalse(insight.burst_detected)

    def test_burst_detection_requires_strong_locomotion_impact(self) -> None:
        features = _row(
            body_mag_energy=9.0,
            body_mag_dom_freq_hz=2.2,
            body_mag_autocorr_peak_freq_hz=2.1,
            body_mag_autocorr_periodicity_ratio=2.0,
            body_mag_interpeak_interval_cv=0.3,
            stft_dom_freq_std=0.2,
            jerk_mag_std=1.6,
            jerk_mag_p90=3.5,
            dwt_high_low_ratio=1.5,
            gravity_tilt_start_end_delta=np.deg2rad(18.0),
            body_vertical_energy=0.8,
            gravity_tilt_abs_z_std=0.06,
            body_mag_coeff_var=0.4,
            gravity_x_mean=0.6,
            gravity_y_mean=0.3,
            gravity_z_mean=-9.2,
            gravity_angle_stability_std=0.02,
            body_mag_std=1.8,
        )

        insight = compute_live_window_insights(
            feature_row=features,
            feature_names=_schema(),
            baseline_body_mag_energy=1.0,
            decoded_activity="run",
            confidence=0.92,
            artifact_score=0.2,
        )

        self.assertTrue(insight.burst_detected)

    def test_aggregator_recap_tracks_durations_and_counts(self) -> None:
        agg = LiveInsightsAggregator(
            feature_names=_schema(),
            baseline_body_mag_energy=1.0,
            window_hop_seconds=1.28,
        )

        still = _row(
            body_mag_energy=1.0,
            body_mag_dom_freq_hz=0.0,
            body_mag_autocorr_peak_freq_hz=0.0,
            body_mag_autocorr_periodicity_ratio=0.0,
            body_mag_interpeak_interval_cv=0.0,
            stft_dom_freq_std=0.0,
            jerk_mag_std=0.2,
            jerk_mag_p90=1.0,
            dwt_high_low_ratio=0.7,
            gravity_tilt_start_end_delta=np.deg2rad(2.0),
            body_vertical_energy=0.05,
            gravity_tilt_abs_z_std=0.01,
            body_mag_coeff_var=0.05,
            gravity_x_mean=0.1,
            gravity_y_mean=0.1,
            gravity_z_mean=-9.8,
            gravity_angle_stability_std=0.005,
            body_mag_std=0.15,
        )
        moderate = _row(
            body_mag_energy=4.0,
            body_mag_dom_freq_hz=1.7,
            body_mag_autocorr_peak_freq_hz=1.6,
            body_mag_autocorr_periodicity_ratio=2.2,
            body_mag_interpeak_interval_cv=0.2,
            stft_dom_freq_std=0.15,
            jerk_mag_std=0.7,
            jerk_mag_p90=1.8,
            dwt_high_low_ratio=0.9,
            gravity_tilt_start_end_delta=np.deg2rad(6.0),
            body_vertical_energy=0.4,
            gravity_tilt_abs_z_std=0.04,
            body_mag_coeff_var=0.25,
            gravity_x_mean=0.4,
            gravity_y_mean=0.2,
            gravity_z_mean=-9.4,
            gravity_angle_stability_std=0.015,
            body_mag_std=1.0,
        )
        high_burst = _row(
            body_mag_energy=9.0,
            body_mag_dom_freq_hz=2.2,
            body_mag_autocorr_peak_freq_hz=2.1,
            body_mag_autocorr_periodicity_ratio=2.0,
            body_mag_interpeak_interval_cv=0.3,
            stft_dom_freq_std=0.2,
            jerk_mag_std=1.6,
            jerk_mag_p90=3.5,
            dwt_high_low_ratio=1.5,
            gravity_tilt_start_end_delta=np.deg2rad(18.0),
            body_vertical_energy=0.8,
            gravity_tilt_abs_z_std=0.06,
            body_mag_coeff_var=0.4,
            gravity_x_mean=0.6,
            gravity_y_mean=0.3,
            gravity_z_mean=-9.2,
            gravity_angle_stability_std=0.02,
            body_mag_std=1.8,
        )

        agg.update(still, decoded_activity="sit/lay", confidence=0.9, artifact_score=0.1)
        agg.update(moderate, decoded_activity="walk", confidence=0.88, artifact_score=0.15)
        agg.update(high_burst, decoded_activity="run", confidence=0.92, artifact_score=0.2)
        recap = agg.recap()

        self.assertAlmostEqual(recap.still_seconds, 1.28, places=6)
        self.assertAlmostEqual(recap.active_seconds, 2.56, places=6)
        self.assertEqual(recap.transition_count, 2)
        self.assertEqual(recap.burst_count, 1)
        self.assertEqual(recap.posture_change_count, 1)
        self.assertGreater(recap.cadence_mean_spm, 90.0)
        self.assertGreater(recap.locomotion_seconds, 0.0)
        self.assertGreater(recap.static_seconds, 0.0)
        self.assertIn("walk", recap.time_by_activity)
        self.assertIn("run", recap.time_by_activity)
        self.assertIn("sit/lay", recap.time_by_activity)
        self.assertGreater(recap.estimated_steps, 0)
        self.assertGreater(recap.estimated_distance_m, 0.0)
        self.assertGreater(recap.peak_cadence_spm, 0.0)
        self.assertGreater(recap.peak_impact_score, 0.0)
        self.assertGreater(recap.signal_quality_mean, 0.0)
        self.assertEqual(recap.phone_reposition_count, 0)

    def test_raises_for_invalid_baseline(self) -> None:
        with self.assertRaises(ValueError):
            compute_live_window_insights(
                feature_row=np.zeros(len(_schema()), dtype=np.float64),
                feature_names=_schema(),
                baseline_body_mag_energy=0.0,
            )


class InsightHelperTests(unittest.TestCase):
    def test_stride_regularity_high(self) -> None:
        score = _stride_regularity(periodicity=2.8, interval_cv=0.1, dom_freq_std=0.05)
        self.assertGreater(score, 80.0)

    def test_stride_regularity_low(self) -> None:
        score = _stride_regularity(periodicity=1.0, interval_cv=0.6, dom_freq_std=0.8)
        self.assertLess(score, 50.0)

    def test_vertical_oscillation_positive(self) -> None:
        mm = _vertical_oscillation(body_vertical_energy=0.5, gravity_tilt_std=0.05)
        self.assertGreater(mm, 0.0)
        self.assertLess(mm, 80.0)

    def test_ground_impact_running(self) -> None:
        score = _ground_impact(jerk_p90=3.5, jerk_std=1.5, body_energy=6.0)
        self.assertGreater(score, 50.0)

    def test_ground_impact_walking(self) -> None:
        score = _ground_impact(jerk_p90=1.2, jerk_std=0.5, body_energy=2.0)
        self.assertLess(score, 50.0)

    def test_posture_stability_still(self) -> None:
        score = _posture_stability(body_energy=0.05, body_std=0.05, gravity_stability=0.01)
        self.assertGreater(score, 80.0)

    def test_posture_stability_unstable(self) -> None:
        score = _posture_stability(body_energy=0.8, body_std=0.5, gravity_stability=0.3)
        self.assertLess(score, 50.0)

    def test_sway_amplitude_conversion(self) -> None:
        deg = _sway_amplitude(0.1)  # ~0.1 radians
        self.assertAlmostEqual(deg, 5.73, places=1)

    def test_phone_orientation_upright(self) -> None:
        self.assertEqual(_phone_orientation(0.0, -9.8, 0.0), "upright")

    def test_phone_orientation_flat(self) -> None:
        self.assertEqual(_phone_orientation(0.0, 0.0, -9.8), "flat")

    def test_phone_orientation_flat_face_down(self) -> None:
        self.assertEqual(_phone_orientation(0.0, 0.0, 9.8), "flat")

    def test_phone_orientation_upside_down(self) -> None:
        self.assertEqual(_phone_orientation(0.0, 9.8, 0.0), "upside-down")

    def test_signal_quality_good(self) -> None:
        score = _signal_quality(artifact_score=0.1, dwt_ratio=0.6, coeff_var=0.2)
        self.assertGreater(score, 70.0)

    def test_signal_quality_poor(self) -> None:
        score = _signal_quality(artifact_score=0.9, dwt_ratio=3.0, coeff_var=1.5)
        self.assertLess(score, 50.0)

    def test_confidence_tier_high(self) -> None:
        self.assertEqual(_confidence_tier(0.85, 0.1), "high")

    def test_confidence_tier_medium(self) -> None:
        self.assertEqual(_confidence_tier(0.6, 0.5), "medium")

    def test_confidence_tier_low(self) -> None:
        self.assertEqual(_confidence_tier(0.3, 0.5), "low")

    def test_activity_insight_text_walk(self) -> None:
        text = _activity_insight_text(
            activity="walk",
            cadence_spm=110,
            impact_score=30,
            stability_score=80,
            regularity_score=85,
            sway_deg=2.0,
            signal_quality=90,
            confidence_tier="high",
        )
        self.assertIn("110", text)
        self.assertIn("walk", text.lower())

    def test_activity_insight_text_run_high_impact(self) -> None:
        text = _activity_insight_text(
            activity="run",
            cadence_spm=165,
            impact_score=75,
            stability_score=70,
            regularity_score=80,
            sway_deg=3.0,
            signal_quality=85,
            confidence_tier="high",
        )
        self.assertIn("softer landing", text)

    def test_activity_insight_text_sit(self) -> None:
        text = _activity_insight_text(
            activity="sit/lay",
            cadence_spm=0,
            impact_score=10,
            stability_score=85,
            regularity_score=0,
            sway_deg=1.5,
            signal_quality=95,
            confidence_tier="high",
        )
        self.assertIn("Stable", text)

    def test_activity_insight_text_low_confidence(self) -> None:
        text = _activity_insight_text(
            activity="walk",
            cadence_spm=100,
            impact_score=30,
            stability_score=70,
            regularity_score=70,
            sway_deg=2.0,
            signal_quality=50,
            confidence_tier="low",
        )
        self.assertIn("Uncertain", text)


class ApplicableInsightsTests(unittest.TestCase):
    def test_locomotion_insights(self) -> None:
        for activity in ("walk", "run", "stairs"):
            applicable = _applicable_insights(activity)
            self.assertIn("cadence_spm", applicable)
            self.assertIn("stride_regularity_score", applicable)
            self.assertIn("ground_impact_score", applicable)
            self.assertNotIn("posture_stability_score", applicable)
            self.assertNotIn("sway_amplitude_deg", applicable)

    def test_static_insights(self) -> None:
        for activity in ("sit/lay", "stand"):
            applicable = _applicable_insights(activity)
            self.assertIn("posture_stability_score", applicable)
            self.assertIn("sway_amplitude_deg", applicable)
            self.assertNotIn("cadence_spm", applicable)
            self.assertNotIn("stride_regularity_score", applicable)

    def test_transition_insights(self) -> None:
        applicable = _applicable_insights("transitions")
        self.assertIn("smoothness_score", applicable)
        self.assertNotIn("cadence_spm", applicable)
        self.assertNotIn("posture_stability_score", applicable)

    def test_other_insights(self) -> None:
        applicable = _applicable_insights("locomotion-other")
        self.assertNotIn("cadence_spm", applicable)
        self.assertNotIn("posture_stability_score", applicable)
        self.assertIn("intensity_ratio", applicable)


class ActivityGroupTests(unittest.TestCase):
    def test_locomotion_group(self) -> None:
        self.assertEqual(_activity_group("walk"), "locomotion")
        self.assertEqual(_activity_group("run"), "locomotion")
        self.assertEqual(_activity_group("stairs"), "locomotion")

    def test_static_group(self) -> None:
        self.assertEqual(_activity_group("sit/lay"), "static")
        self.assertEqual(_activity_group("stand"), "static")

    def test_transition_group(self) -> None:
        self.assertEqual(_activity_group("transitions"), "transition")

    def test_other_group(self) -> None:
        self.assertEqual(_activity_group("locomotion-other"), "other")
        self.assertEqual(_activity_group("unknown"), "other")


class LiveSessionSummaryTests(unittest.TestCase):
    def _make_history_entry(
        self,
        *,
        message_id: int,
        received_at_ns: int,
        activity: str,
        confidence: float = 0.9,
        cadence_spm: float = 110.0,
        signal_quality_score: float = 85.0,
        intensity_ratio: float = 2.0,
    ) -> dict[str, object]:
        return {
            "message_id": message_id,
            "received_at_ns": received_at_ns,
            "status": "success",
            "activity": activity,
            "confidence": confidence,
            "cadence_spm": cadence_spm,
            "signal_quality_score": signal_quality_score,
            "intensity_level": "moderate",
            "intensity_ratio": intensity_ratio,
            "posture_change_detected": False,
            "burst_detected": False,
        }

    def test_summary_folds_short_locomotion_burst_inside_static(self) -> None:
        summary = build_live_session_summary(
            [
                {
                    "message_id": 1,
                    "received_at_ns": 1_000_000_000,
                    "status": "success",
                    "activity": "sit/lay",
                    "confidence": 0.97,
                    "cadence_spm": 0.0,
                    "signal_quality_score": 90.0,
                    "intensity_level": "still",
                    "intensity_ratio": 1.0,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
                {
                    "message_id": 2,
                    "received_at_ns": 1_700_000_000,
                    "status": "success",
                    "activity": "walk",
                    "confidence": 0.41,
                    "cadence_spm": 104.0,
                    "signal_quality_score": 42.0,
                    "intensity_level": "moderate",
                    "intensity_ratio": 2.1,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
                {
                    "message_id": 3,
                    "received_at_ns": 2_400_000_000,
                    "status": "success",
                    "activity": "sit/lay",
                    "confidence": 0.95,
                    "cadence_spm": 0.0,
                    "signal_quality_score": 88.0,
                    "intensity_level": "still",
                    "intensity_ratio": 1.0,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
            ],
            finalized_at_ns=3_000_000_000,
        )

        self.assertEqual(summary["status"], "finalized")
        self.assertEqual(summary["dominant_activity"], "sit/lay")
        self.assertGreater(summary["noise_bouts_folded"], 0)
        self.assertEqual(len(summary["activity_bouts"]), 1)

    def test_summary_folds_short_locomotion_burst_between_different_static_bouts(self) -> None:
        summary = build_live_session_summary(
            [
                {
                    "message_id": 1,
                    "received_at_ns": 1_000_000_000,
                    "status": "success",
                    "activity": "sit/lay",
                    "confidence": 0.97,
                    "cadence_spm": 0.0,
                    "signal_quality_score": 90.0,
                    "intensity_level": "still",
                    "intensity_ratio": 1.0,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
                {
                    "message_id": 2,
                    "received_at_ns": 1_700_000_000,
                    "status": "success",
                    "activity": "walk",
                    "confidence": 0.41,
                    "cadence_spm": 104.0,
                    "signal_quality_score": 42.0,
                    "intensity_level": "moderate",
                    "intensity_ratio": 2.1,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
                {
                    "message_id": 3,
                    "received_at_ns": 2_400_000_000,
                    "status": "success",
                    "activity": "stand",
                    "confidence": 0.95,
                    "cadence_spm": 0.0,
                    "signal_quality_score": 88.0,
                    "intensity_level": "still",
                    "intensity_ratio": 1.0,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
                {
                    "message_id": 4,
                    "received_at_ns": 3_100_000_000,
                    "status": "success",
                    "activity": "stand",
                    "confidence": 0.96,
                    "cadence_spm": 0.0,
                    "signal_quality_score": 89.0,
                    "intensity_level": "still",
                    "intensity_ratio": 1.0,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
            ],
            finalized_at_ns=3_800_000_000,
        )

        self.assertEqual(summary["status"], "finalized")
        self.assertEqual(summary["dominant_activity"], "stand")
        self.assertGreater(summary["noise_bouts_folded"], 0)
        self.assertNotIn("walk", summary["time_by_activity"])

    def test_summary_folds_short_stairs_burst_inside_walk(self) -> None:
        summary = build_live_session_summary(
            [
                {
                    "message_id": 1,
                    "received_at_ns": 1_000_000_000,
                    "status": "success",
                    "activity": "walk",
                    "confidence": 0.92,
                    "cadence_spm": 111.0,
                    "signal_quality_score": 86.0,
                    "intensity_level": "moderate",
                    "intensity_ratio": 2.0,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
                {
                    "message_id": 2,
                    "received_at_ns": 1_700_000_000,
                    "status": "success",
                    "activity": "stairs",
                    "confidence": 0.46,
                    "cadence_spm": 118.0,
                    "signal_quality_score": 48.0,
                    "intensity_level": "moderate",
                    "intensity_ratio": 2.3,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
                {
                    "message_id": 3,
                    "received_at_ns": 2_400_000_000,
                    "status": "success",
                    "activity": "walk",
                    "confidence": 0.91,
                    "cadence_spm": 110.0,
                    "signal_quality_score": 85.0,
                    "intensity_level": "moderate",
                    "intensity_ratio": 2.0,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
            ],
            finalized_at_ns=3_000_000_000,
        )

        self.assertEqual(summary["dominant_activity"], "walk")
        self.assertGreater(summary["noise_bouts_folded"], 0)
        self.assertEqual(len(summary["activity_bouts"]), 1)
        self.assertGreater(summary["walking_seconds"], 0.0)

    def test_summary_folds_short_stairs_tail_after_walk(self) -> None:
        summary = build_live_session_summary(
            [
                {
                    "message_id": 1,
                    "received_at_ns": 1_000_000_000,
                    "status": "success",
                    "activity": "walk",
                    "confidence": 0.92,
                    "cadence_spm": 111.0,
                    "signal_quality_score": 86.0,
                    "intensity_level": "moderate",
                    "intensity_ratio": 2.0,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
                {
                    "message_id": 2,
                    "received_at_ns": 1_700_000_000,
                    "status": "success",
                    "activity": "walk",
                    "confidence": 0.91,
                    "cadence_spm": 110.0,
                    "signal_quality_score": 85.0,
                    "intensity_level": "moderate",
                    "intensity_ratio": 2.0,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
                {
                    "message_id": 3,
                    "received_at_ns": 2_400_000_000,
                    "status": "success",
                    "activity": "stairs",
                    "confidence": 0.46,
                    "cadence_spm": 118.0,
                    "signal_quality_score": 48.0,
                    "intensity_level": "moderate",
                    "intensity_ratio": 2.3,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
                {
                    "message_id": 4,
                    "received_at_ns": 3_100_000_000,
                    "status": "success",
                    "activity": "stairs",
                    "confidence": 0.44,
                    "cadence_spm": 119.0,
                    "signal_quality_score": 46.0,
                    "intensity_level": "moderate",
                    "intensity_ratio": 2.4,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
            ],
            finalized_at_ns=3_800_000_000,
        )

        self.assertEqual(summary["dominant_activity"], "walk")
        self.assertGreater(summary["noise_bouts_folded"], 0)
        self.assertEqual(len(summary["activity_bouts"]), 1)
        self.assertNotIn("stairs", summary["time_by_activity"])

    def test_summary_folds_short_walk_burst_inside_run(self) -> None:
        summary = build_live_session_summary(
            [
                self._make_history_entry(message_id=1, received_at_ns=1_000_000_000, activity="run", cadence_spm=168.0, intensity_ratio=3.0),
                self._make_history_entry(message_id=2, received_at_ns=1_700_000_000, activity="walk", cadence_spm=112.0, confidence=0.45, signal_quality_score=55.0),
                self._make_history_entry(message_id=3, received_at_ns=2_400_000_000, activity="run", cadence_spm=170.0, intensity_ratio=3.1),
            ],
            finalized_at_ns=3_000_000_000,
        )

        self.assertEqual(summary["dominant_activity"], "run")
        self.assertGreater(summary["noise_bouts_folded"], 0)
        self.assertEqual(len(summary["activity_bouts"]), 1)
        self.assertNotIn("walk", summary["time_by_activity"])

    def test_summary_folds_short_run_burst_inside_stairs(self) -> None:
        summary = build_live_session_summary(
            [
                self._make_history_entry(message_id=1, received_at_ns=1_000_000_000, activity="stairs", cadence_spm=124.0, intensity_ratio=2.5),
                self._make_history_entry(message_id=2, received_at_ns=1_700_000_000, activity="run", cadence_spm=172.0, confidence=0.44, signal_quality_score=52.0, intensity_ratio=3.0),
                self._make_history_entry(message_id=3, received_at_ns=2_400_000_000, activity="stairs", cadence_spm=123.0, intensity_ratio=2.4),
            ],
            finalized_at_ns=3_000_000_000,
        )

        self.assertEqual(summary["dominant_activity"], "stairs")
        self.assertGreater(summary["noise_bouts_folded"], 0)
        self.assertEqual(len(summary["activity_bouts"]), 1)
        self.assertNotIn("run", summary["time_by_activity"])


if __name__ == "__main__":
    unittest.main()
