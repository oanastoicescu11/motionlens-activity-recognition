import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.insights import LiveInsightsAggregator, compute_live_window_insights


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
        )

        insight = compute_live_window_insights(
            feature_row=features,
            feature_names=_schema(),
            baseline_body_mag_energy=1.0,
        )

        self.assertEqual(insight.intensity_level, "moderate")
        self.assertGreater(insight.cadence_spm, 100.0)
        self.assertGreater(insight.cadence_consistency_pct, 75.0)
        self.assertFalse(insight.burst_detected)
        self.assertFalse(insight.posture_change_detected)

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
        )
        insight = compute_live_window_insights(
            feature_row=features,
            feature_names=_schema(),
            baseline_body_mag_energy=1.0,
        )
        self.assertTrue(insight.posture_change_detected)
        self.assertGreaterEqual(insight.orientation_change_deg, 24.0 - 1e-6)

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
        )

        agg.update(still)
        agg.update(moderate)
        agg.update(high_burst)
        recap = agg.recap()

        self.assertAlmostEqual(recap.still_seconds, 1.28, places=6)
        self.assertAlmostEqual(recap.active_seconds, 2.56, places=6)
        self.assertEqual(recap.transition_count, 2)
        self.assertEqual(recap.burst_count, 1)
        self.assertEqual(recap.posture_change_count, 1)
        self.assertGreater(recap.cadence_mean_spm, 90.0)

    def test_raises_for_invalid_baseline(self) -> None:
        with self.assertRaises(ValueError):
            compute_live_window_insights(
                feature_row=np.zeros(len(_schema()), dtype=np.float64),
                feature_names=_schema(),
                baseline_body_mag_energy=0.0,
            )


if __name__ == "__main__":
    unittest.main()
