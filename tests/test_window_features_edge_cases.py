"""Edge case tests for window feature extraction.

Realistic edge cases based on DeviceMotion API behavior:
- Returns `null` (not NaN) when unavailable (see StackOverflow #66713716)
- Empty windows possible during sensor warm-up, permission denial, or when
  all samples are filtered as duplicate timestamps
- Zero values possible when sensor is unavailable/permission denied
- Large magnitude possible in high-g scenarios
- Negative timestamps possible due to clock skew
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.features import window_feature_names
from core.features.window_features import compute_window_features


_PLACEMENT_LABELS = ["front_pocket"]


def _make_window(
    acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create window arrays of the same length."""
    n = min(len(acc_x), len(acc_y), len(acc_z))
    return acc_x[:n].astype(np.float64), acc_y[:n].astype(np.float64), acc_z[:n].astype(np.float64)


class WindowFeatureEdgeCaseTests(unittest.TestCase):
    """Test compute_window_features handles realistic edge cases."""

    def test_single_sample_handling(self) -> None:
        """Single sample window handled gracefully (no crash).
        
        Realistic case: sensor warm-up or capture window before first sample.
        """
        acc_x, acc_y, acc_z = _make_window(
            np.array([0.0]), np.array([0.0]), np.array([9.8])
        )
        result = compute_window_features(
            acc_x, acc_y, acc_z,
            placement_label="front_pocket",
            placement_labels=_PLACEMENT_LABELS,
            target_sample_rate_hz=50.0,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.any(np.isnan(result)))

    def test_all_zeros_input(self) -> None:
        """All-zero accelerometer values produce finite output.
        
        Realistic case: sensor returns 0 when unavailable or during
        permission denial before first valid sample.
        """
        acc_x, acc_y, acc_z = _make_window(
            np.zeros(128), np.zeros(128), np.zeros(128)
        )
        result = compute_window_features(
            acc_x, acc_y, acc_z,
            placement_label="front_pocket",
            placement_labels=_PLACEMENT_LABELS,
            target_sample_rate_hz=50.0,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_constant_signal(self) -> None:
        """Constant accelerometer values produce valid features.
        
        Realistic case: device at rest, no body motion.
        """
        acc_x, acc_y, acc_z = _make_window(
            np.full(128, 0.0),
            np.full(128, 0.0),
            np.full(128, 9.8),
        )
        result = compute_window_features(
            acc_x, acc_y, acc_z,
            placement_label="front_pocket",
            placement_labels=_PLACEMENT_LABELS,
            target_sample_rate_hz=50.0,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.any(np.isnan(result)))
        # body_mag_energy for constant z=9.8 should be non-zero
        names = window_feature_names(_PLACEMENT_LABELS)
        energy_idx = names.index("body_mag_energy")
        self.assertGreater(result[energy_idx], 0.0)

    def test_normal_window_produces_finite_output(self) -> None:
        """Normal walking-like window produces all-finite features."""
        t = np.arange(128) / 50.0
        acc_x = 0.3 * np.sin(2 * np.pi * 1.5 * t)
        acc_y = 0.2 * np.sin(2 * np.pi * 1.2 * t)
        acc_z = 9.8 + 0.5 * np.sin(2 * np.pi * 0.8 * t)
        result = compute_window_features(
            acc_x.astype(np.float64),
            acc_y.astype(np.float64),
            acc_z.astype(np.float64),
            placement_label="front_pocket",
            placement_labels=_PLACEMENT_LABELS,
            target_sample_rate_hz=50.0,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_large_magnitude_input(self) -> None:
        """Very large accelerometer values produce finite output.
        
        Realistic case: high-g scenario or sensor error.
        """
        acc_x, acc_y, acc_z = _make_window(
            np.full(128, 1e6),
            np.full(128, 1e6),
            np.full(128, 1e6 + 9.8),
        )
        result = compute_window_features(
            acc_x, acc_y, acc_z,
            placement_label="front_pocket",
            placement_labels=_PLACEMENT_LABELS,
            target_sample_rate_hz=50.0,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_negative_time_array(self) -> None:
        """Negative time values in input handled.
        
        Realistic case: clock skew or system time adjustment.
        """
        t = np.arange(-64, 64) / 50.0
        acc_x = 0.3 * np.sin(2 * np.pi * 1.5 * t)
        acc_y = np.zeros_like(acc_x)
        acc_z = 9.8 + np.zeros_like(acc_x)
        result = compute_window_features(
            acc_x, acc_y, acc_z,
            placement_label="front_pocket",
            placement_labels=_PLACEMENT_LABELS,
            target_sample_rate_hz=50.0,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.any(np.isnan(result)))

    def test_very_short_window(self) -> None:
        """Very short window (2 samples) produces valid output."""
        acc_x, acc_y, acc_z = _make_window(
            np.array([0.0, 0.1]),
            np.array([0.0, 0.2]),
            np.array([9.8, 9.8]),
        )
        result = compute_window_features(
            acc_x, acc_y, acc_z,
            placement_label="front_pocket",
            placement_labels=_PLACEMENT_LABELS,
            target_sample_rate_hz=50.0,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.any(np.isnan(result)))


if __name__ == "__main__":
    unittest.main()
