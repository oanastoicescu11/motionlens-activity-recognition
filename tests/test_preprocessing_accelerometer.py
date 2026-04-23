import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from processing.preprocessing import (
    AccelerometerSegment,
    apply_gravity_split,
    preprocess_numeric_timeseries,
    split_timeseries_by_gaps,
)


class AccelerometerPreprocessingTests(unittest.TestCase):
    def test_split_timeseries_by_gaps_reanchors_segments(self) -> None:
        segment = AccelerometerSegment(
            time_seconds=np.array([0.0, 0.02, 0.04, 0.40, 0.42], dtype=np.float64),
            acc_x=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
            acc_y=np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
            acc_z=np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64),
        )

        segments = split_timeseries_by_gaps(segment, max_gap_seconds=0.25)

        self.assertEqual(len(segments), 2)
        np.testing.assert_allclose(segments[0].time_seconds, np.array([0.0, 0.02, 0.04]))
        np.testing.assert_allclose(segments[1].time_seconds, np.array([0.0, 0.02]))
        np.testing.assert_allclose(segments[1].acc_x, np.array([4.0, 5.0]))

    def test_preprocess_numeric_timeseries_splits_long_gaps_before_resampling(self) -> None:
        preprocessed = preprocess_numeric_timeseries(
            time_seconds=np.array([0.0, 0.04, 0.08, 0.50, 0.54], dtype=np.float64),
            x_values=np.array([0.0, 4.0, 8.0, 10.0, 14.0], dtype=np.float64),
            y_values=np.zeros(5, dtype=np.float64),
            z_values=np.zeros(5, dtype=np.float64),
        )

        self.assertEqual(len(preprocessed), 2)
        np.testing.assert_allclose(
            preprocessed[0].total.time_seconds,
            np.array([0.0, 0.02, 0.04, 0.06, 0.08]),
        )
        np.testing.assert_allclose(
            preprocessed[1].total.time_seconds,
            np.array([0.0, 0.02, 0.04]),
        )

    def test_apply_gravity_split_recovers_low_frequency_gravity_component(self) -> None:
        time_seconds = np.arange(0.0, 20.0, 1.0 / 50.0)
        gravity_component = 9.81 + (0.2 * np.sin(2.0 * np.pi * 0.1 * time_seconds))
        body_component = 0.5 * np.sin(2.0 * np.pi * 1.5 * time_seconds)
        total_signal = gravity_component + body_component

        split = apply_gravity_split(
            AccelerometerSegment(
                time_seconds=time_seconds,
                acc_x=total_signal,
                acc_y=np.zeros_like(total_signal),
                acc_z=np.zeros_like(total_signal),
            )
        )

        window = slice(100, -100)
        np.testing.assert_allclose(
            split.gravity_x[window],
            gravity_component[window],
            atol=0.08,
        )
        np.testing.assert_allclose(
            split.body_x[window],
            body_component[window],
            atol=0.08,
        )


if __name__ == "__main__":
    unittest.main()