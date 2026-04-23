import unittest

import numpy as np

from processing.motionlens_xgboost_baseline import (
    DEFAULT_TARGET_LABELS,
    PLACEMENT_LABELS,
    _window_feature_names,
    compute_window_features,
)


class MotionLensXGBoostBaselineTests(unittest.TestCase):
    def test_default_targets_include_auxiliary_labels(self) -> None:
        self.assertIn("transitions", DEFAULT_TARGET_LABELS)
        self.assertIn("locomotion-other", DEFAULT_TARGET_LABELS)

    def test_feature_vector_has_expected_length_and_placement_onehot(self) -> None:
        t = np.arange(128) / 50.0
        acc_x = np.sin(2.0 * np.pi * 1.2 * t)
        acc_y = np.cos(2.0 * np.pi * 1.2 * t)
        acc_z = np.full_like(t, 9.81)

        features = compute_window_features(
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            placement_label="front_center_mid",
        )

        feature_names = _window_feature_names()
        self.assertEqual(features.shape[0], len(feature_names))
        self.assertTrue(np.isfinite(features).all())

        placement_names = [f"placement_{label}" for label in PLACEMENT_LABELS]
        placement_indices = [feature_names.index(name) for name in placement_names]
        placement_values = features[placement_indices]

        self.assertAlmostEqual(float(np.sum(placement_values)), 1.0, places=6)
        idx_mid = PLACEMENT_LABELS.index("front_center_mid")
        self.assertEqual(float(placement_values[idx_mid]), 1.0)


if __name__ == "__main__":
    unittest.main()
