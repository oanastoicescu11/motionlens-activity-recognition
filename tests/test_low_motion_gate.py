"""Tests for the low-motion gate in the inference worker pipeline."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.inference.guards import (
    LOCOMOTION_LABELS as _LOCOMOTION_LABELS,
    LOW_MOTION_ENERGY_THRESHOLD as _LOW_MOTION_ENERGY_THRESHOLD,
    apply_front_pocket_smooth_walk_guard as _apply_front_pocket_smooth_walk_guard,
    apply_pocket_activity_sanity_correction as _apply_pocket_activity_sanity_correction,
    apply_low_motion_gate as _apply_low_motion_gate,
    apply_overconfident_stairs_failsafe as _apply_overconfident_stairs_failsafe,
    apply_stairs_moving_window_guard as _apply_stairs_moving_window_guard,
    apply_stairs_static_guard as _apply_stairs_static_guard,
)
from core.inference.artifact_scoring import compute_loose_pocket_artifact_score as _compute_loose_pocket_artifact_score
from core.features import window_feature_names

_PLACEMENT_LABELS = ["front_pocket"]
_LABEL_ORDER = ["walk", "run", "stairs", "sit", "stand", "lay", "transitions", "locomotion-other"]


def _make_feature_row(body_mag_energy: float) -> np.ndarray:
    names = window_feature_names(_PLACEMENT_LABELS)
    row = np.zeros(len(names), dtype=np.float64)
    row[names.index("body_mag_energy")] = body_mag_energy
    row[names.index("gravity_tilt_abs_z_mean")] = 0.3
    row[names.index("gravity_y_mean")] = -9.8
    return row


def _uniform_emission(n: int | None = None) -> np.ndarray:
    num = len(_LABEL_ORDER) if n is None else int(n)
    return np.full(num, 1.0 / num, dtype=np.float64)


class LowMotionGateTests(unittest.TestCase):
    def test_loose_pocket_artifact_score_increases_for_noisy_profile(self) -> None:
        names = window_feature_names(_PLACEMENT_LABELS)
        row = np.zeros(len(names), dtype=np.float64)
        row[names.index("jerk_mag_p90")] = 5.0
        row[names.index("body_mag_energy_8_20_hz")] = 4.0
        row[names.index("body_mag_energy_0p5_4_hz")] = 1.0
        row[names.index("body_mag_spectral_entropy")] = 4.8
        row[names.index("gravity_angle_stability_std")] = 0.55
        row[names.index("body_mag_autocorr_periodicity_ratio")] = 1.1
        row[names.index("stft_dom_freq_std")] = 0.7

        score, info = _compute_loose_pocket_artifact_score(row, _PLACEMENT_LABELS)
        self.assertGreater(score, 1.0)
        self.assertGreater(info["artifact_hf_ratio"], 1.0)

    def test_pocket_sanity_correction_demotes_run_to_walk(self) -> None:
        proba = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        proba[_LABEL_ORDER.index("run")] = 0.65
        proba[_LABEL_ORDER.index("walk")] = 0.20
        proba[_LABEL_ORDER.index("stairs")] = 0.10
        proba[_LABEL_ORDER.index("sit")] = 0.05

        out, corrected, reason = _apply_pocket_activity_sanity_correction(
            proba,
            label_order=_LABEL_ORDER,
            placement_label="front_pocket",
            cadence_spm=108.0,
            periodicity_strength=0.30,
            artifact_score=1.8,
        )
        self.assertTrue(corrected)
        self.assertIn("run-to-walk", reason)
        self.assertEqual(_LABEL_ORDER[int(np.argmax(out))], "walk")

    def test_pocket_sanity_correction_demotes_low_cadence_run_even_with_moderate_periodicity(self) -> None:
        proba = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        proba[_LABEL_ORDER.index("run")] = 0.60
        proba[_LABEL_ORDER.index("walk")] = 0.28
        proba[_LABEL_ORDER.index("stairs")] = 0.08
        proba[_LABEL_ORDER.index("sit")] = 0.04

        out, corrected, reason = _apply_pocket_activity_sanity_correction(
            proba,
            label_order=_LABEL_ORDER,
            placement_label="front_pocket",
            cadence_spm=58.0,
            periodicity_strength=0.78,
            artifact_score=1.9,
        )
        self.assertTrue(corrected)
        self.assertIn("run-to-walk", reason)
        self.assertEqual(_LABEL_ORDER[int(np.argmax(out))], "walk")

    def test_pocket_sanity_correction_demotes_transitions_to_run_for_locomotion_burst(self) -> None:
        proba = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        proba[_LABEL_ORDER.index("transitions")] = 0.72
        proba[_LABEL_ORDER.index("run")] = 0.20
        proba[_LABEL_ORDER.index("walk")] = 0.06
        proba[_LABEL_ORDER.index("stairs")] = 0.02

        out, corrected, reason = _apply_pocket_activity_sanity_correction(
            proba,
            label_order=_LABEL_ORDER,
            placement_label="front_pocket",
            cadence_spm=182.0,
            periodicity_strength=0.62,
            artifact_score=1.9,
            raw_label="run",
            raw_confidence=0.55,
        )
        self.assertTrue(corrected)
        self.assertIn("transitions-to-run", reason)
        self.assertEqual(_LABEL_ORDER[int(np.argmax(out))], "run")

    def test_gate_does_not_fire_above_threshold(self) -> None:
        feature_row = _make_feature_row(body_mag_energy=0.5)
        emission = _uniform_emission()
        gated, fired = _apply_low_motion_gate(
            feature_row,
            emission,
            _LABEL_ORDER,
            _PLACEMENT_LABELS,
            "front_pocket",
        )
        self.assertFalse(fired)
        np.testing.assert_array_equal(gated, emission)

    def test_gate_fires_below_threshold(self) -> None:
        feature_row = _make_feature_row(body_mag_energy=0.01)
        emission = _uniform_emission()
        gated, fired = _apply_low_motion_gate(
            feature_row,
            emission,
            _LABEL_ORDER,
            _PLACEMENT_LABELS,
            "front_pocket",
        )
        self.assertTrue(fired)

    def test_gate_zeros_locomotion_probabilities(self) -> None:
        feature_row = _make_feature_row(body_mag_energy=0.01)
        emission = _uniform_emission()
        gated, fired = _apply_low_motion_gate(
            feature_row,
            emission,
            _LABEL_ORDER,
            _PLACEMENT_LABELS,
            "front_pocket",
        )
        self.assertTrue(fired)
        for idx, label in enumerate(_LABEL_ORDER):
            if label in _LOCOMOTION_LABELS:
                self.assertAlmostEqual(float(gated[idx]), 0.0, places=10,
                                       msg=f"Expected {label} probability=0 after gate")

    def test_gate_output_sums_to_one(self) -> None:
        feature_row = _make_feature_row(body_mag_energy=0.0)
        emission = _uniform_emission()
        gated, fired = _apply_low_motion_gate(
            feature_row,
            emission,
            _LABEL_ORDER,
            _PLACEMENT_LABELS,
            "front_pocket",
        )
        self.assertTrue(fired)
        self.assertAlmostEqual(float(np.sum(gated)), 1.0, places=10)

    def test_gate_at_exactly_threshold_does_not_fire(self) -> None:
        feature_row = _make_feature_row(body_mag_energy=_LOW_MOTION_ENERGY_THRESHOLD)
        emission = _uniform_emission()
        _, fired = _apply_low_motion_gate(
            feature_row,
            emission,
            _LABEL_ORDER,
            _PLACEMENT_LABELS,
            "front_pocket",
        )
        self.assertFalse(fired)

    def test_gate_redistributes_mass_to_static_classes(self) -> None:
        """Static classes should collectively receive all probability mass after gate fires."""
        feature_row = _make_feature_row(body_mag_energy=0.02)
        # Make run strongly favoured before the gate
        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("run")] = 0.85
        emission[_LABEL_ORDER.index("sit")] = 0.10
        emission[_LABEL_ORDER.index("lay")] = 0.05
        gated, fired = _apply_low_motion_gate(
            feature_row,
            emission,
            _LABEL_ORDER,
            _PLACEMENT_LABELS,
            "front_pocket",
        )
        self.assertTrue(fired)
        static_mass = sum(
            float(gated[idx])
            for idx, label in enumerate(_LABEL_ORDER)
            if label in {"sit", "stand", "lay"}
        )
        self.assertAlmostEqual(static_mass, 1.0, places=10)
        self.assertAlmostEqual(float(gated[_LABEL_ORDER.index("run")]), 0.0, places=10)
        self.assertAlmostEqual(float(gated[_LABEL_ORDER.index("transitions")]), 0.0, places=10)

    def test_gate_does_not_mutate_original_emission(self) -> None:
        feature_row = _make_feature_row(body_mag_energy=0.01)
        emission = _uniform_emission()
        original = emission.copy()
        _apply_low_motion_gate(
            feature_row,
            emission,
            _LABEL_ORDER,
            _PLACEMENT_LABELS,
            "front_pocket",
        )
        np.testing.assert_array_equal(emission, original)

    def test_stairs_guard_caps_stairs_on_static_like_window(self) -> None:
        names = window_feature_names(_PLACEMENT_LABELS)
        row = np.zeros(len(names), dtype=np.float64)
        row[names.index("body_mag_energy")] = 0.1
        row[names.index("jerk_mag_rms")] = 0.2
        row[names.index("body_mag_dom_freq_hz")] = 0.2

        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("stairs")] = 0.6
        emission[_LABEL_ORDER.index("sit")] = 0.3
        emission[_LABEL_ORDER.index("lay")] = 0.1

        out, applied, _ = _apply_stairs_static_guard(
            feature_row=row,
            emission=emission,
            label_order=_LABEL_ORDER,
            placement_labels=_PLACEMENT_LABELS,
        )
        self.assertTrue(applied)
        self.assertLessEqual(float(out[_LABEL_ORDER.index("stairs")]), 0.03 + 1e-9)
        self.assertAlmostEqual(float(np.sum(out)), 1.0, places=10)

    def test_stairs_guard_does_not_apply_for_dynamic_window(self) -> None:
        names = window_feature_names(_PLACEMENT_LABELS)
        row = np.zeros(len(names), dtype=np.float64)
        row[names.index("body_mag_energy")] = 1.2
        row[names.index("jerk_mag_rms")] = 1.5
        row[names.index("body_mag_dom_freq_hz")] = 1.8

        emission = _uniform_emission()
        out, applied, _ = _apply_stairs_static_guard(
            feature_row=row,
            emission=emission,
            label_order=_LABEL_ORDER,
            placement_labels=_PLACEMENT_LABELS,
        )
        self.assertFalse(applied)
        np.testing.assert_array_equal(out, emission)

    def test_stairs_moving_guard_reduces_stairs_when_walkrun_is_strong(self) -> None:
        names = window_feature_names(_PLACEMENT_LABELS)
        row = np.zeros(len(names), dtype=np.float64)
        row[names.index("body_mag_energy")] = 0.8
        row[names.index("body_mag_dom_freq_hz")] = 1.9

        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("stairs")] = 0.70
        emission[_LABEL_ORDER.index("walk")] = 0.22
        emission[_LABEL_ORDER.index("run")] = 0.08
        previous_state = {
            "posterior": [0.12, 0.08, 0.66, 0.04, 0.04, 0.02, 0.02, 0.02],
            "window_end_ns": 1,
        }

        out, applied, info = _apply_stairs_moving_window_guard(
            feature_row=row,
            emission=emission,
            label_order=_LABEL_ORDER,
            placement_labels=_PLACEMENT_LABELS,
            placement_label="front_pocket",
            previous_state=previous_state,
        )
        self.assertTrue(applied)
        self.assertLess(float(out[_LABEL_ORDER.index("stairs")]), float(emission[_LABEL_ORDER.index("stairs")]))
        self.assertGreater(float(info["stairs_moving_guard_smoothed_stairs"]), 0.0)
        self.assertAlmostEqual(float(np.sum(out)), 1.0, places=10)

    def test_stairs_moving_guard_does_not_apply_outside_front_pocket(self) -> None:
        names = window_feature_names(_PLACEMENT_LABELS)
        row = np.zeros(len(names), dtype=np.float64)
        row[names.index("body_mag_energy")] = 0.8
        row[names.index("body_mag_dom_freq_hz")] = 1.9

        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("stairs")] = 0.70
        emission[_LABEL_ORDER.index("walk")] = 0.22
        emission[_LABEL_ORDER.index("run")] = 0.08

        out, applied, _ = _apply_stairs_moving_window_guard(
            feature_row=row,
            emission=emission,
            label_order=_LABEL_ORDER,
            placement_labels=_PLACEMENT_LABELS,
            placement_label="unknown_free_living",
            previous_state=None,
        )
        self.assertFalse(applied)
        np.testing.assert_array_equal(out, emission)

    def test_stairs_failsafe_caps_near_one_decoded_stairs(self) -> None:
        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("stairs")] = 0.70
        emission[_LABEL_ORDER.index("walk")] = 0.22
        emission[_LABEL_ORDER.index("run")] = 0.08

        decoded = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        decoded[_LABEL_ORDER.index("stairs")] = 0.995
        decoded[_LABEL_ORDER.index("walk")] = 0.003
        decoded[_LABEL_ORDER.index("run")] = 0.002

        out, applied, info = _apply_overconfident_stairs_failsafe(
            emission=emission,
            decoded_proba=decoded,
            label_order=_LABEL_ORDER,
            placement_label="front_pocket",
        )
        self.assertTrue(applied)
        self.assertLess(float(out[_LABEL_ORDER.index("stairs")]), float(decoded[_LABEL_ORDER.index("stairs")]))
        self.assertGreater(float(out[_LABEL_ORDER.index("walk")]), float(decoded[_LABEL_ORDER.index("walk")]))
        self.assertGreater(float(out[_LABEL_ORDER.index("run")]), float(decoded[_LABEL_ORDER.index("run")]))
        self.assertGreater(float(info["stairs_failsafe_raw_walkrun"]), 0.0)

    def test_stairs_failsafe_skips_when_raw_stairs_already_strong(self) -> None:
        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("stairs")] = 0.92
        emission[_LABEL_ORDER.index("walk")] = 0.05
        emission[_LABEL_ORDER.index("run")] = 0.03

        decoded = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        decoded[_LABEL_ORDER.index("stairs")] = 0.995
        decoded[_LABEL_ORDER.index("walk")] = 0.003
        decoded[_LABEL_ORDER.index("run")] = 0.002

        out, applied, _ = _apply_overconfident_stairs_failsafe(
            emission=emission,
            decoded_proba=decoded,
            label_order=_LABEL_ORDER,
            placement_label="front_pocket",
        )
        self.assertFalse(applied)
        np.testing.assert_array_equal(out, decoded)

    def test_walk_guard_reduces_run_and_stairs_for_smooth_walk(self) -> None:
        names = window_feature_names(_PLACEMENT_LABELS)
        row = np.zeros(len(names), dtype=np.float64)
        row[names.index("body_mag_energy")] = 0.9
        row[names.index("body_mag_dom_freq_hz")] = 1.9
        row[names.index("jerk_mag_rms")] = 0.6

        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("walk")] = 0.35
        emission[_LABEL_ORDER.index("run")] = 0.33
        emission[_LABEL_ORDER.index("stairs")] = 0.28
        emission[_LABEL_ORDER.index("sit")] = 0.02
        emission[_LABEL_ORDER.index("lay")] = 0.02

        out, applied, _ = _apply_front_pocket_smooth_walk_guard(
            feature_row=row,
            emission=emission,
            label_order=_LABEL_ORDER,
            placement_labels=_PLACEMENT_LABELS,
            placement_label="front_pocket",
        )
        self.assertTrue(applied)
        self.assertGreater(float(out[_LABEL_ORDER.index("walk")]), float(emission[_LABEL_ORDER.index("walk")]))
        self.assertLess(float(out[_LABEL_ORDER.index("run")]), float(emission[_LABEL_ORDER.index("run")]))
        self.assertLess(float(out[_LABEL_ORDER.index("stairs")]), float(emission[_LABEL_ORDER.index("stairs")]))

    def test_walk_guard_skips_when_not_smooth_walk_like(self) -> None:
        names = window_feature_names(_PLACEMENT_LABELS)
        row = np.zeros(len(names), dtype=np.float64)
        row[names.index("body_mag_energy")] = 2.6
        row[names.index("body_mag_dom_freq_hz")] = 2.8
        row[names.index("jerk_mag_rms")] = 2.1

        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("walk")] = 0.35
        emission[_LABEL_ORDER.index("run")] = 0.33
        emission[_LABEL_ORDER.index("stairs")] = 0.28
        emission[_LABEL_ORDER.index("sit")] = 0.02
        emission[_LABEL_ORDER.index("lay")] = 0.02

        out, applied, _ = _apply_front_pocket_smooth_walk_guard(
            feature_row=row,
            emission=emission,
            label_order=_LABEL_ORDER,
            placement_labels=_PLACEMENT_LABELS,
            placement_label="front_pocket",
        )
        self.assertFalse(applied)
        np.testing.assert_array_equal(out, emission)

    def test_front_pocket_high_tilt_does_not_override_static_distribution(self) -> None:
        feature_row = _make_feature_row(body_mag_energy=0.01)
        names = window_feature_names(_PLACEMENT_LABELS)
        feature_row[names.index("gravity_tilt_abs_z_mean")] = 0.8
        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("sit")] = 0.1
        emission[_LABEL_ORDER.index("lay")] = 0.9
        gated, fired = _apply_low_motion_gate(
            feature_row,
            emission,
            _LABEL_ORDER,
            _PLACEMENT_LABELS,
            "front_pocket",
        )
        self.assertTrue(fired)
        self.assertLess(float(gated[_LABEL_ORDER.index("sit")]), float(gated[_LABEL_ORDER.index("lay")]))

    def test_front_pocket_low_tilt_does_not_override_static_distribution(self) -> None:
        feature_row = _make_feature_row(body_mag_energy=0.01)
        names = window_feature_names(_PLACEMENT_LABELS)
        feature_row[names.index("gravity_tilt_abs_z_mean")] = 0.05
        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("sit")] = 0.9
        emission[_LABEL_ORDER.index("lay")] = 0.1
        gated, fired = _apply_low_motion_gate(
            feature_row,
            emission,
            _LABEL_ORDER,
            _PLACEMENT_LABELS,
            "front_pocket",
        )
        self.assertTrue(fired)
        self.assertLess(float(gated[_LABEL_ORDER.index("lay")]), float(gated[_LABEL_ORDER.index("sit")]))

    def test_upside_down_orientation_has_no_effect_on_static_distribution_high_pitch(self) -> None:
        feature_row = _make_feature_row(body_mag_energy=0.01)
        names = window_feature_names(_PLACEMENT_LABELS)
        feature_row[names.index("gravity_tilt_abs_z_mean")] = 0.8
        feature_row[names.index("gravity_y_mean")] = 9.8
        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("sit")] = 0.1
        emission[_LABEL_ORDER.index("lay")] = 0.9
        gated, fired = _apply_low_motion_gate(
            feature_row,
            emission,
            _LABEL_ORDER,
            _PLACEMENT_LABELS,
            "front_pocket",
        )
        self.assertTrue(fired)
        self.assertLess(float(gated[_LABEL_ORDER.index("sit")]), float(gated[_LABEL_ORDER.index("lay")]))

    def test_upside_down_orientation_has_no_effect_on_static_distribution_low_pitch(self) -> None:
        feature_row = _make_feature_row(body_mag_energy=0.01)
        names = window_feature_names(_PLACEMENT_LABELS)
        feature_row[names.index("gravity_tilt_abs_z_mean")] = 0.05
        feature_row[names.index("gravity_y_mean")] = 9.8
        emission = np.zeros(len(_LABEL_ORDER), dtype=np.float64)
        emission[_LABEL_ORDER.index("sit")] = 0.9
        emission[_LABEL_ORDER.index("lay")] = 0.1
        gated, fired = _apply_low_motion_gate(
            feature_row,
            emission,
            _LABEL_ORDER,
            _PLACEMENT_LABELS,
            "front_pocket",
        )
        self.assertTrue(fired)
        self.assertGreater(float(gated[_LABEL_ORDER.index("sit")]), float(gated[_LABEL_ORDER.index("lay")]))


if __name__ == "__main__":
    unittest.main()
