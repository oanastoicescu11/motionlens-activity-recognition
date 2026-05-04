"""Unit tests for stateful live causal decoding helpers."""

from __future__ import annotations

import unittest

import numpy as np

from core.inference import InferenceArtifacts, causal_hmm_step
from core.inference import should_reset_on_low_motion


class WorkerDecoderStateTests(unittest.TestCase):
    def _artifacts(self) -> InferenceArtifacts:
        # 2-class dummy artifacts for decoder-only tests.
        return InferenceArtifacts(
            base_model=object(),
            label_order=["a", "b"],
            temporal_decoder="causal-hmm",
            log_init_probs=np.log(np.array([0.5, 0.5], dtype=np.float64)),
            log_trans_probs=np.log(np.array([[0.98, 0.02], [0.05, 0.95]], dtype=np.float64)),
        )

    def test_decoder_uses_previous_posterior_when_present(self) -> None:
        artifacts = self._artifacts()
        emission = np.array([0.40, 0.60], dtype=np.float64)

        # Without previous state, class b should win from emission.
        idx_init, _, source_init = causal_hmm_step(
            emission,
            log_init_probs=artifacts.log_init_probs,
            log_trans_probs=artifacts.log_trans_probs,
            previous_posterior=None,
        )
        self.assertEqual(idx_init, 1)
        self.assertEqual(source_init, "initialized")

        # With strong prior from previous class a state, sticky transition keeps class a.
        prev_state = np.array([0.99, 0.01], dtype=np.float64)
        idx_cont, posterior_cont, source_cont = causal_hmm_step(
            emission,
            log_init_probs=artifacts.log_init_probs,
            log_trans_probs=artifacts.log_trans_probs,
            previous_posterior=prev_state,
        )
        self.assertEqual(idx_cont, 0)
        self.assertEqual(source_cont, "continued")
        self.assertAlmostEqual(float(np.sum(posterior_cont)), 1.0, places=6)

    def test_main_prediction_matches_top_probability_source(self) -> None:
        artifacts = self._artifacts()
        emission = np.array([0.25, 0.75], dtype=np.float64)
        idx, posterior, _ = causal_hmm_step(
            emission,
            log_init_probs=artifacts.log_init_probs,
            log_trans_probs=artifacts.log_trans_probs,
            previous_posterior=None,
        )

        # Main index must be argmax of the same probability vector used for reporting.
        self.assertEqual(idx, int(np.argmax(posterior)))

    def test_should_reset_on_low_motion_with_locomotion_state(self) -> None:
        label_order = ["walk", "run", "sit/lay", "stand"]
        previous_state = {"posterior": [0.50, 0.30, 0.10, 0.10]}
        self.assertTrue(should_reset_on_low_motion(previous_state, label_order=label_order))

    def test_should_reset_on_low_motion_with_static_state(self) -> None:
        label_order = ["walk", "run", "sit/lay", "stand"]
        previous_state = {"posterior": [0.10, 0.10, 0.50, 0.30]}
        self.assertFalse(should_reset_on_low_motion(previous_state, label_order=label_order))

    def test_should_reset_on_low_motion_with_no_state(self) -> None:
        label_order = ["walk", "run", "sit/lay", "stand"]
        self.assertTrue(should_reset_on_low_motion(None, label_order=label_order))


if __name__ == "__main__":
    unittest.main()
