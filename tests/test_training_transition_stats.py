import unittest

import numpy as np

from processing.training_transition_stats import estimate_transition_stats


class TrainingTransitionStatsTests(unittest.TestCase):
    def test_raises_when_group_lengths_do_not_match(self) -> None:
        y_train = np.asarray([0, 1, 1], dtype=np.int32)
        sequence_groups = ["g1", "g1"]

        with self.assertRaises(ValueError):
            estimate_transition_stats(
                y_train=y_train,
                sequence_groups=sequence_groups,
                num_classes=2,
                label_order=["walk", "sit"],
                sticky_prior=0.0,
                cross_coarse_penalty=0.0,
            )

    def test_returns_valid_log_probability_matrices(self) -> None:
        y_train = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int32)
        sequence_groups = ["s1", "s1", "s1", "s2", "s2", "s2"]

        log_init, log_trans = estimate_transition_stats(
            y_train=y_train,
            sequence_groups=sequence_groups,
            num_classes=2,
            label_order=["walk", "sit"],
            sticky_prior=0.0,
            cross_coarse_penalty=0.0,
        )

        self.assertEqual(log_init.shape, (2,))
        self.assertEqual(log_trans.shape, (2, 2))
        self.assertTrue(np.isfinite(log_init).all())
        self.assertTrue(np.isfinite(log_trans).all())

        init = np.exp(log_init)
        trans = np.exp(log_trans)

        self.assertAlmostEqual(float(np.sum(init)), 1.0, places=6)
        self.assertTrue(np.allclose(np.sum(trans, axis=1), np.ones(2), atol=1e-6))

    def test_sticky_prior_increases_self_transition_for_persistent_class(self) -> None:
        # Sticky prior should move stay probability toward the duration-derived target.
        y_train = np.asarray([0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int32)
        sequence_groups = ["s1"] * len(y_train)

        _, log_trans_no_sticky = estimate_transition_stats(
            y_train=y_train,
            sequence_groups=sequence_groups,
            num_classes=2,
            label_order=["walk", "sit"],
            sticky_prior=0.0,
            cross_coarse_penalty=0.0,
        )
        _, log_trans_sticky = estimate_transition_stats(
            y_train=y_train,
            sequence_groups=sequence_groups,
            num_classes=2,
            label_order=["walk", "sit"],
            sticky_prior=0.8,
            cross_coarse_penalty=0.0,
        )

        trans_no_sticky = np.exp(log_trans_no_sticky)
        trans_sticky = np.exp(log_trans_sticky)

        # Recompute the target-stay statistic for class 0 with the same smoothing
        # scheme used by the estimator (ones initialization).
        run_length_sum = 1.0
        run_length_count = 1.0
        run_len = 0
        for label in y_train:
            if int(label) == 0:
                run_len += 1
            elif run_len > 0:
                run_length_sum += float(run_len)
                run_length_count += 1.0
                run_len = 0
        if run_len > 0:
            run_length_sum += float(run_len)
            run_length_count += 1.0

        mean_run = run_length_sum / run_length_count
        target_stay = float(np.clip((mean_run - 1.0) / max(mean_run, 1e-8), 0.1, 0.995))

        self.assertLess(
            abs(float(trans_sticky[0, 0]) - target_stay),
            abs(float(trans_no_sticky[0, 0]) - target_stay),
        )

    def test_cross_coarse_penalty_reduces_cross_group_transitions_but_not_transition_row(self) -> None:
        label_order = ["walk", "run", "sit", "transitions"]
        y_train = np.asarray(
            [
                0, 0, 2, 0,  # walk->sit and sit->walk observed
                3, 0, 3, 2,  # transitions as source/target present
                1, 1, 2, 1,  # run<->sit observed
            ],
            dtype=np.int32,
        )
        sequence_groups = ["s1"] * len(y_train)

        _, log_trans_unpenalized = estimate_transition_stats(
            y_train=y_train,
            sequence_groups=sequence_groups,
            num_classes=4,
            label_order=label_order,
            sticky_prior=0.0,
            cross_coarse_penalty=0.0,
        )
        _, log_trans_penalized = estimate_transition_stats(
            y_train=y_train,
            sequence_groups=sequence_groups,
            num_classes=4,
            label_order=label_order,
            sticky_prior=0.0,
            cross_coarse_penalty=1.0,
        )

        trans_unpenalized = np.exp(log_trans_unpenalized)
        trans_penalized = np.exp(log_trans_penalized)

        # walk (locomotion) -> sit (static) is cross-coarse and should decrease.
        self.assertLess(float(trans_penalized[0, 2]), float(trans_unpenalized[0, 2]))

        # transitions row is exempt from coarse penalty logic.
        self.assertTrue(np.allclose(trans_penalized[3], trans_unpenalized[3], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
