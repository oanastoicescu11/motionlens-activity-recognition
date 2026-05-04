"""Training-only transition statistics utilities.

These helpers are used to estimate decoder transition priors from labeled training
sequences before saving model artifacts.
"""

from __future__ import annotations

import math

import numpy as np

from core.inference.model import FINE_TO_COARSE


def estimate_transition_stats(
    y_train: np.ndarray,
    sequence_groups: list[str],
    num_classes: int,
    label_order: list[str],
    sticky_prior: float,
    cross_coarse_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate initial-state and transition log-probabilities for causal decoding.

    This function is intentionally training-only: live inference consumes persisted
    transition probabilities from artifacts and does not re-estimate them online.
    """
    if y_train.size != len(sequence_groups):
        raise ValueError("Training labels and sequence group metadata lengths do not match.")

    init_counts = np.ones(num_classes, dtype=np.float64)
    trans_counts = np.ones((num_classes, num_classes), dtype=np.float64)
    run_length_sums = np.ones(num_classes, dtype=np.float64)
    run_length_counts = np.ones(num_classes, dtype=np.float64)

    prev_group = None
    prev_label = None
    current_run_len = 0
    for idx, group in enumerate(sequence_groups):
        label = int(y_train[idx])
        if group != prev_group:
            if prev_label is not None and current_run_len > 0:
                run_length_sums[int(prev_label)] += float(current_run_len)
                run_length_counts[int(prev_label)] += 1.0
            init_counts[label] += 1.0
            prev_group = group
            prev_label = label
            current_run_len = 1
            continue

        trans_counts[int(prev_label), label] += 1.0
        if label == int(prev_label):
            current_run_len += 1
        else:
            run_length_sums[int(prev_label)] += float(current_run_len)
            run_length_counts[int(prev_label)] += 1.0
            current_run_len = 1
        prev_label = label

    if prev_label is not None and current_run_len > 0:
        run_length_sums[int(prev_label)] += float(current_run_len)
        run_length_counts[int(prev_label)] += 1.0

    init_probs = init_counts / np.sum(init_counts)
    trans_probs = trans_counts / np.sum(trans_counts, axis=1, keepdims=True)

    sticky = float(np.clip(sticky_prior, 0.0, 1.0))
    if sticky > 0.0:
        mean_run = run_length_sums / np.maximum(run_length_counts, 1e-8)
        target_stay = np.clip((mean_run - 1.0) / np.maximum(mean_run, 1e-8), 0.1, 0.995)
        for cls_idx in range(num_classes):
            row = trans_probs[cls_idx].copy()
            base_stay = float(row[cls_idx])
            stay = (1.0 - sticky) * base_stay + sticky * float(target_stay[cls_idx])
            off_sum = float(np.sum(row) - base_stay)
            if off_sum <= 1e-12:
                row = np.full(num_classes, (1.0 - stay) / max(1, num_classes - 1), dtype=np.float64)
                row[cls_idx] = stay
            else:
                scale = (1.0 - stay) / off_sum
                row *= scale
                row[cls_idx] = stay
            trans_probs[cls_idx] = row

    coarse_penalty = max(0.0, float(cross_coarse_penalty))
    if coarse_penalty > 0.0:
        label_to_coarse = {label: FINE_TO_COARSE.get(label, "other") for label in label_order}
        for src_idx, src_label in enumerate(label_order):
            src_coarse = label_to_coarse[src_label]
            if src_coarse == "transition":
                continue
            for dst_idx, dst_label in enumerate(label_order):
                if dst_idx == src_idx:
                    continue
                dst_coarse = label_to_coarse[dst_label]
                if dst_coarse == "transition":
                    continue
                if dst_coarse != src_coarse:
                    trans_probs[src_idx, dst_idx] *= math.exp(-coarse_penalty)
            row_sum = float(np.sum(trans_probs[src_idx]))
            if row_sum > 0.0:
                trans_probs[src_idx] /= row_sum

    return np.log(np.clip(init_probs, 1e-12, 1.0)), np.log(np.clip(trans_probs, 1e-12, 1.0))