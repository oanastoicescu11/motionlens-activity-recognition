"""Cadence and periodicity feature extraction from accelerometer windows."""

from __future__ import annotations

import numpy as np

from .window_features import window_feature_names


def extract_cadence_and_periodicity(
    feature_row: np.ndarray,
    placement_labels: list[str],
) -> tuple[float, float]:
    """Compute cadence in steps-per-minute and normalized periodicity strength [0, 1].

    Returns (cadence_spm, periodicity_strength).
    cadence_spm is 0.0 when no valid locomotion frequency is detected.
    periodicity_strength is clipped to [0, 1] from the autocorrelation ratio.
    """
    feature_names = window_feature_names(placement_labels)
    try:
        auto_freq_idx = feature_names.index("body_mag_autocorr_peak_freq_hz")
        dom_freq_idx = feature_names.index("body_mag_dom_freq_hz")
        periodicity_idx = feature_names.index("body_mag_autocorr_periodicity_ratio")
    except ValueError:
        return 0.0, 0.0

    auto_hz = float(feature_row[auto_freq_idx])
    dom_hz = float(feature_row[dom_freq_idx])
    if 0.5 <= auto_hz <= 4.0:
        cadence_hz = auto_hz
    elif 0.5 <= dom_hz <= 4.0:
        cadence_hz = dom_hz
    else:
        cadence_hz = 0.0

    periodicity_ratio = float(feature_row[periodicity_idx])
    periodicity_strength = float(np.clip((periodicity_ratio - 1.0) / 2.0, 0.0, 1.0))
    return cadence_hz * 60.0, periodicity_strength
