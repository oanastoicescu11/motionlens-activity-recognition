"""Activity guard functions for real-time inference post-processing.

Guards are applied in sequence to emission probabilities before and after
HMM decoding to correct systematic biases from loose-pocket placement,
low-motion conditions, and erratic signal artifacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.features.window_features import window_feature_names
from core.features.cadence import extract_cadence_and_periodicity  # noqa: F401 — re-exported
from core.inference.model import extract_decoder_posterior
from core.inference.artifact_scoring import FRONT_POCKET_LABEL, POCKET_ARTIFACT_SCORE_THRESHOLD

# Low-motion gate constants
LOW_MOTION_ENERGY_THRESHOLD = 0.1
LOCOMOTION_LABELS = frozenset({"walk", "run", "stairs", "locomotion-other"})
STATIC_LABELS = frozenset({"sit/lay", "stand"})
# All labels suppressed in the low-motion gate: locomotion plus transitions.
# Transitions require bodily movement; a near-motionless window cannot be a transition.
_MOTION_REQUIRING_LABELS = frozenset({"walk", "run", "stairs", "locomotion-other", "transitions"})

# Stairs static guard
_STAIRS_GUARD_ENERGY_MAX = 0.35
_STAIRS_GUARD_JERK_MAX = 0.9
_STAIRS_GUARD_DOM_FREQ_MAX = 0.7
_STAIRS_GUARD_MAX_PROBA = 0.03

# Stairs moving window guard
_STAIRS_MOVING_GUARD_MIN_ENERGY = 0.35
_STAIRS_MOVING_GUARD_MIN_DOM_FREQ = 1.2
_STAIRS_MOVING_GUARD_MAX_DOM_FREQ = 3.2
_STAIRS_MOVING_GUARD_PREV_BLEND = 0.40
_STAIRS_MOVING_GUARD_MIN_SMOOTHED_STAIRS = 0.35
_STAIRS_MOVING_GUARD_MIN_WALKRUN_MASS = 0.20
_STAIRS_MOVING_GUARD_MAX_PROBA = 0.22

# Decoder safety failsafe
_STAIRS_FAILSAFE_MIN_DECODED_STAIRS = 0.70
_STAIRS_FAILSAFE_MAX_DECODED_STAIRS = 0.55
_STAIRS_FAILSAFE_MIN_RAW_WALKRUN = 0.15
_STAIRS_FAILSAFE_MAX_RAW_STAIRS = 0.90

# Smooth-walk guard
_WALK_GUARD_MIN_ENERGY = 0.30
_WALK_GUARD_MAX_ENERGY = 1.80
_WALK_GUARD_MIN_DOM_FREQ = 1.35
_WALK_GUARD_MAX_DOM_FREQ = 2.45
_WALK_GUARD_MAX_JERK = 1.20
_WALK_GUARD_MIN_WALK = 0.18
_WALK_GUARD_MAX_RUN = 0.45
_WALK_GUARD_MAX_STAIRS = 0.45
_WALK_GUARD_CAP_RUN = 0.22
_WALK_GUARD_CAP_STAIRS = 0.18

# Stairs hallucination guard
_STAIRS_HALLUCINATION_MAX_ENERGY = 8.0
_STAIRS_HALLUCINATION_MAX_JERK = 40.0
_STAIRS_HALLUCINATION_MAX_STAIRS_PROBA = 0.05

# Locomotion-transitions demotion guard
# When body_mag_energy is high, a "transitions" prediction is locomotion with direction changes,
# not a genuine postural transition (sit→stand). Genuine postural transitions have energy << 0.1 m/s².
_LOCOMOTION_TRANSITIONS_ENERGY_MIN = 5.0   # m/s²: below this, transitions may be genuine
_LOCOMOTION_TRANSITIONS_RUN_ENERGY_MIN = 12.0  # m/s²: above this, demote to run; below, to walk

# Pocket sanity demotion thresholds (used in apply_pocket_activity_sanity_correction)
_RUN_DEMOTE_CADENCE_SPM_MAX = 130.0
_RUN_DEMOTE_STRONG_LOW_CADENCE_SPM_MAX = 95.0
_RUN_DEMOTE_PERIODICITY_MAX = 0.55
_RUN_DEMOTE_MODERATE_PERIODICITY_MAX = 0.75
_STAIRS_DEMOTE_CADENCE_SPM_MAX = 85.0
_STAIRS_DEMOTE_PERIODICITY_MAX = 0.45
_TRANSITIONS_DEMOTE_CADENCE_SPM_MIN = 120.0
_TRANSITIONS_DEMOTE_RAW_CONFIDENCE_MIN = 0.40
_TRANSITIONS_DEMOTE_PERIODICITY_MIN = 0.25


def apply_low_motion_gate(
    feature_row: np.ndarray,
    emission: np.ndarray,
    label_order: list[str],
    placement_labels: list[str],
    placement_label: str,
) -> tuple[np.ndarray, bool]:
    """Zero locomotion probabilities when body_mag_energy is below threshold."""
    feature_names = window_feature_names(placement_labels)
    try:
        energy_idx = feature_names.index("body_mag_energy")
    except ValueError:
        return emission, False

    if float(feature_row[energy_idx]) >= LOW_MOTION_ENERGY_THRESHOLD:
        return emission, False

    locomotion_indices = [
        idx for idx, label in enumerate(label_order) if label in _MOTION_REQUIRING_LABELS
    ]
    if not locomotion_indices:
        return emission, False

    gated = emission.copy()
    gated[locomotion_indices] = 0.0

    total = float(np.sum(gated))
    if total <= 1e-12:
        static_indices = [
            idx for idx, label in enumerate(label_order) if label in STATIC_LABELS
        ]
        if not static_indices:
            return emission, False
        gated = np.zeros_like(emission)
        uniform = 1.0 / len(static_indices)
        for class_idx in static_indices:
            gated[class_idx] = uniform
        return gated, True

    gated /= total
    return gated, True


def apply_locomotion_transitions_guard(
    feature_row: np.ndarray,
    emission: np.ndarray,
    label_order: list[str],
    placement_labels: list[str],
) -> tuple[np.ndarray, bool, str]:
    """Demote high-energy transitions to walk/run when body energy rules out postural transitions.

    Genuine postural transitions (sit→stand, stand→sit) have body_mag_energy well below the
    low-motion threshold (<< 0.1 m/s²). Running with directional changes produces energy of
    10–25 m/s² but looks aperiodic like transitions to the model. When decoded label is
    'transitions' and body_mag_energy is high, the window is locomotion in disguise.

    Demotes 80% of transitions mass to:
    - 'run'  if body_mag_energy ≥ _LOCOMOTION_TRANSITIONS_RUN_ENERGY_MIN (≥12 m/s²)
    - 'walk' if body_mag_energy ≥ _LOCOMOTION_TRANSITIONS_ENERGY_MIN but < 12 m/s²

    Returns
    -------
    (emission, applied, reason_string)
    """
    transitions_idx = next((idx for idx, label in enumerate(label_order) if label == "transitions"), None)
    walk_idx = next((idx for idx, label in enumerate(label_order) if label == "walk"), None)
    run_idx = next((idx for idx, label in enumerate(label_order) if label == "run"), None)
    if transitions_idx is None or walk_idx is None or run_idx is None:
        return emission, False, "none"

    pred_idx = int(np.argmax(emission))
    if label_order[pred_idx] != "transitions":
        return emission, False, "none"

    feature_names = window_feature_names(placement_labels)
    try:
        energy_idx = feature_names.index("body_mag_energy")
    except ValueError:
        return emission, False, "none"

    energy = float(feature_row[energy_idx])
    if energy < _LOCOMOTION_TRANSITIONS_ENERGY_MIN:
        return emission, False, "none"

    target_idx = run_idx if energy >= _LOCOMOTION_TRANSITIONS_RUN_ENERGY_MIN else walk_idx
    target_label = label_order[target_idx]
    reason = f"transitions-to-{target_label}-high-energy"

    out = emission.copy()
    move_mass = min(float(out[transitions_idx]) * 0.80, 0.45)
    if move_mass <= 0.0:
        return emission, False, "none"

    out[transitions_idx] -= move_mass
    out[target_idx] += move_mass
    total = float(np.sum(out))
    if total > 0.0:
        out /= total
    return out, True, reason


def should_reset_on_low_motion(
    previous_state: dict[str, Any] | None,
    *,
    label_order: list[str],
) -> bool:
    """Reset decoder on low-motion only if previous state was locomotion-heavy."""
    if not isinstance(previous_state, dict):
        return True
    posterior = extract_decoder_posterior(previous_state, num_classes=len(label_order))
    if posterior is None:
        return True
    top_idx = int(np.argmax(posterior))
    top_label = label_order[top_idx]
    top_prob = float(posterior[top_idx])
    if top_label in LOCOMOTION_LABELS and top_prob >= 0.45:
        return True
    return False


def apply_stairs_static_guard(
    feature_row: np.ndarray,
    emission: np.ndarray,
    label_order: list[str],
    placement_labels: list[str],
) -> tuple[np.ndarray, bool, dict[str, float]]:
    """Suppress stairs probability in static-like windows."""
    info: dict[str, float] = {
        "stairs_guard_energy": 0.0,
        "stairs_guard_jerk": 0.0,
        "stairs_guard_dom_freq": 0.0,
    }
    stairs_idx = next((idx for idx, label in enumerate(label_order) if label == "stairs"), None)
    if stairs_idx is None:
        return emission, False, info

    feature_names = window_feature_names(placement_labels)
    try:
        energy_idx = feature_names.index("body_mag_energy")
        jerk_idx = feature_names.index("jerk_mag_rms")
        dom_idx = feature_names.index("body_mag_dom_freq_hz")
    except ValueError:
        return emission, False, info

    energy = float(feature_row[energy_idx])
    jerk = float(feature_row[jerk_idx])
    dom = float(feature_row[dom_idx])
    info["stairs_guard_energy"] = energy
    info["stairs_guard_jerk"] = jerk
    info["stairs_guard_dom_freq"] = dom

    is_static_like = (
        energy <= _STAIRS_GUARD_ENERGY_MAX
        and jerk <= _STAIRS_GUARD_JERK_MAX
        and dom <= _STAIRS_GUARD_DOM_FREQ_MAX
    )
    if not is_static_like:
        return emission, False, info

    out = emission.copy()
    current_stairs = float(out[stairs_idx])
    if current_stairs <= _STAIRS_GUARD_MAX_PROBA:
        return out, False, info

    out[stairs_idx] = _STAIRS_GUARD_MAX_PROBA
    remaining_before = max(1e-12, float(np.sum(emission)) - current_stairs)
    remaining_after = max(0.0, 1.0 - _STAIRS_GUARD_MAX_PROBA)
    if remaining_before > 0.0:
        scale = remaining_after / remaining_before
        for idx in range(out.size):
            if idx != stairs_idx:
                out[idx] = float(emission[idx]) * scale

    total = float(np.sum(out))
    if total > 0.0:
        out /= total
    return out, True, info


def apply_stairs_moving_window_guard(
    feature_row: np.ndarray,
    emission: np.ndarray,
    label_order: list[str],
    placement_labels: list[str],
    placement_label: str,
    previous_state: dict[str, Any] | None,
) -> tuple[np.ndarray, bool, dict[str, float]]:
    """Suppress sticky stairs in front-pocket moving windows."""
    info: dict[str, float] = {
        "stairs_moving_guard_energy": 0.0,
        "stairs_moving_guard_dom_freq": 0.0,
        "stairs_moving_guard_prev_stairs": 0.0,
        "stairs_moving_guard_smoothed_stairs": 0.0,
        "stairs_moving_guard_walkrun_mass": 0.0,
    }
    if placement_label != FRONT_POCKET_LABEL:
        return emission, False, info

    stairs_idx = next((idx for idx, label in enumerate(label_order) if label == "stairs"), None)
    walk_idx = next((idx for idx, label in enumerate(label_order) if label == "walk"), None)
    run_idx = next((idx for idx, label in enumerate(label_order) if label == "run"), None)
    if stairs_idx is None or walk_idx is None or run_idx is None:
        return emission, False, info

    feature_names = window_feature_names(placement_labels)
    try:
        energy_idx = feature_names.index("body_mag_energy")
        dom_idx = feature_names.index("body_mag_dom_freq_hz")
    except ValueError:
        return emission, False, info

    energy = float(feature_row[energy_idx])
    dom = float(feature_row[dom_idx])
    info["stairs_moving_guard_energy"] = energy
    info["stairs_moving_guard_dom_freq"] = dom

    is_moving_like = (
        energy >= _STAIRS_MOVING_GUARD_MIN_ENERGY
        and _STAIRS_MOVING_GUARD_MIN_DOM_FREQ <= dom <= _STAIRS_MOVING_GUARD_MAX_DOM_FREQ
    )
    if not is_moving_like:
        return emission, False, info

    current_stairs = float(emission[stairs_idx])
    previous_posterior = extract_decoder_posterior(previous_state, num_classes=emission.size)
    prev_stairs = current_stairs if previous_posterior is None else float(previous_posterior[stairs_idx])
    smoothed_stairs = (
        (1.0 - _STAIRS_MOVING_GUARD_PREV_BLEND) * current_stairs
        + _STAIRS_MOVING_GUARD_PREV_BLEND * prev_stairs
    )
    walkrun_mass = float(emission[walk_idx] + emission[run_idx])

    info["stairs_moving_guard_prev_stairs"] = prev_stairs
    info["stairs_moving_guard_smoothed_stairs"] = smoothed_stairs
    info["stairs_moving_guard_walkrun_mass"] = walkrun_mass

    if smoothed_stairs < _STAIRS_MOVING_GUARD_MIN_SMOOTHED_STAIRS:
        return emission, False, info
    if walkrun_mass < _STAIRS_MOVING_GUARD_MIN_WALKRUN_MASS:
        return emission, False, info
    if current_stairs <= _STAIRS_MOVING_GUARD_MAX_PROBA:
        return emission, False, info

    out = emission.copy()
    stairs_before = float(out[stairs_idx])
    out[stairs_idx] = _STAIRS_MOVING_GUARD_MAX_PROBA
    delta = stairs_before - _STAIRS_MOVING_GUARD_MAX_PROBA

    if walkrun_mass <= 1e-12:
        out[walk_idx] += delta * 0.5
        out[run_idx] += delta * 0.5
    else:
        out[walk_idx] += delta * float(emission[walk_idx] / walkrun_mass)
        out[run_idx] += delta * float(emission[run_idx] / walkrun_mass)

    total = float(np.sum(out))
    if total > 0.0:
        out /= total
    return out, True, info


def apply_overconfident_stairs_failsafe(
    emission: np.ndarray,
    decoded_proba: np.ndarray,
    label_order: list[str],
    placement_label: str,
) -> tuple[np.ndarray, bool, dict[str, float]]:
    """Cap implausibly overconfident decoded stairs in front-pocket windows."""
    info: dict[str, float] = {
        "stairs_failsafe_raw_stairs": 0.0,
        "stairs_failsafe_raw_walkrun": 0.0,
        "stairs_failsafe_decoded_stairs": 0.0,
        "stairs_failsafe_decoded_walkrun": 0.0,
    }
    if placement_label != FRONT_POCKET_LABEL:
        return decoded_proba, False, info

    stairs_idx = next((idx for idx, label in enumerate(label_order) if label == "stairs"), None)
    walk_idx = next((idx for idx, label in enumerate(label_order) if label == "walk"), None)
    run_idx = next((idx for idx, label in enumerate(label_order) if label == "run"), None)
    if stairs_idx is None or walk_idx is None or run_idx is None:
        return decoded_proba, False, info

    raw_stairs = float(emission[stairs_idx])
    raw_walkrun = float(emission[walk_idx] + emission[run_idx])
    decoded_stairs = float(decoded_proba[stairs_idx])
    decoded_walkrun = float(decoded_proba[walk_idx] + decoded_proba[run_idx])

    info["stairs_failsafe_raw_stairs"] = raw_stairs
    info["stairs_failsafe_raw_walkrun"] = raw_walkrun
    info["stairs_failsafe_decoded_stairs"] = decoded_stairs
    info["stairs_failsafe_decoded_walkrun"] = decoded_walkrun

    is_decoded_stairs = int(np.argmax(decoded_proba)) == stairs_idx
    should_apply = (
        is_decoded_stairs
        and decoded_stairs >= _STAIRS_FAILSAFE_MIN_DECODED_STAIRS
        and raw_walkrun >= _STAIRS_FAILSAFE_MIN_RAW_WALKRUN
        and raw_stairs <= _STAIRS_FAILSAFE_MAX_RAW_STAIRS
    )
    if not should_apply:
        return decoded_proba, False, info

    out = decoded_proba.copy()
    delta = float(out[stairs_idx] - _STAIRS_FAILSAFE_MAX_DECODED_STAIRS)
    if delta <= 0.0:
        return out, False, info

    out[stairs_idx] = _STAIRS_FAILSAFE_MAX_DECODED_STAIRS
    raw_walk = float(emission[walk_idx])
    raw_run = float(emission[run_idx])
    raw_sum = raw_walk + raw_run
    if raw_sum > 1e-12:
        out[walk_idx] += delta * (raw_walk / raw_sum)
        out[run_idx] += delta * (raw_run / raw_sum)
    else:
        out[walk_idx] += delta * 0.5
        out[run_idx] += delta * 0.5

    total = float(np.sum(out))
    if total > 0.0:
        out /= total
    return out, True, info


def apply_front_pocket_smooth_walk_guard(
    feature_row: np.ndarray,
    emission: np.ndarray,
    label_order: list[str],
    placement_labels: list[str],
    placement_label: str,
) -> tuple[np.ndarray, bool, dict[str, float]]:
    """Bias toward walk for smooth front-pocket gait windows."""
    info: dict[str, float] = {
        "walk_guard_energy": 0.0,
        "walk_guard_dom_freq": 0.0,
        "walk_guard_jerk": 0.0,
        "walk_guard_walk": 0.0,
        "walk_guard_run": 0.0,
        "walk_guard_stairs": 0.0,
    }
    if placement_label != FRONT_POCKET_LABEL:
        return emission, False, info

    walk_idx = next((idx for idx, label in enumerate(label_order) if label == "walk"), None)
    run_idx = next((idx for idx, label in enumerate(label_order) if label == "run"), None)
    stairs_idx = next((idx for idx, label in enumerate(label_order) if label == "stairs"), None)
    if walk_idx is None or run_idx is None or stairs_idx is None:
        return emission, False, info

    feature_names = window_feature_names(placement_labels)
    try:
        energy_idx = feature_names.index("body_mag_energy")
        dom_idx = feature_names.index("body_mag_dom_freq_hz")
        jerk_idx = feature_names.index("jerk_mag_rms")
    except ValueError:
        return emission, False, info

    energy = float(feature_row[energy_idx])
    dom = float(feature_row[dom_idx])
    jerk = float(feature_row[jerk_idx])
    walk_prob = float(emission[walk_idx])
    run_prob = float(emission[run_idx])
    stairs_prob = float(emission[stairs_idx])

    info["walk_guard_energy"] = energy
    info["walk_guard_dom_freq"] = dom
    info["walk_guard_jerk"] = jerk
    info["walk_guard_walk"] = walk_prob
    info["walk_guard_run"] = run_prob
    info["walk_guard_stairs"] = stairs_prob

    is_smooth_walk_like = (
        _WALK_GUARD_MIN_ENERGY <= energy <= _WALK_GUARD_MAX_ENERGY
        and _WALK_GUARD_MIN_DOM_FREQ <= dom <= _WALK_GUARD_MAX_DOM_FREQ
        and jerk <= _WALK_GUARD_MAX_JERK
    )
    if not is_smooth_walk_like:
        return emission, False, info

    if walk_prob < _WALK_GUARD_MIN_WALK:
        return emission, False, info
    if run_prob > _WALK_GUARD_MAX_RUN:
        return emission, False, info
    if stairs_prob > _WALK_GUARD_MAX_STAIRS:
        return emission, False, info

    out = emission.copy()
    run_before = float(out[run_idx])
    stairs_before = float(out[stairs_idx])
    run_after = min(run_before, _WALK_GUARD_CAP_RUN)
    stairs_after = min(stairs_before, _WALK_GUARD_CAP_STAIRS)
    reclaimed = (run_before - run_after) + (stairs_before - stairs_after)
    if reclaimed <= 0.0:
        return emission, False, info

    out[run_idx] = run_after
    out[stairs_idx] = stairs_after
    out[walk_idx] += reclaimed

    total = float(np.sum(out))
    if total > 0.0:
        out /= total
    return out, True, info


def apply_stairs_hallucination_guard(
    feature_row: np.ndarray,
    emission: np.ndarray,
    label_order: list[str],
    placement_labels: list[str],
    placement_label: str,
) -> tuple[np.ndarray, bool, dict[str, float]]:
    """Reject stairs when motion signature is too erratic (excessive jitter)."""
    info: dict[str, float] = {
        "hallucination_guard_energy": 0.0,
        "hallucination_guard_jerk": 0.0,
        "hallucination_guard_stairs": 0.0,
    }
    if placement_label != FRONT_POCKET_LABEL:
        return emission, False, info

    stairs_idx = next((idx for idx, label in enumerate(label_order) if label == "stairs"), None)
    if stairs_idx is None:
        return emission, False, info

    feature_names = window_feature_names(placement_labels)
    try:
        energy_idx = feature_names.index("body_mag_energy")
        jerk_idx = feature_names.index("jerk_mag_rms")
    except ValueError:
        return emission, False, info

    energy = float(feature_row[energy_idx])
    jerk = float(feature_row[jerk_idx])
    stairs_prob = float(emission[stairs_idx])

    info["hallucination_guard_energy"] = energy
    info["hallucination_guard_jerk"] = jerk
    info["hallucination_guard_stairs"] = stairs_prob

    is_too_erratic = energy > _STAIRS_HALLUCINATION_MAX_ENERGY and jerk > _STAIRS_HALLUCINATION_MAX_JERK
    if not is_too_erratic:
        return emission, False, info

    if stairs_prob <= _STAIRS_HALLUCINATION_MAX_STAIRS_PROBA:
        return emission, False, info

    out = emission.copy()
    stairs_before = float(out[stairs_idx])
    out[stairs_idx] = _STAIRS_HALLUCINATION_MAX_STAIRS_PROBA
    delta = stairs_before - _STAIRS_HALLUCINATION_MAX_STAIRS_PROBA

    walk_idx = next((idx for idx, label in enumerate(label_order) if label == "walk"), None)
    run_idx = next((idx for idx, label in enumerate(label_order) if label == "run"), None)
    if walk_idx is not None and run_idx is not None:
        walk_mass = float(emission[walk_idx] + emission[run_idx])
        if walk_mass > 1e-12:
            out[walk_idx] += delta * float(emission[walk_idx] / walk_mass)
            out[run_idx] += delta * float(emission[run_idx] / walk_mass)
        else:
            out[walk_idx] += delta * 0.5
            out[run_idx] += delta * 0.5
    else:
        if walk_idx is not None:
            out[walk_idx] += delta

    total = float(np.sum(out))
    if total > 0.0:
        out /= total
    return out, True, info


def apply_pocket_activity_sanity_correction(
    decoded_proba: np.ndarray,
    *,
    label_order: list[str],
    placement_label: str,
    cadence_spm: float,
    periodicity_strength: float,
    artifact_score: float,
    raw_label: str | None = None,
    raw_confidence: float = 0.0,
) -> tuple[np.ndarray, bool, str]:
    """Demote run/stairs to walk for loose-pocket-like windows with weak rhythmic support."""
    if placement_label != FRONT_POCKET_LABEL:
        return decoded_proba, False, "none"
    if artifact_score <= POCKET_ARTIFACT_SCORE_THRESHOLD:
        return decoded_proba, False, "none"

    walk_idx = next((idx for idx, label in enumerate(label_order) if label == "walk"), None)
    run_idx = next((idx for idx, label in enumerate(label_order) if label == "run"), None)
    stairs_idx = next((idx for idx, label in enumerate(label_order) if label == "stairs"), None)
    transitions_idx = next((idx for idx, label in enumerate(label_order) if label == "transitions"), None)
    if walk_idx is None or run_idx is None or stairs_idx is None:
        return decoded_proba, False, "none"

    pred_idx = int(np.argmax(decoded_proba))
    pred_label = label_order[pred_idx]
    should_downgrade = False
    reason = "none"

    if pred_label == "run":
        if cadence_spm < _RUN_DEMOTE_STRONG_LOW_CADENCE_SPM_MAX:
            should_downgrade = True
            reason = "run-to-walk-low-cadence-artifact"
        elif (
            cadence_spm < _RUN_DEMOTE_CADENCE_SPM_MAX
            and periodicity_strength < _RUN_DEMOTE_MODERATE_PERIODICITY_MAX
        ):
            should_downgrade = True
            reason = "run-to-walk-low-cadence-artifact"
    elif pred_label == "stairs":
        if (
            cadence_spm > 0.0 and cadence_spm < _STAIRS_DEMOTE_CADENCE_SPM_MAX
        ) or periodicity_strength < _STAIRS_DEMOTE_PERIODICITY_MAX:
            should_downgrade = True
            reason = "stairs-to-walk-low-periodicity-artifact"
    elif pred_label == "transitions" and transitions_idx is not None:
        if (
            raw_label in {"walk", "run", "stairs"}
            and raw_confidence >= _TRANSITIONS_DEMOTE_RAW_CONFIDENCE_MIN
            and cadence_spm >= _TRANSITIONS_DEMOTE_CADENCE_SPM_MIN
            and periodicity_strength >= _TRANSITIONS_DEMOTE_PERIODICITY_MIN
        ):
            should_downgrade = True
            reason = f"transitions-to-{raw_label}-locomotion-burst"

    if not should_downgrade:
        return decoded_proba, False, "none"

    out = decoded_proba.copy()
    move_mass = min(float(out[pred_idx]) * 0.80, 0.35)
    if move_mass <= 0.0:
        return decoded_proba, False, "none"

    out[pred_idx] -= move_mass

    if pred_label == "transitions" and raw_label in {"walk", "run", "stairs"}:
        target_idx = {"walk": walk_idx, "run": run_idx, "stairs": stairs_idx}[raw_label]
        out[target_idx] += move_mass
    else:
        out[walk_idx] += move_mass

    total = float(np.sum(out))
    if total > 0.0:
        out /= total
    return out, True, reason
