"""Train pocket-only XGBoost model -- identical logic to motionlens_xgboost_baseline.py
but with two differences:
  1. Data is filtered to smartphone pocket / lateral-waist placements only.
  2. Placement one-hot features are NOT included (placement_labels=[]).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import logging
import math
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from core.features import (
    compute_window_features as core_compute_window_features,
    window_feature_names as core_window_feature_names,
)
from core.inference import (
    apply_static_specialist_refinement as core_apply_static_specialist_refinement,
    causal_hmm_decode as core_causal_hmm_decode,
    predict_hierarchical_proba as core_predict_hierarchical_proba,
)
from processing.inference import (
    decode_predictions as core_decode_predictions,
)
from processing.training_transition_stats import (
    estimate_transition_stats as training_estimate_transition_stats,
)


TARGET_SAMPLE_RATE_HZ = 50.0
WINDOW_SAMPLES = 128

DEFAULT_TARGET_LABELS = (
    "walk",
    "run",
    "stairs",
    "sit/lay",
    "stand",
    "transitions",
    "locomotion-other",
)

# Raw contract labels "sit" and "lay" are merged into the joint class "sit/lay".
LABEL_REMAP: dict[str, str] = {"sit": "sit/lay", "lay": "sit/lay"}

# No placement one-hot features for pocket-only model
PLACEMENT_LABELS: tuple[str, ...] = ()

# Pocket-only data filter -- only these datasets and placements are used for training
POCKET_DATASETS = frozenset({
    "wisdm",
    "motion_sense",
    "uma_fall",
    "unimib_shar",
    "wisdm_v2",
    "real_world",
    "shoaib_2013",
    "shoaib_sensors",
    "ut_complex",
    "iphone_sweep",
})
POCKET_PLACEMENTS = frozenset({
    "front_pocket",
    "lateral_left_lower",
    "lateral_right_lower",
})

AXES = ("x", "y", "z")

COARSE_LABEL_ORDER = ("locomotion", "static", "transition", "other")
STATIC_SPECIALIST_LABELS = ("sit/lay", "stand")
FINE_TO_COARSE = {
    "walk": "locomotion",
    "run": "locomotion",
    "stairs": "locomotion",
    "sit/lay": "static",
    "stand": "static",
    "transitions": "transition",
    "locomotion-other": "other",
}

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a pocket-only XGBoost model (no placement features)."
    )
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=Path("output") / "motionlens_contract",
        help="Path to the contract output directory with samples/ and manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "model_pocket_only_v1",
        help="Directory for model and evaluation artifacts.",
    )
    parser.add_argument(
        "--target-labels",
        nargs="*",
        default=list(DEFAULT_TARGET_LABELS),
        help="Canonical labels to include as supervised targets.",
    )
    parser.add_argument(
        "--exclude-train-only",
        action="store_true",
        help="Exclude split=train_only windows from training.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap per split partition on processed parquet files (0 means all).",
    )
    parser.add_argument(
        "--tune-small-search",
        action="store_true",
        help="Run a small random hyperparameter search on a train/validation split.",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=12,
        help="Number of sampled trials when --tune-small-search is enabled.",
    )
    parser.add_argument(
        "--tune-metric",
        choices=["accuracy", "macro-f1"],
        default="macro-f1",
        help="Validation metric to optimize during --tune-small-search.",
    )
    parser.add_argument(
        "--tune-search-mode",
        choices=["random", "bayesian"],
        default="random",
        help="Hyperparameter search algorithm used when tuning is enabled.",
    )
    parser.add_argument(
        "--tune-search-space",
        choices=["small", "wide"],
        default="small",
        help="Hyperparameter search space used when tuning is enabled.",
    )
    parser.add_argument(
        "--tune-split-mode",
        choices=["stratified-windows", "grouped-subject", "grouped-dataset"],
        default="grouped-subject",
        help="Validation split strategy used during tuning.",
    )
    parser.add_argument(
        "--enforce-causal-streaming",
        action="store_true",
        help="Enforce streaming-safe decoding settings (no lookahead decoders).",
    )
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=Path("output") / "model_experiments" / "experiment_history.csv",
        help="CSV file that accumulates run configs and metrics for experiment comparison.",
    )
    parser.add_argument(
        "--use-hierarchical",
        action="store_true",
        help="Train coarse-to-fine hierarchical classifiers before temporal decoding.",
    )
    parser.add_argument(
        "--temporal-decoder",
        choices=["none", "causal-hmm"],
        default="causal-hmm",
        help="Temporal decoding method applied to test predictions.",
    )
    parser.add_argument(
        "--decoder-sticky-prior",
        type=float,
        default=0.25,
        help="Blend factor for duration-informed sticky transition priors (0 disables).",
    )
    parser.add_argument(
        "--decoder-cross-coarse-penalty",
        type=float,
        default=0.15,
        help="Penalty strength for transitions across coarse groups (except transition class).",
    )
    parser.add_argument(
        "--disable-static-specialist",
        action="store_true",
        help="Disable static-posture specialist refinement for still/lay.",
    )
    parser.add_argument(
        "--static-specialist-threshold",
        type=float,
        default=0.35,
        help="Minimum static gate probability required to apply specialist blending.",
    )
    parser.add_argument(
        "--static-specialist-blend",
        type=float,
        default=0.6,
        help="Blend factor between base and specialist probabilities in static windows.",
    )
    parser.add_argument(
        "--disable-hard-negative-mining",
        action="store_true",
        help="Disable the one-pass hard-negative mining refinement during model fitting.",
    )
    parser.add_argument(
        "--hard-negative-factor",
        type=float,
        default=1.75,
        help="Weight multiplier for misclassified training windows during hard-negative refinement.",
    )
    parser.add_argument(
        "--dataset-balance-mode",
        choices=["none", "inverse", "sqrt-inverse"],
        default="inverse",
        help="Optional per-dataset balancing factor applied to training sample weights.",
    )
    parser.add_argument(
        "--dataset-balance-strength",
        type=float,
        default=1.0,
        help="Blend strength for dataset balancing in [0,1]; 1 applies the full factor.",
    )
    parser.add_argument(
        "--xgb-n-estimators",
        type=int,
        default=None,
        help="Override XGBoost n_estimators without running tuning.",
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=None,
        help="Override XGBoost max_depth without running tuning.",
    )
    parser.add_argument(
        "--xgb-min-child-weight",
        type=int,
        default=None,
        help="Override XGBoost min_child_weight without running tuning.",
    )
    parser.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=None,
        help="Override XGBoost learning_rate without running tuning.",
    )
    parser.add_argument(
        "--xgb-subsample",
        type=float,
        default=None,
        help="Override XGBoost subsample without running tuning.",
    )
    parser.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=None,
        help="Override XGBoost colsample_bytree without running tuning.",
    )
    parser.add_argument(
        "--xgb-gamma",
        type=float,
        default=None,
        help="Override XGBoost gamma without running tuning.",
    )
    parser.add_argument(
        "--xgb-reg-alpha",
        type=float,
        default=None,
        help="Override XGBoost reg_alpha without running tuning.",
    )
    parser.add_argument(
        "--xgb-reg-lambda",
        type=float,
        default=None,
        help="Override XGBoost reg_lambda without running tuning.",
    )
    return parser.parse_args()


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0))


def _safe_skew(values: np.ndarray) -> float:
    std = _safe_std(values)
    if std < 1e-8:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean((centered / std) ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    std = _safe_std(values)
    if std < 1e-8:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean((centered / std) ** 4) - 3.0)


def _zero_crossing_rate(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.mean(np.diff(np.signbit(values)) != 0))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return 0.0
    if _safe_std(a) < 1e-8 or _safe_std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _band_energy(power: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def _spectral_entropy(power: np.ndarray) -> float:
    total_power = float(np.sum(power))
    if total_power <= 0.0:
        return 0.0
    probs = power / total_power
    probs = probs[probs > 0.0]
    if probs.size <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(entropy / math.log(probs.size))


def _dominant_frequency(power: np.ndarray, freqs: np.ndarray) -> float:
    if power.size == 0:
        return 0.0
    valid = freqs > 0.0
    if not np.any(valid):
        return 0.0
    idx = np.argmax(power[valid])
    return float(freqs[valid][idx])


def _gravity_angle_stability(gravity_x: np.ndarray, gravity_y: np.ndarray, gravity_z: np.ndarray) -> float:
    gravity_vectors = np.column_stack((gravity_x, gravity_y, gravity_z))
    mean_vec = np.mean(gravity_vectors, axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    norms = np.linalg.norm(gravity_vectors, axis=1)
    valid = (norms > 1e-8) & (mean_norm > 1e-8)
    if not np.any(valid):
        return 0.0

    dots = np.sum(gravity_vectors[valid] * mean_vec, axis=1)
    cos_angles = np.clip(dots / (norms[valid] * mean_norm), -1.0, 1.0)
    angles = np.arccos(cos_angles)
    return float(np.std(angles, ddof=0))


def _spectral_centroid(power: np.ndarray, freqs: np.ndarray) -> float:
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    return float(np.sum(freqs * power) / total)


def _stft_band_energy_stats(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    nperseg: int = 32,
    noverlap: int = 16,
) -> tuple[float, float]:
    """Mean and temporal std of per-frame STFT band energy (time-frequency feature)."""
    _, _, Zxx = _compute_stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    n_freq_bins = Zxx.shape[0]
    stft_freqs = np.linspace(0.0, fs / 2.0, n_freq_bins)
    power = np.abs(Zxx) ** 2  # (freq_bins, time_frames)
    mask = (stft_freqs >= low_hz) & (stft_freqs <= high_hz)
    if not np.any(mask) or power.shape[1] == 0:
        return 0.0, 0.0
    band_energy_over_time = np.sum(power[mask, :], axis=0)
    return float(np.mean(band_energy_over_time)), float(np.std(band_energy_over_time, ddof=0))


def _stft_dom_freq_std(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 32,
    noverlap: int = 16,
) -> float:
    """Std of dominant frequency across STFT frames (temporal frequency stability)."""
    _, _, Zxx = _compute_stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    n_freq_bins = Zxx.shape[0]
    stft_freqs = np.linspace(0.0, fs / 2.0, n_freq_bins)
    power = np.abs(Zxx) ** 2
    if power.shape[1] == 0:
        return 0.0
    valid = stft_freqs > 0.0
    if not np.any(valid):
        return 0.0
    dom_freq_per_frame = stft_freqs[valid][np.argmax(power[valid, :], axis=0)]
    return float(np.std(dom_freq_per_frame, ddof=0))


def _dwt_subband_energies(signal: np.ndarray, wavelet: str = "db4", level: int = 4) -> list[float]:
    """Normalised DWT subband energies [D1, D2, D3, D4, A4] + high/low ratio.

    At 50 Hz with level=4:  D1≈12.5-25 Hz, D2≈6.25-12.5 Hz, D3≈3.1-6.25 Hz,
    D4≈1.56-3.1 Hz, A4≈0-1.56 Hz.
    """
    coeffs = _pywt.wavedec(signal, wavelet, level=level)
    # coeffs[0]=A4, coeffs[1]=D4, ..., coeffs[4]=D1
    approx_energy = float(np.sum(coeffs[0] ** 2))
    detail_energies = [float(np.sum(c ** 2)) for c in reversed(coeffs[1:])]  # D1..D4
    total = sum(detail_energies) + approx_energy
    if total <= 0.0:
        return [0.0] * 6
    normed = [e / total for e in detail_energies]  # [D1, D2, D3, D4]
    normed_approx = approx_energy / total  # A4
    high = normed[0] + normed[1] + normed[2]  # D1+D2+D3
    low = normed[3] + normed_approx  # D4+A4
    ratio = high / (low + 1e-8)
    return normed + [normed_approx, ratio]  # 6 values


def _window_feature_names() -> list[str]:
    # Backward-compatible wrapper for tests; canonical schema lives in core.features.
    return core_window_feature_names(PLACEMENT_LABELS)


def compute_window_features(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    placement_label: str,
) -> np.ndarray:
    # Backward-compatible wrapper for tests; canonical computation lives in core.features.
    return core_compute_window_features(
        acc_x=acc_x,
        acc_y=acc_y,
        acc_z=acc_z,
        placement_label=placement_label,
        placement_labels=PLACEMENT_LABELS,
        target_sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
    )


def _iter_window_records(parquet_path: Path):
    stream_id = "unknown"
    for part in parquet_path.parts:
        if part.startswith("stream_id="):
            stream_id = part.split("=", 1)[1]
            break

    pf = pq.ParquetFile(parquet_path)
    columns = [
        "global_window_id",
        "sample_index",
        "acc_x",
        "acc_y",
        "acc_z",
        "activity_label",
        "placement_label",
        "dataset_id",
        "split",
        "global_subject_id",
    ]

    current_window_id = None
    x_buf: list[float] = []
    y_buf: list[float] = []
    z_buf: list[float] = []
    meta: dict[str, str] = {}

    def flush_current():
        nonlocal current_window_id, x_buf, y_buf, z_buf, meta
        if current_window_id is None:
            return None
        if len(x_buf) != WINDOW_SAMPLES:
            current_window_id = None
            x_buf, y_buf, z_buf = [], [], []
            meta = {}
            return None

        record = {
            "global_window_id": current_window_id,
            "acc_x": np.asarray(x_buf, dtype=np.float64),
            "acc_y": np.asarray(y_buf, dtype=np.float64),
            "acc_z": np.asarray(z_buf, dtype=np.float64),
            **meta,
        }
        current_window_id = None
        x_buf, y_buf, z_buf = [], [], []
        meta = {}
        return record

    for batch in pf.iter_batches(columns=columns, batch_size=8192):
        window_ids = batch.column(0).to_pylist()
        sample_indices = np.asarray(batch.column(1).to_numpy(zero_copy_only=False), dtype=np.int32)
        acc_x = np.asarray(batch.column(2).to_numpy(zero_copy_only=False), dtype=np.float64)
        acc_y = np.asarray(batch.column(3).to_numpy(zero_copy_only=False), dtype=np.float64)
        acc_z = np.asarray(batch.column(4).to_numpy(zero_copy_only=False), dtype=np.float64)
        activity_labels = batch.column(5).to_pylist()
        placement_labels = batch.column(6).to_pylist()
        dataset_ids = batch.column(7).to_pylist()
        splits = batch.column(8).to_pylist()
        subject_ids = batch.column(9).to_pylist()

        for idx in range(len(window_ids)):
            window_id = window_ids[idx]
            if current_window_id is None:
                current_window_id = window_id
                meta = {
                    "activity_label": activity_labels[idx],
                    "placement_label": placement_labels[idx],
                    "dataset_id": dataset_ids[idx],
                    "split": splits[idx],
                    "global_subject_id": subject_ids[idx],
                    "stream_id": stream_id,
                }
            elif window_id != current_window_id:
                record = flush_current()
                if record is not None:
                    yield record
                current_window_id = window_id
                meta = {
                    "activity_label": activity_labels[idx],
                    "placement_label": placement_labels[idx],
                    "dataset_id": dataset_ids[idx],
                    "split": splits[idx],
                    "global_subject_id": subject_ids[idx],
                    "stream_id": stream_id,
                }

            if 0 <= sample_indices[idx] < WINDOW_SAMPLES:
                x_buf.append(float(acc_x[idx]))
                y_buf.append(float(acc_y[idx]))
                z_buf.append(float(acc_z[idx]))

    record = flush_current()
    if record is not None:
        yield record


def collect_training_examples(
    contract_dir: Path,
    target_labels: set[str],
    include_train_only: bool,
    max_files: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    sample_root = contract_dir / "samples"
    if not sample_root.exists():
        raise ValueError(f"Missing sample parquet root: {sample_root}")

    all_files = sorted(sample_root.rglob("part-0.parquet"))
    selected_train_files = 0
    selected_test_files = 0
    selected_train_only_files = 0
    if max_files > 0:
        def has_split(path: Path, split_value: str) -> bool:
            return any(part == f"split={split_value}" for part in path.parts)

        train_files = [path for path in all_files if has_split(path, "train")]
        test_files = [path for path in all_files if has_split(path, "test")]
        train_only_files = [path for path in all_files if has_split(path, "train_only")]

        selected_train_files = min(len(train_files), max_files)
        selected_test_files = min(len(test_files), max_files)
        selected_train_only_files = min(len(train_only_files), max_files)

        selected = test_files[:max_files] + train_files[:max_files] + train_only_files[:max_files]
        all_files = sorted(dict.fromkeys(selected))
        LOGGER.info(
            "Selected %d parquet files with --max-files=%d (test=%d, train=%d, train_only=%d)",
            len(all_files),
            max_files,
            selected_test_files,
            selected_train_files,
            selected_train_only_files,
        )
    else:
        LOGGER.info("Scanning all %d parquet files under %s", len(all_files), sample_root)

    x_train: list[np.ndarray] = []
    y_train: list[str] = []
    dataset_train: list[str] = []
    x_test: list[np.ndarray] = []
    y_test: list[str] = []
    dataset_test: list[str] = []
    placement_train: list[str] = []
    placement_test: list[str] = []
    subject_train: list[str] = []
    subject_test: list[str] = []
    train_sequence_groups: list[str] = []
    test_sequence_groups: list[str] = []

    for parquet_file in all_files:
        for record in _iter_window_records(parquet_file):
            activity_label = str(record["activity_label"])
            # Merge sit and lay into joint class before all downstream processing
            activity_label = LABEL_REMAP.get(activity_label, activity_label)
            split = str(record["split"])
            dataset_id = str(record["dataset_id"])
            placement_label = str(record["placement_label"])

            # Pocket-only filter: skip records from wrong datasets or placements
            if dataset_id not in POCKET_DATASETS:
                continue
            if placement_label not in POCKET_PLACEMENTS:
                continue
            if activity_label not in target_labels:
                continue

            features = core_compute_window_features(
                acc_x=record["acc_x"],
                acc_y=record["acc_y"],
                acc_z=record["acc_z"],
                placement_label=str(record["placement_label"]),
                placement_labels=PLACEMENT_LABELS,
                target_sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
            )

            if split == "test":
                x_test.append(features)
                y_test.append(activity_label)
                dataset_test.append(str(record["dataset_id"]))
                placement_test.append(str(record["placement_label"]))
                test_sequence_groups.append(
                    f"{record['dataset_id']}|{record['global_subject_id']}|{record['stream_id']}"
                )
            elif split == "train" or (include_train_only and split == "train_only"):
                x_train.append(features)
                y_train.append(activity_label)
                dataset_train.append(str(record["dataset_id"]))
                placement_train.append(str(record["placement_label"]))
                subject_train.append(f"{record['dataset_id']}|{record['global_subject_id']}")
                train_sequence_groups.append(
                    f"{record['dataset_id']}|{record['global_subject_id']}|{record['stream_id']}"
                )
            if split == "test":
                subject_test.append(f"{record['dataset_id']}|{record['global_subject_id']}")

    if not x_train:
        raise ValueError("No training windows were collected. Check input directory and label filters.")
    if not x_test:
        raise ValueError("No test windows were collected. Check input directory and split filters.")

    LOGGER.info(
        "Collected %d training windows from %d datasets and %d test windows from %d datasets",
        len(x_train),
        len(set(dataset_train)),
        len(x_test),
        len(set(dataset_test)),
    )

    feature_names = core_window_feature_names(PLACEMENT_LABELS)
    return (
        np.vstack(x_train),
        np.asarray(y_train),
        np.vstack(x_test),
        np.asarray(y_test),
        dataset_train,
        dataset_test,
        feature_names,
        train_sequence_groups,
        test_sequence_groups,
        subject_train,
        subject_test,
        placement_train,
        placement_test,
    )


def _base_xgb_params(num_classes: int) -> dict[str, float | int | str]:
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "n_estimators": 400,
        "max_depth": 6,
        "min_child_weight": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        "random_state": 42,
    }


def _apply_xgb_param_overrides(
    params: dict[str, float | int | str],
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    """Apply optional CLI XGBoost overrides for train-only runs."""

    out = dict(params)
    overrides: dict[str, float | int | None] = {
        "n_estimators": args.xgb_n_estimators,
        "max_depth": args.xgb_max_depth,
        "min_child_weight": args.xgb_min_child_weight,
        "learning_rate": args.xgb_learning_rate,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample_bytree,
        "gamma": args.xgb_gamma,
        "reg_alpha": args.xgb_reg_alpha,
        "reg_lambda": args.xgb_reg_lambda,
    }
    for key, value in overrides.items():
        if value is not None:
            out[key] = value
    return out


def _sample_hyperparameter_candidates(trials: int, search_space: str) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(42)
    sampled: list[dict[str, float | int]] = []
    seen: set[tuple[float, ...]] = set()

    if search_space == "small":
        choices = {
            "max_depth": [4, 6, 8],
            "min_child_weight": [1, 3, 5],
            "gamma": [0.0, 0.25, 0.5, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "reg_lambda": [1.0, 2.0, 4.0, 8.0],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "decoder_sticky_prior": [0.0, 0.1, 0.2, 0.3],
            "decoder_cross_coarse_penalty": [0.0, 0.1, 0.2, 0.3],
            "static_specialist_threshold": [0.2, 0.35, 0.5],
            "static_specialist_blend": [0.3, 0.6, 0.8],
        }
        keys = sorted(choices)
        while len(sampled) < max(1, trials):
            candidate: dict[str, float | int] = {}
            for key in keys:
                candidate[key] = choices[key][int(rng.integers(0, len(choices[key])))]
            signature = tuple(float(candidate[key]) for key in keys)
            if signature in seen:
                continue
            seen.add(signature)
            sampled.append(candidate)
        return sampled

    if search_space != "wide":
        raise ValueError(f"Unsupported tune_search_space: {search_space}")

    while len(sampled) < max(1, trials):
        candidate = {
            "max_depth": int(rng.integers(3, 11)),
            "min_child_weight": int(rng.integers(1, 11)),
            "gamma": float(rng.uniform(0.0, 2.5)),
            "reg_alpha": float(rng.uniform(0.0, 2.0)),
            "reg_lambda": float(rng.uniform(0.5, 15.0)),
            "subsample": float(rng.uniform(0.65, 1.0)),
            "colsample_bytree": float(rng.uniform(0.55, 1.0)),
            "learning_rate": float(10 ** rng.uniform(np.log10(0.01), np.log10(0.2))),
            "n_estimators": int(rng.integers(250, 901)),
            "decoder_sticky_prior": float(rng.uniform(0.0, 0.35)),
            "decoder_cross_coarse_penalty": float(rng.uniform(0.0, 0.35)),
            "static_specialist_threshold": float(rng.uniform(0.15, 0.65)),
            "static_specialist_blend": float(rng.uniform(0.2, 0.9)),
        }
        signature = (
            float(candidate["max_depth"]),
            float(candidate["min_child_weight"]),
            round(float(candidate["gamma"]), 4),
            round(float(candidate["reg_alpha"]), 4),
            round(float(candidate["reg_lambda"]), 4),
            round(float(candidate["subsample"]), 4),
            round(float(candidate["colsample_bytree"]), 4),
            round(float(candidate["learning_rate"]), 5),
            float(candidate["n_estimators"]),
            round(float(candidate["decoder_sticky_prior"]), 4),
            round(float(candidate["decoder_cross_coarse_penalty"]), 4),
            round(float(candidate["static_specialist_threshold"]), 4),
            round(float(candidate["static_specialist_blend"]), 4),
        )
        if signature in seen:
            continue
        seen.add(signature)
        sampled.append(candidate)

    return sampled


def _suggest_optuna_candidate(trial, search_space: str) -> dict[str, float | int]:
    if search_space == "small":
        return {
            "max_depth": int(trial.suggest_categorical("max_depth", [4, 6, 8])),
            "min_child_weight": int(trial.suggest_categorical("min_child_weight", [1, 3, 5])),
            "gamma": float(trial.suggest_categorical("gamma", [0.0, 0.25, 0.5, 1.0])),
            "reg_alpha": float(trial.suggest_categorical("reg_alpha", [0.0, 0.1, 0.5, 1.0])),
            "reg_lambda": float(trial.suggest_categorical("reg_lambda", [1.0, 2.0, 4.0, 8.0])),
            "subsample": float(trial.suggest_categorical("subsample", [0.8, 0.9, 1.0])),
            "colsample_bytree": float(trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9])),
            "decoder_sticky_prior": float(
                trial.suggest_categorical("decoder_sticky_prior", [0.0, 0.1, 0.2, 0.3])
            ),
            "decoder_cross_coarse_penalty": float(
                trial.suggest_categorical("decoder_cross_coarse_penalty", [0.0, 0.1, 0.2, 0.3])
            ),
            "static_specialist_threshold": float(
                trial.suggest_categorical("static_specialist_threshold", [0.2, 0.35, 0.5])
            ),
            "static_specialist_blend": float(
                trial.suggest_categorical("static_specialist_blend", [0.3, 0.6, 0.8])
            ),
        }
    if search_space == "wide":
        return {
            "max_depth": int(trial.suggest_int("max_depth", 3, 10)),
            "min_child_weight": int(trial.suggest_int("min_child_weight", 1, 10)),
            "gamma": float(trial.suggest_float("gamma", 0.0, 2.5)),
            "reg_alpha": float(trial.suggest_float("reg_alpha", 0.0, 2.0)),
            "reg_lambda": float(trial.suggest_float("reg_lambda", 0.5, 15.0)),
            "subsample": float(trial.suggest_float("subsample", 0.65, 1.0)),
            "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.55, 1.0)),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.01, 0.2, log=True)),
            "n_estimators": int(trial.suggest_int("n_estimators", 250, 900, step=25)),
            "decoder_sticky_prior": float(trial.suggest_float("decoder_sticky_prior", 0.0, 0.35)),
            "decoder_cross_coarse_penalty": float(
                trial.suggest_float("decoder_cross_coarse_penalty", 0.0, 0.35)
            ),
            "static_specialist_threshold": float(trial.suggest_float("static_specialist_threshold", 0.15, 0.65)),
            "static_specialist_blend": float(trial.suggest_float("static_specialist_blend", 0.2, 0.9)),
        }
    raise ValueError(f"Unsupported tune_search_space: {search_space}")


def _compute_sample_weights(
    y_train: np.ndarray,
    label_order: list[str],
    dataset_ids: list[str] | None = None,
    dataset_balance_mode: str = "none",
    dataset_balance_strength: float = 1.0,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    counts = np.bincount(y_train, minlength=len(label_order)).astype(np.float64)
    total = float(np.sum(counts))
    base = total / np.maximum(1.0, counts * len(label_order))

    class_weight: dict[str, float] = {}
    still_boost = 1.25
    for idx, label in enumerate(label_order):
        weight = float(base[idx])
        if label == "still":
            weight *= still_boost
        class_weight[label] = weight

    sample_weight = np.asarray([class_weight[label_order[idx]] for idx in y_train], dtype=np.float32)

    dataset_weight: dict[str, float] = {}
    if dataset_ids is not None:
        if len(dataset_ids) != y_train.shape[0]:
            raise ValueError("dataset_ids length must match y_train length.")
        if dataset_balance_mode not in {"none", "inverse", "sqrt-inverse"}:
            raise ValueError(f"Unsupported dataset_balance_mode: {dataset_balance_mode}")

        if dataset_balance_mode != "none":
            strength = float(np.clip(dataset_balance_strength, 0.0, 1.0))
            per_dataset_count: dict[str, int] = {}
            for dataset_id in dataset_ids:
                per_dataset_count[dataset_id] = per_dataset_count.get(dataset_id, 0) + 1

            mean_count = float(np.mean(list(per_dataset_count.values())))
            for dataset_id, count in per_dataset_count.items():
                raw_factor = mean_count / float(max(1, count))
                if dataset_balance_mode == "sqrt-inverse":
                    raw_factor = math.sqrt(raw_factor)
                dataset_weight[dataset_id] = float(raw_factor ** strength)

            balance_vector = np.asarray(
                [dataset_weight.get(dataset_id, 1.0) for dataset_id in dataset_ids], dtype=np.float32
            )
            sample_weight *= balance_vector

    return sample_weight, class_weight, dataset_weight


def _build_tuning_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sequence_groups: list[str],
    dataset_ids: list[str],
    tune_split_mode: str,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if tune_split_mode == "stratified-windows":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        return next(splitter.split(x_train, y_train))

    if tune_split_mode == "grouped-subject":
        groups = np.asarray(sequence_groups, dtype=object)
    elif tune_split_mode == "grouped-dataset":
        groups = np.asarray(dataset_ids, dtype=object)
    else:
        raise ValueError(f"Unsupported tune_split_mode: {tune_split_mode}")

    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        LOGGER.warning(
            "Insufficient unique groups for %s split; falling back to stratified windows.",
            tune_split_mode,
        )
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        return next(splitter.split(x_train, y_train))

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    return next(splitter.split(x_train, y_train, groups=groups))


def _tune_model_params(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    base_params: dict[str, float | int | str],
    tune_trials: int,
    tune_metric: str,
    tune_search_mode: str,
    tune_search_space: str,
    tune_split_mode: str,
    train_subject_groups: list[str],
    train_stream_groups: list[str],
    train_dataset_ids: list[str],
    label_order: list[str],
    static_specialist_enabled: bool,
    temporal_decoder: str,
):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for this baseline. Install with 'pip install xgboost'."
        ) from exc

    train_idx, val_idx = _build_tuning_split(
        x_train=x_train,
        y_train=y_train,
        sequence_groups=train_subject_groups,
        dataset_ids=train_dataset_ids,
        tune_split_mode=tune_split_mode,
        test_size=0.15,
    )
    x_fit = x_train[train_idx]
    y_fit = y_train[train_idx]
    w_fit = sample_weight[train_idx]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    seq_fit = [train_stream_groups[idx] for idx in train_idx]
    seq_val = [train_stream_groups[idx] for idx in val_idx]

    static_specialist_labels = [label for label in STATIC_SPECIALIST_LABELS if label in label_order]
    static_specialist_indices = [
        idx for idx, label in enumerate(label_order) if label in set(static_specialist_labels)
    ]

    model_param_keys = {
        "max_depth",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "subsample",
        "colsample_bytree",
        "learning_rate",
        "n_estimators",
    }
    postprocess_param_keys = {
        "decoder_sticky_prior",
        "decoder_cross_coarse_penalty",
        "static_specialist_threshold",
        "static_specialist_blend",
    }

    best_postprocess = {
        "decoder_sticky_prior": 0.25,
        "decoder_cross_coarse_penalty": 0.15,
        "static_specialist_threshold": 0.35,
        "static_specialist_blend": 0.6,
    }

    def evaluate_candidate(candidate: dict[str, float | int]):
        params = dict(base_params)
        params.update({key: value for key, value in candidate.items() if key in model_param_keys})

        model = xgb.XGBClassifier(**params)
        model.fit(x_fit, y_fit, sample_weight=w_fit)

        y_proba_val = np.asarray(model.predict_proba(x_val), dtype=np.float64)
        gate_scores = (
            np.sum(y_proba_val[:, static_specialist_indices], axis=1)
            if static_specialist_indices
            else np.zeros(y_proba_val.shape[0], dtype=np.float64)
        )

        static_model = None
        if static_specialist_enabled and len(static_specialist_labels) >= 2:
            specialist_label_to_local = {label: idx for idx, label in enumerate(static_specialist_labels)}
            train_labels = np.asarray([label_order[idx] for idx in y_fit], dtype=object)
            specialist_mask = np.asarray(
                [label in specialist_label_to_local for label in train_labels],
                dtype=bool,
            )
            if np.any(specialist_mask):
                x_static = x_fit[specialist_mask]
                y_static = np.asarray(
                    [specialist_label_to_local[str(lbl)] for lbl in train_labels[specialist_mask]],
                    dtype=np.int32,
                )
                if len(set(y_static.tolist())) >= 2:
                    static_params = _base_xgb_params(num_classes=len(static_specialist_labels))
                    static_params.update(
                        {
                            "max_depth": int(params["max_depth"]),
                            "min_child_weight": int(params["min_child_weight"]),
                            "subsample": float(params["subsample"]),
                            "colsample_bytree": float(params["colsample_bytree"]),
                            "gamma": float(params["gamma"]),
                            "reg_alpha": float(params["reg_alpha"]),
                            "reg_lambda": float(params["reg_lambda"]),
                        }
                    )
                    static_model = xgb.XGBClassifier(**static_params)
                    static_model.fit(x_static, y_static, sample_weight=w_fit[specialist_mask])

        y_proba_refined = _apply_static_specialist_refinement(
            y_proba=y_proba_val,
            x_values=x_val,
            label_order=label_order,
            static_model=static_model,
            static_labels=static_specialist_labels,
            gate_scores=gate_scores,
            gate_threshold=float(candidate["static_specialist_threshold"]),
            blend=float(candidate["static_specialist_blend"]),
        )

        if temporal_decoder == "causal-hmm":
            log_init, log_trans = _estimate_transition_stats(
                y_train=y_fit,
                sequence_groups=seq_fit,
                num_classes=len(label_order),
                label_order=label_order,
                sticky_prior=float(candidate["decoder_sticky_prior"]),
                cross_coarse_penalty=float(candidate["decoder_cross_coarse_penalty"]),
            )
            val_pred = _causal_hmm_decode(
                proba=y_proba_refined,
                sequence_groups=seq_val,
                log_init_probs=log_init,
                log_trans_probs=log_trans,
            )
        elif temporal_decoder == "none":
            val_pred = np.argmax(y_proba_refined, axis=1).astype(np.int32)
        else:
            raise ValueError(f"Unsupported temporal_decoder for tuning: {temporal_decoder}")

        val_acc = float(np.mean(val_pred == y_val))
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        selected_score = val_acc if tune_metric == "accuracy" else val_macro_f1
        return params, val_acc, val_macro_f1, selected_score

    best_score = -1.0
    best_acc = -1.0
    best_macro_f1 = -1.0
    best_params = dict(base_params)
    trial_count = max(1, tune_trials)

    if tune_search_mode == "random":
        candidates = _sample_hyperparameter_candidates(trial_count, tune_search_space)
        LOGGER.info(
            "Running random hyperparameter search with %d trials (metric=%s, space=%s, split=%s)",
            len(candidates),
            tune_metric,
            tune_search_space,
            tune_split_mode,
        )

        for trial_idx, candidate in enumerate(candidates, start=1):
            params, val_acc, val_macro_f1, selected_score = evaluate_candidate(candidate)
            LOGGER.info(
                "Tune trial %d/%d validation_accuracy=%.4f validation_macro_f1=%.4f selected_score=%.4f params=%s",
                trial_idx,
                len(candidates),
                val_acc,
                val_macro_f1,
                selected_score,
                candidate,
            )
            if selected_score > best_score:
                best_score = selected_score
                best_acc = val_acc
                best_macro_f1 = val_macro_f1
                best_params = params
                best_postprocess = {
                    key: float(candidate[key]) for key in postprocess_param_keys if key in candidate
                }
    elif tune_search_mode == "bayesian":
        try:
            optuna = importlib.import_module("optuna")
        except ImportError as exc:
            raise ImportError(
                "optuna is required for --tune-search-mode bayesian. Install with 'pip install optuna' or use --tune-search-mode random."
            ) from exc

        LOGGER.info(
            "Running bayesian hyperparameter search with %d trials (metric=%s, space=%s, split=%s)",
            trial_count,
            tune_metric,
            tune_search_space,
            tune_split_mode,
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial):
            candidate = _suggest_optuna_candidate(trial, tune_search_space)
            params, val_acc, val_macro_f1, selected_score = evaluate_candidate(candidate)
            trial.set_user_attr("validation_accuracy", val_acc)
            trial.set_user_attr("validation_macro_f1", val_macro_f1)
            LOGGER.info(
                "Tune trial %d/%d validation_accuracy=%.4f validation_macro_f1=%.4f selected_score=%.4f params=%s",
                trial.number + 1,
                trial_count,
                val_acc,
                val_macro_f1,
                selected_score,
                candidate,
            )
            return selected_score

        study.optimize(objective, n_trials=trial_count)
        best_trial = study.best_trial
        best_score = float(best_trial.value)
        best_acc = float(best_trial.user_attrs.get("validation_accuracy", -1.0))
        best_macro_f1 = float(best_trial.user_attrs.get("validation_macro_f1", -1.0))
        best_params = dict(base_params)
        best_params.update({k: v for k, v in best_trial.params.items() if k in model_param_keys})
        best_postprocess = {
            key: float(best_trial.params[key])
            for key in postprocess_param_keys
            if key in best_trial.params
        }
    else:
        raise ValueError(f"Unsupported tune_search_mode: {tune_search_mode}")

    LOGGER.info(
        "Selected tuned params with selected_score=%.4f validation_accuracy=%.4f validation_macro_f1=%.4f",
        best_score,
        best_acc,
        best_macro_f1,
    )
    return best_params, best_postprocess, best_score, best_acc, best_macro_f1


def _fit_with_hard_negative_mining(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    enabled: bool,
    hard_negative_factor: float,
) -> tuple[float, bool]:
    """Fit once, then optionally refit with boosted weight on model mistakes."""
    model.fit(x_train, y_train, sample_weight=sample_weight)
    if not enabled:
        return 0.0, False

    _train_pred_raw = model.predict(x_train)
    if np.asarray(_train_pred_raw).ndim == 2:
        # XGBoost multi:softprob with num_class=2 returns a probability matrix
        train_pred = np.argmax(_train_pred_raw, axis=1).astype(np.int32)
    else:
        train_pred = np.asarray(_train_pred_raw, dtype=np.int32)
    miss_mask = train_pred != y_train
    miss_rate = float(np.mean(miss_mask))
    if not np.any(miss_mask):
        return miss_rate, False

    refined_weight = sample_weight.astype(np.float32, copy=True)
    refined_weight[miss_mask] *= float(hard_negative_factor)
    model.fit(x_train, y_train, sample_weight=refined_weight)
    return miss_rate, True


def _estimate_transition_stats(
    y_train: np.ndarray,
    sequence_groups: list[str],
    num_classes: int,
    label_order: list[str],
    sticky_prior: float,
    cross_coarse_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    return training_estimate_transition_stats(
        y_train=y_train,
        sequence_groups=sequence_groups,
        num_classes=num_classes,
        label_order=label_order,
        sticky_prior=sticky_prior,
        cross_coarse_penalty=cross_coarse_penalty,
    )


def _apply_static_specialist_refinement(
    y_proba: np.ndarray,
    x_values: np.ndarray,
    label_order: list[str],
    static_model,
    static_labels: list[str],
    gate_scores: np.ndarray,
    gate_threshold: float,
    blend: float,
) -> np.ndarray:
    return core_apply_static_specialist_refinement(
        y_proba=y_proba,
        x_values=x_values,
        label_order=label_order,
        static_model=static_model,
        static_labels=static_labels,
        gate_scores=gate_scores,
        gate_threshold=gate_threshold,
        blend=blend,
    )


def _iter_group_ranges(sequence_groups: list[str]) -> list[tuple[int, int]]:
    if not sequence_groups:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(sequence_groups)):
        if sequence_groups[idx] != sequence_groups[idx - 1]:
            ranges.append((start, idx))
            start = idx
    ranges.append((start, len(sequence_groups)))
    return ranges


def _causal_hmm_decode(
    proba: np.ndarray,
    sequence_groups: list[str],
    log_init_probs: np.ndarray,
    log_trans_probs: np.ndarray,
) -> np.ndarray:
    return core_causal_hmm_decode(
        proba=proba,
        sequence_groups=sequence_groups,
        log_init_probs=log_init_probs,
        log_trans_probs=log_trans_probs,
    )


def _predict_hierarchical_proba(
    x_values: np.ndarray,
    label_order: list[str],
    coarse_model,
    fine_models: dict[str, object],
) -> np.ndarray:
    return core_predict_hierarchical_proba(
        x_values=x_values,
        label_order=label_order,
        coarse_model=coarse_model,
        fine_models=fine_models,
    )


def _decode_predictions(
    decoder: str,
    y_pred_raw: np.ndarray,
    y_proba_raw: np.ndarray,
    sequence_groups: list[str],
    log_init_probs: np.ndarray,
    log_trans_probs: np.ndarray,
) -> np.ndarray:
    return core_decode_predictions(
        decoder=decoder,
        y_pred_raw=y_pred_raw,
        y_proba_raw=y_proba_raw,
        sequence_groups=sequence_groups,
        log_init_probs=log_init_probs,
        log_trans_probs=log_trans_probs,
    )


def _compute_class_metrics(label_order: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for idx, label in enumerate(label_order):
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        support = int(np.sum(y_true == idx))
        pred_count = int(np.sum(y_pred == idx))
        recall = float(tp / support) if support else 0.0
        precision = float(tp / pred_count) if pred_count else 0.0
        rows.append(
            {
                "label": label,
                "support": support,
                "precision": precision,
                "recall": recall,
            }
        )
    return rows


def _write_confusion_matrix(path: Path, label_order: list[str], conf: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *label_order])
        for row_idx, label in enumerate(label_order):
            writer.writerow([label, *conf[row_idx].tolist()])


def _append_experiment_history(path: Path, row: dict[str, str]) -> None:
    fieldnames = [
        "timestamp_utc",
        "output_dir",
        "overall_accuracy_raw",
        "overall_accuracy_smoothed",
        "num_train_windows",
        "num_test_windows",
        "num_features",
        "include_train_only",
        "max_files",
        "tune_small_search",
        "tune_trials",
        "tune_metric",
        "tune_search_mode",
        "tune_search_space",
        "tune_split_mode",
        "dataset_balance_mode",
        "dataset_balance_strength",
        "temporal_decoder",
        "decoder_sticky_prior",
        "decoder_cross_coarse_penalty",
        "use_hierarchical",
        "static_specialist_enabled",
        "static_specialist_threshold",
        "static_specialist_blend",
        "hard_negative_mining_enabled",
        "hard_negative_factor",
        "target_labels",
        "class_weights",
        "xgb_params",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    needs_rewrite = False
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames != fieldnames:
                needs_rewrite = True
            for existing in reader:
                normalized = {name: existing.get(name, "") for name in fieldnames}
                existing_rows.append(normalized)

    if needs_rewrite:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow(existing)

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not path.exists() or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_group_metrics(
    path: Path,
    group_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    group_column: str,
) -> None:
    by_group: dict[str, list[int]] = {}
    for idx, group_name in enumerate(group_names):
        by_group.setdefault(group_name, []).append(idx)

    rows: list[tuple[str, int, float]] = []
    for group_name in sorted(by_group):
        indices = by_group[group_name]
        subset_true = y_true[indices]
        subset_pred = y_pred[indices]
        accuracy = float(np.mean(subset_true == subset_pred))
        rows.append((group_name, len(indices), accuracy))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([group_column, "num_windows", "accuracy"])
        for group_name, count, accuracy in rows:
            writer.writerow([group_name, count, f"{accuracy:.6f}"])


def _write_per_dataset_metrics(path: Path, dataset_names: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    _write_group_metrics(
        path=path,
        group_names=dataset_names,
        y_true=y_true,
        y_pred=y_pred,
        group_column="dataset_id",
    )


def _write_per_placement_metrics(path: Path, placement_names: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    _write_group_metrics(
        path=path,
        group_names=placement_names,
        y_true=y_true,
        y_pred=y_pred,
        group_column="placement_label",
    )


def main() -> None:
    args = parse_args()
    if args.enforce_causal_streaming and args.temporal_decoder not in {"none", "causal-hmm"}:
        raise ValueError("--enforce-causal-streaming requires a causal temporal decoder.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )

    target_labels = tuple(args.target_labels)
    target_set = set(target_labels)

    LOGGER.info(
        "Starting baseline training with contract_dir=%s output_dir=%s include_train_only=%s max_files=%d",
        args.contract_dir,
        args.output_dir,
        not args.exclude_train_only,
        args.max_files,
    )
    LOGGER.info("Target labels: %s", ", ".join(target_labels))

    (
        x_train,
        y_train_str,
        x_test,
        y_test_str,
        dataset_train,
        dataset_test,
        feature_names,
        train_sequence_groups,
        test_sequence_groups,
        train_subject_groups,
        _,
        _,
        placement_test,
    ) = collect_training_examples(
        contract_dir=args.contract_dir,
        target_labels=target_set,
        include_train_only=not args.exclude_train_only,
        max_files=args.max_files,
    )

    label_order = [label for label in target_labels if label in set(y_train_str)]
    label_to_idx = {label: idx for idx, label in enumerate(label_order)}

    y_train = np.asarray([label_to_idx[label] for label in y_train_str], dtype=np.int32)

    test_mask = np.asarray([label in label_to_idx for label in y_test_str], dtype=bool)
    x_test = x_test[test_mask]
    y_test_str = y_test_str[test_mask]
    dataset_test = [dataset_test[idx] for idx, keep in enumerate(test_mask) if keep]
    placement_test = [placement_test[idx] for idx, keep in enumerate(test_mask) if keep]
    y_test = np.asarray([label_to_idx[label] for label in y_test_str], dtype=np.int32)

    if x_test.size == 0:
        raise ValueError("No compatible test windows remain after label filtering.")

    LOGGER.info(
        "Prepared matrices: x_train=%s x_test=%s num_features=%d num_classes=%d",
        x_train.shape,
        x_test.shape,
        len(feature_names),
        len(label_order),
    )

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for this baseline. Install with 'pip install xgboost'."
        ) from exc

    sample_weight, class_weights, dataset_weights = _compute_sample_weights(
        y_train,
        label_order,
        dataset_ids=dataset_train,
        dataset_balance_mode=args.dataset_balance_mode,
        dataset_balance_strength=args.dataset_balance_strength,
    )
    LOGGER.info("Using class-balanced sample weights (still boosted): %s", class_weights)
    if dataset_weights:
        LOGGER.info(
            "Using dataset balancing mode=%s strength=%.2f weights=%s",
            args.dataset_balance_mode,
            args.dataset_balance_strength,
            dataset_weights,
        )

    model_params = _base_xgb_params(num_classes=len(label_order))
    model_params = _apply_xgb_param_overrides(model_params, args)
    static_specialist_enabled = not args.disable_static_specialist
    effective_decoder_sticky_prior = float(args.decoder_sticky_prior)
    effective_decoder_cross_coarse_penalty = float(args.decoder_cross_coarse_penalty)
    effective_static_specialist_threshold = float(args.static_specialist_threshold)
    effective_static_specialist_blend = float(args.static_specialist_blend)
    tuning_best_score = None
    tuning_best_val_acc = None
    tuning_best_val_macro_f1 = None
    if args.tune_small_search:
        (
            model_params,
            tuned_postprocess,
            tuning_best_score,
            tuning_best_val_acc,
            tuning_best_val_macro_f1,
        ) = _tune_model_params(
            x_train=x_train,
            y_train=y_train,
            sample_weight=sample_weight,
            base_params=model_params,
            tune_trials=args.tune_trials,
            tune_metric=args.tune_metric,
            tune_search_mode=args.tune_search_mode,
            tune_search_space=args.tune_search_space,
            tune_split_mode=args.tune_split_mode,
            train_subject_groups=train_subject_groups,
            train_stream_groups=train_sequence_groups,
            train_dataset_ids=dataset_train,
            label_order=label_order,
            static_specialist_enabled=static_specialist_enabled,
            temporal_decoder=args.temporal_decoder,
        )
        effective_decoder_sticky_prior = float(
            tuned_postprocess.get("decoder_sticky_prior", effective_decoder_sticky_prior)
        )
        effective_decoder_cross_coarse_penalty = float(
            tuned_postprocess.get("decoder_cross_coarse_penalty", effective_decoder_cross_coarse_penalty)
        )
        effective_static_specialist_threshold = float(
            tuned_postprocess.get("static_specialist_threshold", effective_static_specialist_threshold)
        )
        effective_static_specialist_blend = float(
            tuned_postprocess.get("static_specialist_blend", effective_static_specialist_blend)
        )

    log_init_probs, log_trans_probs = _estimate_transition_stats(
        y_train=y_train,
        sequence_groups=train_sequence_groups,
        num_classes=len(label_order),
        label_order=label_order,
        sticky_prior=effective_decoder_sticky_prior,
        cross_coarse_penalty=effective_decoder_cross_coarse_penalty,
    )

    model = xgb.XGBClassifier(**model_params)
    LOGGER.info("Training XGBoost model on CPU with tree_method=%s", "hist")
    hard_negative_enabled = not args.disable_hard_negative_mining
    base_hnm_miss_rate, base_hnm_applied = _fit_with_hard_negative_mining(
        model=model,
        x_train=x_train,
        y_train=y_train,
        sample_weight=sample_weight,
        enabled=hard_negative_enabled,
        hard_negative_factor=args.hard_negative_factor,
    )
    if hard_negative_enabled:
        LOGGER.info(
            "Base model hard-negative mining: applied=%s pre_refit_miss_rate=%.4f factor=%.2f",
            base_hnm_applied,
            base_hnm_miss_rate,
            args.hard_negative_factor,
        )

    coarse_model = None
    fine_models: dict[str, object] = {}
    static_specialist_model = None
    static_specialist_labels: list[str] = []
    static_specialist_enabled = not args.disable_static_specialist
    static_gate_train = np.zeros(x_train.shape[0], dtype=np.float64)
    if args.use_hierarchical:
        coarse_targets = np.asarray(
            [COARSE_LABEL_ORDER.index(FINE_TO_COARSE[label_order[idx]]) for idx in y_train],
            dtype=np.int32,
        )
        coarse_weight, _, _ = _compute_sample_weights(coarse_targets, list(COARSE_LABEL_ORDER))
        coarse_params = _base_xgb_params(num_classes=len(COARSE_LABEL_ORDER))
        coarse_params.update(
            {
                "max_depth": int(model_params["max_depth"]),
                "min_child_weight": int(model_params["min_child_weight"]),
                "subsample": float(model_params["subsample"]),
                "colsample_bytree": float(model_params["colsample_bytree"]),
                "gamma": float(model_params["gamma"]),
                "reg_alpha": float(model_params["reg_alpha"]),
                "reg_lambda": float(model_params["reg_lambda"]),
            }
        )
        coarse_model = xgb.XGBClassifier(**coarse_params)
        coarse_hnm_miss_rate, coarse_hnm_applied = _fit_with_hard_negative_mining(
            model=coarse_model,
            x_train=x_train,
            y_train=coarse_targets,
            sample_weight=coarse_weight,
            enabled=hard_negative_enabled,
            hard_negative_factor=args.hard_negative_factor,
        )
        if hard_negative_enabled:
            LOGGER.info(
                "Coarse model hard-negative mining: applied=%s pre_refit_miss_rate=%.4f",
                coarse_hnm_applied,
                coarse_hnm_miss_rate,
            )

            coarse_proba_train = np.asarray(coarse_model.predict_proba(x_train), dtype=np.float64)
            static_gate_train = coarse_proba_train[:, int(COARSE_LABEL_ORDER.index("static"))]

        for coarse_label in COARSE_LABEL_ORDER:
            fine_indices = [idx for idx, lbl in enumerate(label_order) if FINE_TO_COARSE.get(lbl) == coarse_label]
            if len(fine_indices) <= 1:
                continue
            fine_mask = np.asarray([FINE_TO_COARSE.get(lbl) == coarse_label for lbl in y_train_str], dtype=bool)
            x_fine = x_train[fine_mask]
            y_fine_idx = y_train[fine_mask]
            local_map = {fine_idx: local_idx for local_idx, fine_idx in enumerate(fine_indices)}
            y_fine = np.asarray([local_map[int(idx)] for idx in y_fine_idx], dtype=np.int32)
            if x_fine.shape[0] == 0 or len(set(y_fine.tolist())) < 2:
                continue
            fine_weight, _, _ = _compute_sample_weights(
                y_fine,
                [label_order[idx] for idx in fine_indices],
            )
            fine_params = _base_xgb_params(num_classes=len(fine_indices))
            fine_params.update(
                {
                    "max_depth": int(model_params["max_depth"]),
                    "min_child_weight": int(model_params["min_child_weight"]),
                    "subsample": float(model_params["subsample"]),
                    "colsample_bytree": float(model_params["colsample_bytree"]),
                    "gamma": float(model_params["gamma"]),
                    "reg_alpha": float(model_params["reg_alpha"]),
                    "reg_lambda": float(model_params["reg_lambda"]),
                }
            )
            fine_model = xgb.XGBClassifier(**fine_params)
            fine_hnm_miss_rate, fine_hnm_applied = _fit_with_hard_negative_mining(
                model=fine_model,
                x_train=x_fine,
                y_train=y_fine,
                sample_weight=fine_weight,
                enabled=hard_negative_enabled,
                hard_negative_factor=args.hard_negative_factor,
            )
            if hard_negative_enabled:
                LOGGER.info(
                    "Fine model (%s) hard-negative mining: applied=%s pre_refit_miss_rate=%.4f",
                    coarse_label,
                    fine_hnm_applied,
                    fine_hnm_miss_rate,
                )
            fine_models[coarse_label] = fine_model

    if static_specialist_enabled:
        static_specialist_labels = [label for label in STATIC_SPECIALIST_LABELS if label in label_order]
        if len(static_specialist_labels) >= 2:
            specialist_label_to_local = {label: idx for idx, label in enumerate(static_specialist_labels)}
            train_labels = np.asarray([label_order[idx] for idx in y_train], dtype=object)
            specialist_mask = np.asarray([lbl in specialist_label_to_local for lbl in train_labels], dtype=bool)
            if np.any(specialist_mask):
                x_static = x_train[specialist_mask]
                y_static = np.asarray(
                    [specialist_label_to_local[str(lbl)] for lbl in train_labels[specialist_mask]],
                    dtype=np.int32,
                )
                static_weights = sample_weight[specialist_mask]

                static_params = _base_xgb_params(num_classes=len(static_specialist_labels))
                static_params.update(
                    {
                        "max_depth": int(model_params["max_depth"]),
                        "min_child_weight": int(model_params["min_child_weight"]),
                        "subsample": float(model_params["subsample"]),
                        "colsample_bytree": float(model_params["colsample_bytree"]),
                        "gamma": float(model_params["gamma"]),
                        "reg_alpha": float(model_params["reg_alpha"]),
                        "reg_lambda": float(model_params["reg_lambda"]),
                    }
                )
                static_specialist_model = xgb.XGBClassifier(**static_params)
                specialist_hnm_miss_rate, specialist_hnm_applied = _fit_with_hard_negative_mining(
                    model=static_specialist_model,
                    x_train=x_static,
                    y_train=y_static,
                    sample_weight=static_weights,
                    enabled=hard_negative_enabled,
                    hard_negative_factor=args.hard_negative_factor,
                )
                LOGGER.info(
                    "Static specialist trained on %d windows labels=%s hard_negative_applied=%s pre_refit_miss_rate=%.4f",
                    x_static.shape[0],
                    ",".join(static_specialist_labels),
                    specialist_hnm_applied,
                    specialist_hnm_miss_rate,
                )

    LOGGER.info("Training complete; running prediction on %d test windows", x_test.shape[0])

    if coarse_model is not None:
        y_proba_raw = _predict_hierarchical_proba(
            x_values=x_test,
            label_order=label_order,
            coarse_model=coarse_model,
            fine_models=fine_models,
        )
        coarse_proba_test = np.asarray(coarse_model.predict_proba(x_test), dtype=np.float64)
        static_gate_test = coarse_proba_test[:, int(COARSE_LABEL_ORDER.index("static"))]
        y_pred_raw = np.argmax(y_proba_raw, axis=1).astype(np.int32)
    else:
        y_proba_raw = np.asarray(model.predict_proba(x_test), dtype=np.float64)
        static_idx = [idx for idx, label in enumerate(label_order) if label in STATIC_SPECIALIST_LABELS]
        if static_idx:
            static_gate_test = np.sum(y_proba_raw[:, static_idx], axis=1)
        else:
            static_gate_test = np.zeros(y_proba_raw.shape[0], dtype=np.float64)
        y_pred_raw = np.asarray(model.predict(x_test), dtype=np.int32)

    y_proba_refined = _apply_static_specialist_refinement(
        y_proba=y_proba_raw,
        x_values=x_test,
        label_order=label_order,
        static_model=static_specialist_model,
        static_labels=static_specialist_labels,
        gate_scores=static_gate_test,
        gate_threshold=effective_static_specialist_threshold,
        blend=effective_static_specialist_blend,
    )
    y_proba_raw = y_proba_refined
    y_pred_raw = np.argmax(y_proba_raw, axis=1).astype(np.int32)

    y_pred = _decode_predictions(
        decoder=args.temporal_decoder,
        y_pred_raw=y_pred_raw,
        y_proba_raw=y_proba_raw,
        sequence_groups=test_sequence_groups,
        log_init_probs=log_init_probs,
        log_trans_probs=log_trans_probs,
    )

    overall_accuracy_raw = float(np.mean(y_pred_raw == y_test))
    overall_accuracy = float(np.mean(y_pred == y_test))
    conf_raw = confusion_matrix(y_test, y_pred_raw, labels=np.arange(len(label_order)))
    conf = confusion_matrix(y_test, y_pred, labels=np.arange(len(label_order)))
    LOGGER.info(
        "Evaluation complete; raw_accuracy=%.4f decoded_accuracy=%.4f (decoder=%s)",
        overall_accuracy_raw,
        overall_accuracy,
        args.temporal_decoder,
    )

    output_dir = args.output_dir

    joblib.dump(model, output_dir / "xgboost_model.joblib")
    joblib.dump(coarse_model, output_dir / "coarse_model.joblib")
    joblib.dump(fine_models, output_dir / "fine_models.joblib")
    joblib.dump(static_specialist_model, output_dir / "static_specialist_model.joblib")
    np.savez_compressed(
        output_dir / "transition_stats.npz",
        log_init_probs=np.asarray(log_init_probs, dtype=np.float64),
        log_trans_probs=np.asarray(log_trans_probs, dtype=np.float64),
    )

    with (output_dir / "feature_schema.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_names": feature_names,
                "num_features": len(feature_names),
                "placement_labels": list(PLACEMENT_LABELS),
                "target_labels": label_order,
            },
            handle,
            indent=2,
        )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "overall_accuracy": overall_accuracy,
                "overall_accuracy_raw": overall_accuracy_raw,
                "num_train_windows": int(x_train.shape[0]),
                "num_test_windows": int(x_test.shape[0]),
                "target_labels": label_order,
                "included_train_only": not args.exclude_train_only,
                "temporal_decoder": args.temporal_decoder,
                "use_hierarchical": args.use_hierarchical,
                "tune_small_search": args.tune_small_search,
                "tune_trials": args.tune_trials,
                "tune_metric": args.tune_metric,
                "tune_search_mode": args.tune_search_mode,
                "tune_search_space": args.tune_search_space,
                "tune_split_mode": args.tune_split_mode,
                "dataset_balance_mode": args.dataset_balance_mode,
                "dataset_balance_strength": args.dataset_balance_strength,
                "tuning_best_validation_score": tuning_best_score,
                "tuning_best_validation_accuracy": tuning_best_val_acc,
                "tuning_best_validation_macro_f1": tuning_best_val_macro_f1,
                "decoder_sticky_prior": effective_decoder_sticky_prior,
                "decoder_cross_coarse_penalty": effective_decoder_cross_coarse_penalty,
                "static_specialist_enabled": static_specialist_enabled,
                "static_specialist_threshold": effective_static_specialist_threshold,
                "static_specialist_blend": effective_static_specialist_blend,
                "hard_negative_mining_enabled": hard_negative_enabled,
                "hard_negative_factor": args.hard_negative_factor,
                "hard_negative_base_pre_refit_miss_rate": base_hnm_miss_rate,
                "hard_negative_base_applied": base_hnm_applied,
                "class_weights": class_weights,
                "xgb_params": model_params,
            },
            handle,
            indent=2,
        )

    _write_confusion_matrix(output_dir / "confusion_matrix_raw.csv", label_order, conf_raw)
    _write_confusion_matrix(output_dir / "confusion_matrix.csv", label_order, conf)

    _write_per_dataset_metrics(
        path=output_dir / "per_dataset_metrics_raw.csv",
        dataset_names=dataset_test,
        y_true=y_test,
        y_pred=y_pred_raw,
    )

    _write_per_placement_metrics(
        path=output_dir / "per_placement_metrics_raw.csv",
        placement_names=placement_test,
        y_true=y_test,
        y_pred=y_pred_raw,
    )

    _write_per_dataset_metrics(
        path=output_dir / "per_dataset_metrics.csv",
        dataset_names=dataset_test,
        y_true=y_test,
        y_pred=y_pred,
    )

    _write_per_placement_metrics(
        path=output_dir / "per_placement_metrics.csv",
        placement_names=placement_test,
        y_true=y_test,
        y_pred=y_pred,
    )

    class_metrics = {
        "raw": _compute_class_metrics(label_order, y_test, y_pred_raw),
        "decoded": _compute_class_metrics(label_order, y_test, y_pred),
        "decoder": args.temporal_decoder,
    }
    with (output_dir / "class_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(class_metrics, handle, indent=2)

    experiment_row = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "output_dir": str(output_dir),
        "overall_accuracy_raw": f"{overall_accuracy_raw:.6f}",
        "overall_accuracy_smoothed": f"{overall_accuracy:.6f}",
        "num_train_windows": str(int(x_train.shape[0])),
        "num_test_windows": str(int(x_test.shape[0])),
        "num_features": str(len(feature_names)),
        "include_train_only": str(not args.exclude_train_only),
        "max_files": str(args.max_files),
        "tune_small_search": str(args.tune_small_search),
        "tune_trials": str(args.tune_trials),
        "tune_metric": args.tune_metric,
        "tune_search_mode": args.tune_search_mode,
        "tune_search_space": args.tune_search_space,
        "tune_split_mode": args.tune_split_mode,
        "dataset_balance_mode": args.dataset_balance_mode,
        "dataset_balance_strength": f"{args.dataset_balance_strength:.4f}",
        "temporal_decoder": args.temporal_decoder,
        "decoder_sticky_prior": f"{effective_decoder_sticky_prior:.4f}",
        "decoder_cross_coarse_penalty": f"{effective_decoder_cross_coarse_penalty:.4f}",
        "use_hierarchical": str(args.use_hierarchical),
        "static_specialist_enabled": str(static_specialist_enabled),
        "static_specialist_threshold": f"{effective_static_specialist_threshold:.4f}",
        "static_specialist_blend": f"{effective_static_specialist_blend:.4f}",
        "hard_negative_mining_enabled": str(hard_negative_enabled),
        "hard_negative_factor": f"{args.hard_negative_factor:.4f}",
        "target_labels": json.dumps(label_order),
        "class_weights": json.dumps(class_weights, sort_keys=True),
        "xgb_params": json.dumps(model_params, sort_keys=True),
    }
    _append_experiment_history(args.experiment_log, experiment_row)

    with (output_dir / "experiment_record.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_row, handle, indent=2)

    inference_bundle = {
        "base_model": model,
        "coarse_model": coarse_model,
        "fine_models": fine_models,
        "static_specialist_model": static_specialist_model,
        "label_order": label_order,
        "feature_names": feature_names,
        "placement_labels": list(PLACEMENT_LABELS),
        "temporal_decoder": args.temporal_decoder,
        "log_init_probs": np.asarray(log_init_probs, dtype=np.float64),
        "log_trans_probs": np.asarray(log_trans_probs, dtype=np.float64),
        "use_hierarchical": bool(args.use_hierarchical),
        "static_specialist_labels": static_specialist_labels,
        "static_specialist_threshold": float(effective_static_specialist_threshold),
        "static_specialist_blend": float(effective_static_specialist_blend),
        "decoder_sticky_prior": float(effective_decoder_sticky_prior),
        "decoder_cross_coarse_penalty": float(effective_decoder_cross_coarse_penalty),
        "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "window_samples": WINDOW_SAMPLES,
    }
    joblib.dump(inference_bundle, output_dir / "inference_bundle.joblib")
    # Standalone base model for the worker pipeline (artifacts/model/model_pocket_only.joblib)
    joblib.dump(model, output_dir / "model_pocket_only.joblib")

    with (output_dir / "inference_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "label_order": label_order,
                "feature_names": feature_names,
                "placement_labels": list(PLACEMENT_LABELS),
                "temporal_decoder": args.temporal_decoder,
                "use_hierarchical": bool(args.use_hierarchical),
                "static_specialist_labels": static_specialist_labels,
                "static_specialist_threshold": float(effective_static_specialist_threshold),
                "static_specialist_blend": float(effective_static_specialist_blend),
                "decoder_sticky_prior": float(effective_decoder_sticky_prior),
                "decoder_cross_coarse_penalty": float(effective_decoder_cross_coarse_penalty),
                "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
                "window_samples": WINDOW_SAMPLES,
            },
            handle,
            indent=2,
        )

    LOGGER.info("Wrote model and evaluation artifacts to %s", output_dir)
    LOGGER.info("Updated experiment history at %s", args.experiment_log)


if __name__ == "__main__":
    main()
