"""Train and evaluate a first XGBoost baseline on MotionLens contract windows."""

from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import confusion_matrix

from processing.preprocessing import AccelerometerSegment, apply_gravity_split


TARGET_SAMPLE_RATE_HZ = 50.0
WINDOW_SAMPLES = 128

DEFAULT_TARGET_LABELS = (
    "walk",
    "run",
    "stairs",
    "sit",
    "stand",
    "lay",
    "transitions",
    "locomotion-other",
)

PLACEMENT_LABELS = (
    "chest",
    "front_center_mid",
    "front_center_lower",
    "front_side_left_mid",
    "front_side_right_mid",
    "lateral_left_lower",
    "lateral_right_lower",
    "front_pocket",
    "unknown_free_living",
)

AXES = ("x", "y", "z")

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost baseline from MotionLens contract parquet artifacts."
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
        default=Path("output") / "model_baseline",
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
    return parser.parse_args()


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0))


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


def _window_feature_names() -> list[str]:
    names: list[str] = []
    for axis in AXES:
        names.extend(
            [
                f"body_{axis}_mean",
                f"body_{axis}_std",
                f"body_{axis}_rms",
                f"body_{axis}_mad",
                f"body_{axis}_min",
                f"body_{axis}_max",
                f"body_{axis}_zcr",
            ]
        )

    names.extend(
        [
            "body_corr_xy",
            "body_corr_xz",
            "body_corr_yz",
            "body_mag_dom_freq_hz",
            "body_mag_spectral_entropy",
            "body_mag_energy_0p5_4_hz",
            "body_mag_energy_0p1_0p5_hz",
            "gravity_mag_mean",
            "gravity_angle_stability_std",
        ]
    )

    for placement in PLACEMENT_LABELS:
        names.append(f"placement_{placement}")
    return names


def compute_window_features(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    placement_label: str,
) -> np.ndarray:
    segment = AccelerometerSegment(
        time_seconds=np.arange(acc_x.size, dtype=np.float64) / TARGET_SAMPLE_RATE_HZ,
        acc_x=acc_x.astype(np.float64, copy=False),
        acc_y=acc_y.astype(np.float64, copy=False),
        acc_z=acc_z.astype(np.float64, copy=False),
    )
    split = apply_gravity_split(segment, sample_rate_hz=TARGET_SAMPLE_RATE_HZ)

    body_axes = {
        "x": split.body_x,
        "y": split.body_y,
        "z": split.body_z,
    }

    features: list[float] = []
    for axis in AXES:
        values = body_axes[axis]
        features.extend(
            [
                float(np.mean(values)),
                _safe_std(values),
                float(np.sqrt(np.mean(values ** 2))),
                float(np.mean(np.abs(values - np.mean(values)))),
                float(np.min(values)),
                float(np.max(values)),
                _zero_crossing_rate(values),
            ]
        )

    features.extend(
        [
            _safe_corr(body_axes["x"], body_axes["y"]),
            _safe_corr(body_axes["x"], body_axes["z"]),
            _safe_corr(body_axes["y"], body_axes["z"]),
        ]
    )

    body_mag = np.sqrt(split.body_x ** 2 + split.body_y ** 2 + split.body_z ** 2)
    centered_mag = body_mag - np.mean(body_mag)
    fft_values = np.fft.rfft(centered_mag)
    freqs = np.fft.rfftfreq(centered_mag.size, d=1.0 / TARGET_SAMPLE_RATE_HZ)
    power = np.abs(fft_values) ** 2

    features.extend(
        [
            _dominant_frequency(power, freqs),
            _spectral_entropy(power),
            _band_energy(power, freqs, 0.5, 4.0),
            _band_energy(power, freqs, 0.1, 0.5),
        ]
    )

    gravity_mag = np.sqrt(split.gravity_x ** 2 + split.gravity_y ** 2 + split.gravity_z ** 2)
    features.extend(
        [
            float(np.mean(gravity_mag)),
            _gravity_angle_stability(split.gravity_x, split.gravity_y, split.gravity_z),
        ]
    )

    placement_set = set(PLACEMENT_LABELS)
    normalized_placement = placement_label if placement_label in placement_set else "unknown_free_living"
    for placement in PLACEMENT_LABELS:
        features.append(1.0 if placement == normalized_placement else 0.0)

    return np.asarray(features, dtype=np.float32)


def _iter_window_records(parquet_path: Path):
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
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

    for parquet_file in all_files:
        for record in _iter_window_records(parquet_file):
            activity_label = str(record["activity_label"])
            split = str(record["split"])

            if activity_label not in target_labels:
                continue

            features = compute_window_features(
                acc_x=record["acc_x"],
                acc_y=record["acc_y"],
                acc_z=record["acc_z"],
                placement_label=str(record["placement_label"]),
            )

            if split == "test":
                x_test.append(features)
                y_test.append(activity_label)
                dataset_test.append(str(record["dataset_id"]))
            elif split == "train" or (include_train_only and split == "train_only"):
                x_train.append(features)
                y_train.append(activity_label)
                dataset_train.append(str(record["dataset_id"]))

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

    feature_names = _window_feature_names()
    return (
        np.vstack(x_train),
        np.asarray(y_train),
        np.vstack(x_test),
        np.asarray(y_test),
        dataset_train,
        dataset_test,
        feature_names,
    )


def _write_per_dataset_metrics(path: Path, dataset_names: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    by_dataset: dict[str, list[int]] = {}
    for idx, dataset_id in enumerate(dataset_names):
        by_dataset.setdefault(dataset_id, []).append(idx)

    rows: list[tuple[str, int, float]] = []
    for dataset_id in sorted(by_dataset):
        indices = by_dataset[dataset_id]
        subset_true = y_true[indices]
        subset_pred = y_pred[indices]
        accuracy = float(np.mean(subset_true == subset_pred))
        rows.append((dataset_id, len(indices), accuracy))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_id", "num_windows", "accuracy"])
        for dataset_id, count, accuracy in rows:
            writer.writerow([dataset_id, count, f"{accuracy:.6f}"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

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

    x_train, y_train_str, x_test, y_test_str, _, dataset_test, feature_names = collect_training_examples(
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

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_order),
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )
    LOGGER.info("Training XGBoost model on CPU with tree_method=%s", "hist")
    model.fit(x_train, y_train)
    LOGGER.info("Training complete; running prediction on %d test windows", x_test.shape[0])

    y_pred = model.predict(x_test)

    overall_accuracy = float(np.mean(y_pred == y_test))
    conf = confusion_matrix(y_test, y_pred, labels=np.arange(len(label_order)))
    LOGGER.info("Evaluation complete; overall_accuracy=%.4f", overall_accuracy)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "xgboost_model.joblib")

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
                "num_train_windows": int(x_train.shape[0]),
                "num_test_windows": int(x_test.shape[0]),
                "target_labels": label_order,
                "included_train_only": not args.exclude_train_only,
            },
            handle,
            indent=2,
        )

    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *label_order])
        for row_idx, label in enumerate(label_order):
            writer.writerow([label, *conf[row_idx].tolist()])

    _write_per_dataset_metrics(
        path=output_dir / "per_dataset_metrics.csv",
        dataset_names=dataset_test,
        y_true=y_test,
        y_pred=y_pred,
    )
    LOGGER.info("Wrote model and evaluation artifacts to %s", output_dir)


if __name__ == "__main__":
    main()
