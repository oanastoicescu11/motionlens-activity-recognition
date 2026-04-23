"""Build a MotionLens-contract unified accelerometer dataset.

This script rebuilds the approved MotionLens training-pool datasets onto one
canonical processing contract:

- accelerometer-only canonical sample stream with columns ``acc_x``, ``acc_y``,
  and ``acc_z`` in m/s^2
- 50 Hz target sampling rate
- 128-sample windows with 50% overlap
- explicit canonical activity, placement, and split metadata

The implementation is intentionally Arrow-first and streaming-oriented. Large
sample tables are written incrementally to parquet so the pipeline does not need
to materialize the full unified dataset in memory.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from processing.preprocessing import (
    AccelerometerSegment,
    preprocess_numeric_timeseries as preprocess_numeric_segments,
    preprocess_timestamp_timeseries as preprocess_timestamp_segments,
    resample_accelerometer_segment,
    sanitize_numeric_timeseries as shared_sanitize_numeric_timeseries,
    sanitize_timestamp_timeseries as shared_sanitize_timestamp_timeseries,
)


LOGGER = logging.getLogger(__name__)

TARGET_SAMPLE_RATE_HZ = 50
WINDOW_SAMPLES = 128
WINDOW_STEP_SAMPLES = 64
WINDOW_DURATION_SECONDS = WINDOW_SAMPLES / TARGET_SAMPLE_RATE_HZ
SECONDS_PER_SAMPLE = 1.0 / TARGET_SAMPLE_RATE_HZ

MS2_PER_G = 9.81
WISDM_MS2_PER_UNIT = MS2_PER_G / 10.0

DEFAULT_OUTPUT_DIR = Path("output") / "motionlens_contract"
DEFAULT_CACHE_DIR = Path("output") / "whar_datasets_cache"
DEFAULT_REALWORLD_DIR = Path("data") / "realworld2016_dataset"
DEFAULT_IPHONE_DIR = Path("data") / "Acceleration"

CORE_REPORT_LABELS = {"walk", "run", "stairs", "sit", "stand", "lay"}

IPHONE_ACTIVITIES = (
    "standingstill",
    "sittingstanding",
    "running",
    "walking",
    "sitting",
    "laying",
)

IPHONE_PLACEMENT_ALIASES = {
    "front-side-left": "front-side-left-mid",
    "front-side-right": "front-side-right-mid",
}

IPHONE_PLACEMENT_TO_MODEL = {
    "front-center-upper": "chest",
    "front-center-mid": "front_center_mid",
    "front-center-lower": "front_center_lower",
    "front-side-left-mid": "front_side_left_mid",
    "front-side-right-mid": "front_side_right_mid",
    "lateral-left-lower": "lateral_left_lower",
    "lateral-right-lower": "lateral_right_lower",
}


@dataclass(frozen=True)
class StreamSpec:
    """One accelerometer stream exported as its own canonical placement source."""

    stream_id: str
    source_columns: tuple[str, str, str]
    placement_label: str
    unit_scale_to_ms2: float


@dataclass(frozen=True)
class DatasetSpec:
    """Source configuration for one contract dataset."""

    dataset_id: str
    dataset_enum: str
    source_kind: str
    source_root: Path
    streams: tuple[StreamSpec, ...]
    activity_map: dict[str, str | None]
    holdout_kind: str
    holdout_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionMeta:
    """Canonical session metadata shared by all windows in the session."""

    dataset_id: str
    dataset_enum: str
    stream_id: str
    subject_local_id: str
    source_session_id: str
    activity_name_raw: str
    activity_label: str
    placement_label: str
    split: str

    @property
    def global_subject_id(self) -> str:
        return f"{self.dataset_id}:subject:{self.subject_local_id}"

    @property
    def global_session_id(self) -> str:
        return f"{self.dataset_id}:{self.stream_id}:session:{self.source_session_id}"


@dataclass(frozen=True)
class BuiltWindow:
    """Window-level metadata generated from one canonical session."""

    window_id: str
    session_meta: SessionMeta
    window_index: int


def configure_logging(verbose: bool) -> None:
    """Configure module logging."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a MotionLens-contract unified processing dataset from the "
            "approved training-pool sources."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the unified dataset artifacts will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Root directory for cached WHAR session artifacts.",
    )
    parser.add_argument(
        "--realworld-dir",
        type=Path,
        default=DEFAULT_REALWORLD_DIR,
        help="Root directory for the local RealWorld2016 raw files.",
    )
    parser.add_argument(
        "--iphone-dir",
        type=Path,
        default=DEFAULT_IPHONE_DIR,
        help="Root directory for the local iPhone placement sweep files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["ALL"],
        metavar="DATASET",
        help=(
            "Subset of contract datasets to build. Accepts dataset ids or enums, "
            "for example HAPT WISDM REAL_WORLD IPHONE_SWEEP."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete prior output artifacts before writing new ones.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def build_dataset_specs(
    cache_dir: Path,
    realworld_dir: Path,
    iphone_dir: Path,
) -> dict[str, DatasetSpec]:
    """Build the approved contract dataset registry."""

    return {
        "hapt": DatasetSpec(
            dataset_id="hapt",
            dataset_enum="HAPT",
            source_kind="cached_whar",
            source_root=cache_dir / "hapt",
            streams=(
                StreamSpec(
                    stream_id="default",
                    source_columns=("acc_x", "acc_y", "acc_z"),
                    placement_label="lateral_right_lower",
                    unit_scale_to_ms2=MS2_PER_G,
                ),
            ),
            activity_map={
                "walking": "walk",
                "walking upstairs": "stairs",
                "walking downstairs": "stairs",
                "sitting": "sit",
                "standing": "stand",
                "laying": "lay",
                "stand to sit": "transitions",
                "sit to stand": "transitions",
                "sit to lie": "transitions",
                "lie to sit": "transitions",
                "stand to lie": "transitions",
                "lie to stand": "transitions",
            },
            holdout_kind="numeric_list",
            holdout_values=("4", "9", "14", "19", "24", "29"),
        ),
        "wisdm": DatasetSpec(
            dataset_id="wisdm",
            dataset_enum="WISDM",
            source_kind="cached_whar",
            source_root=cache_dir / "wisdm",
            streams=(
                StreamSpec(
                    stream_id="default",
                    source_columns=("accel_x", "accel_y", "accel_z"),
                    placement_label="front_pocket",
                    unit_scale_to_ms2=WISDM_MS2_PER_UNIT,
                ),
            ),
            activity_map={
                "walking": "walk",
                "jogging": "run",
                "upstairs": "stairs",
                "downstairs": "stairs",
                "sitting": "sit",
                "standing": "stand",
            },
            holdout_kind="numeric_list",
            holdout_values=("4", "9", "14", "19", "24", "29", "34"),
        ),
        "motion_sense": DatasetSpec(
            dataset_id="motion_sense",
            dataset_enum="MOTION_SENSE",
            source_kind="cached_whar",
            source_root=cache_dir / "motion_sense",
            streams=(
                StreamSpec(
                    stream_id="default",
                    source_columns=(
                        "userAcceleration.x",
                        "userAcceleration.y",
                        "userAcceleration.z",
                    ),
                    placement_label="front_pocket",
                    unit_scale_to_ms2=MS2_PER_G,
                ),
            ),
            activity_map={
                "walking": "walk",
                "jogging": "run",
                "upstairs": "stairs",
                "downstairs": "stairs",
                "sitting": "sit",
                "standing": "stand",
            },
            holdout_kind="numeric_list",
            holdout_values=("4", "9", "14", "19"),
        ),
        "hhar": DatasetSpec(
            dataset_id="hhar",
            dataset_enum="HHAR",
            source_kind="cached_whar",
            source_root=cache_dir / "hhar",
            streams=(
                StreamSpec(
                    stream_id="default",
                    source_columns=("accel_x", "accel_y", "accel_z"),
                    placement_label="unknown_free_living",
                    unit_scale_to_ms2=1.0,
                ),
            ),
            activity_map={
                "walk": "walk",
                "stairsup": "stairs",
                "stairsdown": "stairs",
                "sit": "sit",
                "stand": "stand",
                "bike": "locomotion-other",
            },
            holdout_kind="hhar_alpha",
            holdout_values=("b", "g"),
        ),
        "real_world": DatasetSpec(
            dataset_id="real_world",
            dataset_enum="REAL_WORLD",
            source_kind="realworld_local",
            source_root=realworld_dir,
            streams=(
                StreamSpec(
                    stream_id="chest",
                    source_columns=("x", "y", "z"),
                    placement_label="chest",
                    unit_scale_to_ms2=1.0,
                ),
                StreamSpec(
                    stream_id="waist",
                    source_columns=("x", "y", "z"),
                    placement_label="front_center_mid",
                    unit_scale_to_ms2=1.0,
                ),
            ),
            activity_map={
                "walking": "walk",
                "running": "run",
                "climbingup": "stairs",
                "climbingdown": "stairs",
                "sitting": "sit",
                "standing": "stand",
                "lying": "lay",
                "jumping": None,
            },
            holdout_kind="numeric_exact",
            holdout_values=("3", "8", "13"),
        ),
        "iphone_sweep": DatasetSpec(
            dataset_id="iphone_sweep",
            dataset_enum="IPHONE_SWEEP",
            source_kind="iphone_local",
            source_root=iphone_dir,
            streams=(
                StreamSpec(
                    stream_id="default",
                    source_columns=("x", "y", "z"),
                    placement_label="",
                    unit_scale_to_ms2=MS2_PER_G,
                ),
            ),
            activity_map={
                "standingstill": "stand",
                "sitting": "sit",
                "laying": "lay",
                "walking": "walk",
                "running": "run",
                "sittingstanding": "transitions",
            },
            holdout_kind="none",
        ),
        "pamap2": DatasetSpec(
            dataset_id="pamap2",
            dataset_enum="PAMAP2",
            source_kind="cached_whar",
            source_root=cache_dir / "pamap2",
            streams=(
                StreamSpec(
                    stream_id="chest",
                    source_columns=("chest_acc_x", "chest_acc_y", "chest_acc_z"),
                    placement_label="chest",
                    unit_scale_to_ms2=1.0,
                ),
            ),
            activity_map={
                "lying": "lay",
                "sitting": "sit",
                "standing": "stand",
                "walking": "walk",
                "running": "run",
                "ascending stairs": "stairs",
                "descending stairs": "stairs",
                "cycling": "locomotion-other",
                "other": None,
                "ironing": None,
                "vacuum cleaning": None,
                "nordic walking": "walk",
                "rope jumping": None,
            },
            holdout_kind="numeric_list",
            holdout_values=("2", "7"),
        ),
        "mhealth": DatasetSpec(
            dataset_id="mhealth",
            dataset_enum="MHEALTH",
            source_kind="cached_whar",
            source_root=cache_dir / "mhealth",
            streams=(
                StreamSpec(
                    stream_id="chest",
                    source_columns=("chest_acc_x", "chest_acc_y", "chest_acc_z"),
                    placement_label="chest",
                    unit_scale_to_ms2=1.0,
                ),
            ),
            activity_map={
                "standing still": "stand",
                "sitting and relaxing": "sit",
                "lying down": "lay",
                "walking": "walk",
                "jogging": "run",
                "running": "run",
                "climbing stairs": "stairs",
                "cycling": "locomotion-other",
                "unknown": None,
                "waist bends forward": None,
                "frontal elevation of arms": None,
                "knees bending crouching": None,
                "jump front and back": None,
            },
            holdout_kind="numeric_list",
            holdout_values=("2", "7", "10"),
        ),
        "sad": DatasetSpec(
            dataset_id="sad",
            dataset_enum="SAD",
            source_kind="cached_whar",
            source_root=cache_dir / "sad",
            streams=(
                StreamSpec(
                    stream_id="belt",
                    source_columns=("Belt_Ax", "Belt_Ay", "Belt_Az"),
                    placement_label="lateral_right_lower",
                    unit_scale_to_ms2=1.0,
                ),
            ),
            activity_map={
                "walking": "walk",
                "jogging": "run",
                "sitting": "sit",
                "standing": "stand",
                "upstairs": "stairs",
                "downstairs": "stairs",
                "biking": "locomotion-other",
            },
            holdout_kind="numeric_list",
            holdout_values=("2", "7", "10"),
        ),
        "uma_fall": DatasetSpec(
            dataset_id="uma_fall",
            dataset_enum="UMA_FALL",
            source_kind="cached_whar",
            source_root=cache_dir / "uma_fall",
            streams=(
                StreamSpec(
                    stream_id="chest",
                    source_columns=("chest_acc_x", "chest_acc_y", "chest_acc_z"),
                    placement_label="chest",
                    unit_scale_to_ms2=MS2_PER_G,
                ),
                StreamSpec(
                    stream_id="waist",
                    source_columns=("waist_acc_x", "waist_acc_y", "waist_acc_z"),
                    placement_label="front_center_lower",
                    unit_scale_to_ms2=MS2_PER_G,
                ),
                StreamSpec(
                    stream_id="right_pocket",
                    source_columns=(
                        "right_pocket_phone_acc_x",
                        "right_pocket_phone_acc_y",
                        "right_pocket_phone_acc_z",
                    ),
                    placement_label="front_pocket",
                    unit_scale_to_ms2=MS2_PER_G,
                ),
            ),
            activity_map={
                "walking": "walk",
                "jogging": "run",
                "go upstairs": "stairs",
                "go downstairs": "stairs",
                "lying down on abed": "lay",
                "sitting getting up on achair": "transitions",
                "bending": None,
                "hands up": None,
                "hopping": None,
                "making acall": None,
                "opening door": None,
                "aplausing": None,
                "forward fall": None,
                "backward fall": None,
                "lateral fall": None,
            },
            holdout_kind="every_fifth_numeric",
        ),
    }


def resolve_selected_specs(
    all_specs: dict[str, DatasetSpec],
    raw_names: Sequence[str],
) -> list[DatasetSpec]:
    """Resolve dataset names from CLI input."""

    if not raw_names or (len(raw_names) == 1 and raw_names[0].upper() == "ALL"):
        ordered_keys = (
            "hapt",
            "wisdm",
            "motion_sense",
            "hhar",
            "real_world",
            "iphone_sweep",
            "pamap2",
            "mhealth",
            "sad",
            "uma_fall",
        )
        return [all_specs[key] for key in ordered_keys]

    enum_lookup = {spec.dataset_enum.lower(): spec for spec in all_specs.values()}
    selected: list[DatasetSpec] = []
    for raw_name in raw_names:
        normalized = raw_name.strip().lower()
        spec = all_specs.get(normalized) or enum_lookup.get(normalized)
        if spec is None:
            valid = ", ".join(sorted({*all_specs.keys(), *enum_lookup.keys()}))
            raise ValueError(f"Unknown dataset '{raw_name}'. Valid values: {valid}")
        selected.append(spec)
    return selected


def clean_output_dir(output_dir: Path) -> None:
    """Delete previously written artifacts under the output directory."""

    if not output_dir.exists():
        return

    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def ensure_exists(path: Path, label: str) -> None:
    """Raise a clear error when a required path is missing."""

    if not path.exists():
        raise ValueError(f"Missing {label}: {path}")


def read_small_csv(path: Path) -> list[dict[str, str]]:
    """Read a small metadata CSV without pandas."""

    ensure_exists(path, "metadata CSV")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            cleaned = {
                key: value
                for key, value in row.items()
                if key and key.strip() and not key.startswith("Unnamed:")
            }
            rows.append(cleaned)
        return rows


def normalize_activity_name(value: str) -> str:
    """Normalize raw activity strings for stable mapping."""

    return " ".join(value.strip().lower().split())


def resolve_activity_label(activity_map: dict[str, str | None], raw_activity_name: str) -> str | None:
    """Map a raw activity name into the canonical contract label."""

    normalized = normalize_activity_name(raw_activity_name)
    return activity_map.get(normalized)


def resolve_holdout_subjects(spec: DatasetSpec, observed_subject_ids: set[str]) -> set[str]:
    """Resolve held-out subjects against the actual observed subject ids."""

    if spec.holdout_kind == "none":
        return set()

    if spec.holdout_kind == "hhar_alpha":
        if all(value.lstrip("-").isdigit() for value in observed_subject_ids):
            letter_to_zero_based = {letter: str(index) for index, letter in enumerate("abcdefghi")}
            return {letter_to_zero_based[value] for value in spec.holdout_values}
        return set(spec.holdout_values)

    numeric_subjects = {int(value) for value in observed_subject_ids}

    if spec.holdout_kind == "every_fifth_numeric":
        if 0 in numeric_subjects:
            return {str(value) for value in numeric_subjects if (value + 1) % 5 == 0}
        return {str(value) for value in numeric_subjects if value % 5 == 0}

    requested = {int(value) for value in spec.holdout_values}

    if spec.holdout_kind == "numeric_exact":
        return {str(value) for value in requested}

    if spec.holdout_kind == "numeric_list":
        if requested.issubset(numeric_subjects):
            return {str(value) for value in requested}

        zero_based_requested = {value - 1 for value in requested}
        if zero_based_requested.issubset(numeric_subjects):
            return {str(value) for value in zero_based_requested}

        raise ValueError(
            f"Unable to reconcile holdout subjects for {spec.dataset_id}. "
            f"Observed={sorted(numeric_subjects)} Requested={sorted(requested)}"
        )

    raise ValueError(f"Unsupported holdout rule: {spec.holdout_kind}")


def build_cached_session_lookup(spec: DatasetSpec) -> dict[int, tuple[str, str, str]]:
    """Build session_id -> (subject_id, raw_activity_name, split) metadata."""

    activity_rows = read_small_csv(spec.source_root / "metadata" / "activity_df.csv")
    session_rows = read_small_csv(spec.source_root / "metadata" / "session_df.csv")

    activity_lookup = {
        int(row["activity_id"]): row["activity_name"]
        for row in activity_rows
    }

    observed_subject_ids = {row["subject_id"].strip() for row in session_rows}
    holdout_subjects = resolve_holdout_subjects(spec, observed_subject_ids)

    session_lookup: dict[int, tuple[str, str, str]] = {}
    for row in session_rows:
        subject_id = row["subject_id"].strip()
        raw_activity_name = activity_lookup[int(row["activity_id"])]
        split = "test" if subject_id in holdout_subjects else "train"
        session_lookup[int(row["session_id"])] = (subject_id, raw_activity_name, split)

    return session_lookup


def sample_table_schema() -> pa.Schema:
    """Return the canonical sample parquet schema."""

    return pa.schema(
        [
            ("dataset_enum", pa.string()),
            ("dataset_id", pa.string()),
            ("stream_id", pa.string()),
            ("split", pa.string()),
            ("global_subject_id", pa.string()),
            ("subject_local_id", pa.string()),
            ("global_session_id", pa.string()),
            ("source_session_id", pa.string()),
            ("global_window_id", pa.string()),
            ("window_index", pa.int32()),
            ("sample_index", pa.int16()),
            ("time_s", pa.float32()),
            ("activity_label", pa.string()),
            ("activity_name_raw", pa.string()),
            ("is_report_activity", pa.bool_()),
            ("placement_label", pa.string()),
            ("sample_rate_hz", pa.int16()),
            ("acc_x", pa.float32()),
            ("acc_y", pa.float32()),
            ("acc_z", pa.float32()),
        ]
    )


class WriterRegistry:
    """Manage split/dataset/stream parquet writers."""

    def __init__(self, output_dir: Path, schema: pa.Schema):
        self._output_dir = output_dir
        self._schema = schema
        self._writers: dict[tuple[str, str, str], pq.ParquetWriter] = {}

    def write(self, split: str, dataset_id: str, stream_id: str, table: pa.Table) -> None:
        """Write a table chunk to the correct partition file."""

        key = (split, dataset_id, stream_id)
        writer = self._writers.get(key)
        if writer is None:
            file_path = (
                self._output_dir
                / "samples"
                / f"split={split}"
                / f"dataset_id={dataset_id}"
                / f"stream_id={stream_id}"
                / "part-0.parquet"
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(file_path, self._schema, compression="zstd")
            self._writers[key] = writer

        writer.write_table(table)

    def close(self) -> None:
        """Close all open parquet writers."""

        for writer in self._writers.values():
            writer.close()


def sanitize_numeric_timeseries(
    time_seconds: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort, de-duplicate, and remove invalid numeric rows."""

    segment = shared_sanitize_numeric_timeseries(
        time_seconds=time_seconds,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )
    return segment.time_seconds, segment.acc_x, segment.acc_y, segment.acc_z


def sanitize_timestamp_timeseries(
    timestamps: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert timestamp values to elapsed seconds and sanitize the series."""

    segment = shared_sanitize_timestamp_timeseries(
        timestamps=timestamps,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )
    return segment.time_seconds, segment.acc_x, segment.acc_y, segment.acc_z


def resample_to_contract(
    time_seconds: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample one accelerometer stream onto the canonical 50 Hz grid."""

    segment = resample_accelerometer_segment(
        AccelerometerSegment(
            time_seconds=np.asarray(time_seconds, dtype=np.float64),
            acc_x=np.asarray(x_values, dtype=np.float64),
            acc_y=np.asarray(y_values, dtype=np.float64),
            acc_z=np.asarray(z_values, dtype=np.float64),
        ),
        target_sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
    )
    return segment.time_seconds, segment.acc_x, segment.acc_y, segment.acc_z


def _build_segment_source_session_id(
    base_source_session_id: str,
    segment_index: int,
    segment_count: int,
) -> str:
    """Keep session ids stable unless gap splitting creates multiple segments."""

    if segment_count <= 1:
        return base_source_session_id
    return f"{base_source_session_id}:segment:{segment_index}"


def build_window_starts(num_samples: int) -> range:
    """Return the canonical sliding-window start indices."""

    if num_samples < WINDOW_SAMPLES:
        return range(0)
    return range(0, num_samples - WINDOW_SAMPLES + 1, WINDOW_STEP_SAMPLES)


def build_session_sample_table(
    session_meta: SessionMeta,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    window_manifest_rows: list[dict[str, object]],
    session_manifest_rows: list[dict[str, object]],
) -> tuple[str, pa.Table] | None:
    """Convert one resampled session into a parquet-ready sample table."""

    window_starts = list(build_window_starts(x_values.size))
    if not window_starts:
        return None

    num_windows = len(window_starts)
    total_rows = num_windows * WINDOW_SAMPLES
    acc_x = np.empty(total_rows, dtype=np.float32)
    acc_y = np.empty(total_rows, dtype=np.float32)
    acc_z = np.empty(total_rows, dtype=np.float32)
    sample_index = np.tile(np.arange(WINDOW_SAMPLES, dtype=np.int16), num_windows)
    time_seconds = sample_index.astype(np.float32) / TARGET_SAMPLE_RATE_HZ

    dataset_enum = [session_meta.dataset_enum] * total_rows
    dataset_id = [session_meta.dataset_id] * total_rows
    stream_id = [session_meta.stream_id] * total_rows
    split = [session_meta.split] * total_rows
    global_subject_id = [session_meta.global_subject_id] * total_rows
    subject_local_id = [session_meta.subject_local_id] * total_rows
    global_session_id = [session_meta.global_session_id] * total_rows
    source_session_id = [session_meta.source_session_id] * total_rows
    activity_label = [session_meta.activity_label] * total_rows
    activity_name_raw = [session_meta.activity_name_raw] * total_rows
    is_report_activity = [session_meta.activity_label in CORE_REPORT_LABELS] * total_rows
    placement_label = [session_meta.placement_label] * total_rows
    sample_rate_hz = [TARGET_SAMPLE_RATE_HZ] * total_rows
    global_window_ids: list[str] = []
    window_indices: list[int] = []

    for window_index, start in enumerate(window_starts):
        row_start = window_index * WINDOW_SAMPLES
        row_end = row_start + WINDOW_SAMPLES
        acc_x[row_start:row_end] = x_values[start : start + WINDOW_SAMPLES]
        acc_y[row_start:row_end] = y_values[start : start + WINDOW_SAMPLES]
        acc_z[row_start:row_end] = z_values[start : start + WINDOW_SAMPLES]

        global_window_id = f"{session_meta.global_session_id}:window:{window_index}"
        global_window_ids.extend([global_window_id] * WINDOW_SAMPLES)
        window_indices.extend([window_index] * WINDOW_SAMPLES)

        window_manifest_rows.append(
            {
                "dataset_enum": session_meta.dataset_enum,
                "dataset_id": session_meta.dataset_id,
                "stream_id": session_meta.stream_id,
                "split": session_meta.split,
                "global_subject_id": session_meta.global_subject_id,
                "subject_local_id": session_meta.subject_local_id,
                "global_session_id": session_meta.global_session_id,
                "source_session_id": session_meta.source_session_id,
                "global_window_id": global_window_id,
                "window_index": window_index,
                "activity_label": session_meta.activity_label,
                "activity_name_raw": session_meta.activity_name_raw,
                "is_report_activity": session_meta.activity_label in CORE_REPORT_LABELS,
                "placement_label": session_meta.placement_label,
                "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
                "window_samples": WINDOW_SAMPLES,
            }
        )

    session_manifest_rows.append(
        {
            "dataset_enum": session_meta.dataset_enum,
            "dataset_id": session_meta.dataset_id,
            "stream_id": session_meta.stream_id,
            "split": session_meta.split,
            "global_subject_id": session_meta.global_subject_id,
            "subject_local_id": session_meta.subject_local_id,
            "global_session_id": session_meta.global_session_id,
            "source_session_id": session_meta.source_session_id,
            "activity_label": session_meta.activity_label,
            "activity_name_raw": session_meta.activity_name_raw,
            "is_report_activity": session_meta.activity_label in CORE_REPORT_LABELS,
            "placement_label": session_meta.placement_label,
            "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "window_count": num_windows,
        }
    )

    table = pa.table(
        {
            "dataset_enum": dataset_enum,
            "dataset_id": dataset_id,
            "stream_id": stream_id,
            "split": split,
            "global_subject_id": global_subject_id,
            "subject_local_id": subject_local_id,
            "global_session_id": global_session_id,
            "source_session_id": source_session_id,
            "global_window_id": global_window_ids,
            "window_index": window_indices,
            "sample_index": sample_index,
            "time_s": time_seconds,
            "activity_label": activity_label,
            "activity_name_raw": activity_name_raw,
            "is_report_activity": is_report_activity,
            "placement_label": placement_label,
            "sample_rate_hz": sample_rate_hz,
            "acc_x": acc_x,
            "acc_y": acc_y,
            "acc_z": acc_z,
        },
        schema=sample_table_schema(),
    )
    return session_meta.split, table


def process_resampled_session(
    session_meta: SessionMeta,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    writer_registry: WriterRegistry,
    window_manifest_rows: list[dict[str, object]],
    session_manifest_rows: list[dict[str, object]],
) -> int:
    """Write one already-resampled canonical session into the unified dataset."""

    built = build_session_sample_table(
        session_meta=session_meta,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        window_manifest_rows=window_manifest_rows,
        session_manifest_rows=session_manifest_rows,
    )
    if built is None:
        return 0

    split, table = built
    writer_registry.write(split, session_meta.dataset_id, session_meta.stream_id, table)
    return table.num_rows


def process_cached_dataset(
    spec: DatasetSpec,
    writer_registry: WriterRegistry,
    window_manifest_rows: list[dict[str, object]],
    session_manifest_rows: list[dict[str, object]],
) -> None:
    """Build canonical windows from cached session parquet."""

    sessions_parquet = spec.source_root / "sessions" / "sessions.parquet"
    ensure_exists(sessions_parquet, "cached session parquet")

    session_lookup = build_cached_session_lookup(spec)
    column_names = sorted(
        {
            "timestamp",
            "session_id",
            *(column for stream in spec.streams for column in stream.source_columns),
        }
    )

    parquet_file = pq.ParquetFile(sessions_parquet)
    current_session_id: int | None = None
    current_chunks: dict[str, list[np.ndarray]] = defaultdict(list)

    def flush_current_session() -> None:
        nonlocal current_session_id
        if current_session_id is None:
            return

        subject_local_id, raw_activity_name, split = session_lookup[current_session_id]
        activity_label = resolve_activity_label(spec.activity_map, raw_activity_name)
        if activity_label is None:
            current_session_id = None
            current_chunks.clear()
            return

        timestamps = np.concatenate(current_chunks["timestamp"])
        for stream in spec.streams:
            x_values = np.concatenate(current_chunks[stream.source_columns[0]]) * stream.unit_scale_to_ms2
            y_values = np.concatenate(current_chunks[stream.source_columns[1]]) * stream.unit_scale_to_ms2
            z_values = np.concatenate(current_chunks[stream.source_columns[2]]) * stream.unit_scale_to_ms2
            preprocessed_segments = preprocess_timestamp_segments(
                timestamps=timestamps,
                x_values=x_values,
                y_values=y_values,
                z_values=z_values,
            )
            segment_count = len(preprocessed_segments)
            for preprocessed in preprocessed_segments:
                if preprocessed.total.time_seconds.size < 2:
                    continue

                session_meta = SessionMeta(
                    dataset_id=spec.dataset_id,
                    dataset_enum=spec.dataset_enum,
                    stream_id=stream.stream_id,
                    subject_local_id=subject_local_id,
                    source_session_id=_build_segment_source_session_id(
                        str(current_session_id),
                        preprocessed.segment_index,
                        segment_count,
                    ),
                    activity_name_raw=raw_activity_name,
                    activity_label=activity_label,
                    placement_label=stream.placement_label,
                    split=split,
                )
                process_resampled_session(
                    session_meta=session_meta,
                    x_values=preprocessed.total.acc_x.astype(np.float32, copy=False),
                    y_values=preprocessed.total.acc_y.astype(np.float32, copy=False),
                    z_values=preprocessed.total.acc_z.astype(np.float32, copy=False),
                    writer_registry=writer_registry,
                    window_manifest_rows=window_manifest_rows,
                    session_manifest_rows=session_manifest_rows,
                )

        current_session_id = None
        current_chunks.clear()

    for batch in parquet_file.iter_batches(columns=column_names, batch_size=65_536):
        session_ids = np.asarray(batch.column(batch.schema.get_field_index("session_id")))
        arrays = {
            name: np.asarray(batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False))
            for name in column_names
            if name != "session_id"
        }

        start = 0
        while start < len(session_ids):
            session_id = int(session_ids[start])
            end = start + 1
            while end < len(session_ids) and int(session_ids[end]) == session_id:
                end += 1

            if current_session_id is None:
                current_session_id = session_id
            elif session_id != current_session_id:
                flush_current_session()
                current_session_id = session_id

            for name, values in arrays.items():
                current_chunks[name].append(values[start:end])

            start = end

    flush_current_session()


def parse_iphone_folder_name(folder_name: str) -> dict[str, str] | None:
    """Extract placement and activity labels from one iPhone folder name."""

    stem = folder_name
    if len(folder_name) > 20 and folder_name[-20] == "-":
        stem = folder_name[:-20]

    activity = None
    placement = None
    for candidate in IPHONE_ACTIVITIES:
        token = f"-{candidate}"
        if stem.endswith(token):
            activity = candidate
            placement = stem[: -len(token)]
            break

    if activity is None or not placement:
        return None

    placement = IPHONE_PLACEMENT_ALIASES.get(placement, placement)
    return {
        "folder_name": folder_name,
        "activity": activity,
        "placement": placement,
    }


def process_iphone_sweep(
    spec: DatasetSpec,
    writer_registry: WriterRegistry,
    window_manifest_rows: list[dict[str, object]],
    session_manifest_rows: list[dict[str, object]],
) -> None:
    """Build canonical windows from the local iPhone placement sweep."""

    ensure_exists(spec.source_root, "iPhone sweep directory")
    for folder_path in sorted(path for path in spec.source_root.iterdir() if path.is_dir()):
        parsed = parse_iphone_folder_name(folder_path.name)
        if parsed is None:
            continue

        activity_label = resolve_activity_label(spec.activity_map, parsed["activity"])
        if activity_label is None:
            continue

        csv_path = folder_path / "Accelerometer.csv"
        ensure_exists(csv_path, "iPhone accelerometer CSV")
        table = pa_csv.read_csv(csv_path)
        column_names = set(table.column_names)
        required = {"seconds_elapsed", "x", "y", "z"}
        if not required.issubset(column_names):
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(column_names)}")

        time_seconds = table.column("seconds_elapsed").to_numpy(zero_copy_only=False).astype(np.float64)
        x_values = table.column("x").to_numpy(zero_copy_only=False).astype(np.float64) * MS2_PER_G
        y_values = table.column("y").to_numpy(zero_copy_only=False).astype(np.float64) * MS2_PER_G
        z_values = table.column("z").to_numpy(zero_copy_only=False).astype(np.float64) * MS2_PER_G

        time_seconds, x_values, y_values, z_values = sanitize_numeric_timeseries(
            time_seconds=time_seconds,
            x_values=x_values,
            y_values=y_values,
            z_values=z_values,
        )
        if time_seconds.size < 2:
            continue

        trim_start = 5.0
        trim_end = 15.0
        trimmed_mask = (
            (time_seconds >= trim_start)
            & (time_seconds <= (time_seconds[-1] - trim_end))
        )
        if np.count_nonzero(trimmed_mask) >= WINDOW_SAMPLES:
            trimmed_time = time_seconds[trimmed_mask]
            time_seconds = trimmed_time - trimmed_time[0]
            x_values = x_values[trimmed_mask]
            y_values = y_values[trimmed_mask]
            z_values = z_values[trimmed_mask]

        preprocessed_segments = preprocess_numeric_segments(
            time_seconds=time_seconds,
            x_values=x_values,
            y_values=y_values,
            z_values=z_values,
        )
        segment_count = len(preprocessed_segments)
        for preprocessed in preprocessed_segments:
            if preprocessed.total.time_seconds.size < 2:
                continue

            session_meta = SessionMeta(
                dataset_id=spec.dataset_id,
                dataset_enum=spec.dataset_enum,
                stream_id=parsed["placement"],
                subject_local_id="1",
                source_session_id=_build_segment_source_session_id(
                    parsed["folder_name"],
                    preprocessed.segment_index,
                    segment_count,
                ),
                activity_name_raw=parsed["activity"],
                activity_label=activity_label,
                placement_label=IPHONE_PLACEMENT_TO_MODEL[parsed["placement"]],
                split="train_only",
            )
            process_resampled_session(
                session_meta=session_meta,
                x_values=preprocessed.total.acc_x.astype(np.float32, copy=False),
                y_values=preprocessed.total.acc_y.astype(np.float32, copy=False),
                z_values=preprocessed.total.acc_z.astype(np.float32, copy=False),
                writer_registry=writer_registry,
                window_manifest_rows=window_manifest_rows,
                session_manifest_rows=session_manifest_rows,
            )


def process_realworld_local(
    spec: DatasetSpec,
    writer_registry: WriterRegistry,
    window_manifest_rows: list[dict[str, object]],
    session_manifest_rows: list[dict[str, object]],
) -> None:
    """Build canonical windows from the local RealWorld2016 CSV files."""

    ensure_exists(spec.source_root, "RealWorld2016 directory")
    holdout_subjects = resolve_holdout_subjects(spec, {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"})

    for proband_dir in sorted(path for path in spec.source_root.iterdir() if path.is_dir() and path.name.startswith("proband")):
        subject_local_id = proband_dir.name.removeprefix("proband")
        data_dir = proband_dir / "data"
        ensure_exists(data_dir, f"RealWorld2016 data directory for {proband_dir.name}")

        for stream in spec.streams:
            position_name = stream.stream_id
            for csv_path in sorted(data_dir.rglob(f"acc_*_{position_name}.csv")):
                file_name = csv_path.stem
                raw_activity_name = file_name.removeprefix("acc_")
                raw_activity_name = raw_activity_name[: -(len(position_name) + 1)]
                activity_label = resolve_activity_label(spec.activity_map, raw_activity_name)
                if activity_label is None:
                    continue

                convert_options = pa_csv.ConvertOptions(
                    column_types={
                        "id": pa.int64(),
                        "attr_time": pa.float64(),
                        "attr_x": pa.float64(),
                        "attr_y": pa.float64(),
                        "attr_z": pa.float64(),
                    }
                )
                table = pa_csv.read_csv(csv_path, convert_options=convert_options)
                if "attr_time" not in table.column_names:
                    table = pa_csv.read_csv(
                        csv_path,
                        read_options=pa_csv.ReadOptions(skip_rows=1, autogenerate_column_names=False),
                        convert_options=convert_options,
                    )

                preprocessed_segments = preprocess_numeric_segments(
                    time_seconds=table.column("attr_time").to_numpy(zero_copy_only=False).astype(np.float64) / 1000.0,
                    x_values=table.column("attr_x").to_numpy(zero_copy_only=False).astype(np.float64),
                    y_values=table.column("attr_y").to_numpy(zero_copy_only=False).astype(np.float64),
                    z_values=table.column("attr_z").to_numpy(zero_copy_only=False).astype(np.float64),
                )
                split = "test" if subject_local_id in holdout_subjects else "train"
                segment_count = len(preprocessed_segments)
                base_source_session_id = f"{proband_dir.name}:{raw_activity_name}:{position_name}"
                for preprocessed in preprocessed_segments:
                    if preprocessed.total.time_seconds.size < 2:
                        continue

                    session_meta = SessionMeta(
                        dataset_id=spec.dataset_id,
                        dataset_enum=spec.dataset_enum,
                        stream_id=stream.stream_id,
                        subject_local_id=subject_local_id,
                        source_session_id=_build_segment_source_session_id(
                            base_source_session_id,
                            preprocessed.segment_index,
                            segment_count,
                        ),
                        activity_name_raw=raw_activity_name,
                        activity_label=activity_label,
                        placement_label=stream.placement_label,
                        split=split,
                    )
                    process_resampled_session(
                        session_meta=session_meta,
                        x_values=preprocessed.total.acc_x.astype(np.float32, copy=False),
                        y_values=preprocessed.total.acc_y.astype(np.float32, copy=False),
                        z_values=preprocessed.total.acc_z.astype(np.float32, copy=False),
                        writer_registry=writer_registry,
                        window_manifest_rows=window_manifest_rows,
                        session_manifest_rows=session_manifest_rows,
                    )


def write_manifest_parquet(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write a small metadata manifest to parquet."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path, compression="zstd")


def write_summary_csv(
    window_manifest_rows: list[dict[str, object]],
    session_manifest_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write a compact per-dataset summary for inspection."""

    session_counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    window_counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for row in session_manifest_rows:
        key = (
            str(row["dataset_id"]),
            str(row["stream_id"]),
            str(row["split"]),
            str(row["activity_label"]),
            str(row["placement_label"]),
        )
        session_counts[key] += 1
    for row in window_manifest_rows:
        key = (
            str(row["dataset_id"]),
            str(row["stream_id"]),
            str(row["split"]),
            str(row["activity_label"]),
            str(row["placement_label"]),
        )
        window_counts[key] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_id",
                "stream_id",
                "split",
                "activity_label",
                "placement_label",
                "sessions",
                "windows",
                "samples",
            ]
        )
        for key in sorted(window_counts):
            dataset_id, stream_id, split, activity_label, placement_label = key
            windows = window_counts[key]
            sessions = session_counts[key]
            writer.writerow(
                [
                    dataset_id,
                    stream_id,
                    split,
                    activity_label,
                    placement_label,
                    sessions,
                    windows,
                    windows * WINDOW_SAMPLES,
                ]
            )


def run_contract_build(selected_specs: Sequence[DatasetSpec], output_dir: Path) -> None:
    """Run the unified dataset build."""

    writer_registry = WriterRegistry(output_dir=output_dir, schema=sample_table_schema())
    window_manifest_rows: list[dict[str, object]] = []
    session_manifest_rows: list[dict[str, object]] = []
    try:
        for spec in selected_specs:
            LOGGER.info("Processing %s", spec.dataset_enum)
            if spec.source_kind == "cached_whar":
                process_cached_dataset(
                    spec=spec,
                    writer_registry=writer_registry,
                    window_manifest_rows=window_manifest_rows,
                    session_manifest_rows=session_manifest_rows,
                )
            elif spec.source_kind == "realworld_local":
                process_realworld_local(
                    spec=spec,
                    writer_registry=writer_registry,
                    window_manifest_rows=window_manifest_rows,
                    session_manifest_rows=session_manifest_rows,
                )
            elif spec.source_kind == "iphone_local":
                process_iphone_sweep(
                    spec=spec,
                    writer_registry=writer_registry,
                    window_manifest_rows=window_manifest_rows,
                    session_manifest_rows=session_manifest_rows,
                )
            else:
                raise ValueError(f"Unsupported source kind: {spec.source_kind}")
    finally:
        writer_registry.close()

    write_manifest_parquet(window_manifest_rows, output_dir / "window_manifest.parquet")
    write_manifest_parquet(session_manifest_rows, output_dir / "session_manifest.parquet")
    write_summary_csv(
        window_manifest_rows=window_manifest_rows,
        session_manifest_rows=session_manifest_rows,
        output_path=output_dir / "dataset_summary.csv",
    )


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    configure_logging(args.verbose)

    if args.overwrite:
        clean_output_dir(args.output_dir)

    specs = build_dataset_specs(
        cache_dir=args.cache_dir,
        realworld_dir=args.realworld_dir,
        iphone_dir=args.iphone_dir,
    )
    selected_specs = resolve_selected_specs(specs, args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_contract_build(selected_specs=selected_specs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()