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
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from offline.processing.preprocessing import (
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
DEFAULT_CACHE_DIR = Path("data") / "whar_datasets_cache"
DEFAULT_REALWORLD_DIR = Path("data") / "realworld2016_dataset"
DEFAULT_IPHONE_DIR = Path("data") / "Acceleration"
DEFAULT_UNIMIB_DIR = Path("data") / "UniMiB-SHAR" / "UniMiB-SHAR"
DEFAULT_WISDM_V2_DIR = Path("data") / "wisdm-dataset" / "wisdm-dataset"
DEFAULT_SHOAIB2013_DIR = Path("data") / "activity-recognition-dataset-shoaib" / "Activity_Recognition_DataSet"
DEFAULT_SHOAIB_SENSORS_DIR = Path("data") / "sensors-activity-recognition-dataset-shoaib" / "DataSet"
DEFAULT_UT_COMPLEX_DIR = Path("data") / "ut-data-complex" / "UT_Data_Complex"

CORE_REPORT_LABELS = {"walk", "run", "stairs", "sit/lay", "stand"}

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
        "--unimib-dir",
        type=Path,
        default=DEFAULT_UNIMIB_DIR,
        help="Root directory for the local UniMiB-SHAR files.",
    )
    parser.add_argument(
        "--wisdm-v2-dir",
        type=Path,
        default=DEFAULT_WISDM_V2_DIR,
        help="Root directory for the local WISDM v2 raw phone accel files.",
    )
    parser.add_argument(
        "--shoaib2013-dir",
        type=Path,
        default=DEFAULT_SHOAIB2013_DIR,
        help="Root directory for the Shoaib 2013 pocket xlsx file.",
    )
    parser.add_argument(
        "--shoaib-sensors-dir",
        type=Path,
        default=DEFAULT_SHOAIB_SENSORS_DIR,
        help="Root directory for the Shoaib Sensors per-participant CSV files.",
    )
    parser.add_argument(
        "--ut-complex-dir",
        type=Path,
        default=DEFAULT_UT_COMPLEX_DIR,
        help="Root directory for the UT Complex smartphone-at-pocket CSV file.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["ALL"],
        metavar="DATASET",
        help=(
            "Subset of contract datasets to build. Accepts dataset ids or enums, "
            "for example WISDM REAL_WORLD IPHONE_SWEEP."
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
    unimib_dir: Path = DEFAULT_UNIMIB_DIR,
    wisdm_v2_dir: Path = DEFAULT_WISDM_V2_DIR,
    shoaib2013_dir: Path = DEFAULT_SHOAIB2013_DIR,
    shoaib_sensors_dir: Path = DEFAULT_SHOAIB_SENSORS_DIR,
    ut_complex_dir: Path = DEFAULT_UT_COMPLEX_DIR,
) -> dict[str, DatasetSpec]:
    """Build the approved contract dataset registry."""

    return {
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
                StreamSpec(
                    stream_id="thigh",
                    source_columns=("x", "y", "z"),
                    placement_label="front_pocket",
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
                    unit_scale_to_ms2=1.0,
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
        "unimib_shar": DatasetSpec(
            dataset_id="unimib_shar",
            dataset_enum="UNIMIB_SHAR",
            source_kind="unimib_local",
            source_root=unimib_dir,
            streams=(
                StreamSpec(
                    stream_id="front_pocket",
                    source_columns=("x", "y", "z"),
                    placement_label="front_pocket",
                    unit_scale_to_ms2=1.0,
                ),
            ),
            activity_map={
                "standingupfs": "transitions",
                "standingupfl": "transitions",
                "walking": "walk",
                "running": "run",
                "goingups": "stairs",
                "jumping": "locomotion-other",
                "goingdowns": "stairs",
                "lyingdownfs": "transitions",
                "sittingdown": "transitions",
                "fallingforw": None,
                "fallingright": None,
                "fallingback": None,
                "hittingobstacle": None,
                "fallingwithps": None,
                "fallingbacksc": None,
                "syncope": None,
                "fallingleft": None,
            },
            holdout_kind="numeric_list",
            holdout_values=("4", "9", "14", "19", "24", "29"),
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
        "wisdm_v2": DatasetSpec(
            dataset_id="wisdm_v2",
            dataset_enum="WISDM_V2",
            source_kind="wisdm_v2_local",
            source_root=wisdm_v2_dir,
            streams=(
                StreamSpec(
                    stream_id="default",
                    source_columns=("x", "y", "z"),
                    placement_label="front_pocket",
                    unit_scale_to_ms2=1.0,  # already m/s²
                ),
            ),
            activity_map={
                "a": "walk",          # walking
                "b": "run",           # jogging
                "c": "stairs",        # stairs
                "d": "sit",           # sitting
                "e": "stand",         # standing
                "f": None,            # typing
                "g": None,            # brushing teeth
                "h": None,            # eating soup
                "i": None,            # eating chips
                "j": None,            # eating pasta
                "k": None,            # drinking from cup
                "l": None,            # eating sandwich
                "m": "locomotion-other",  # kicking soccer ball
                "o": None,            # playing catch
                "p": "locomotion-other",  # dribbling basketball
                "q": None,            # writing
                "r": None,            # clapping
                "s": None,            # folding clothes
            },
            holdout_kind="numeric_list",
            holdout_values=(
                "1604", "1609", "1614", "1619", "1624",
                "1629", "1634", "1639", "1644", "1649",
            ),
        ),
        "shoaib_2013": DatasetSpec(
            dataset_id="shoaib_2013",
            dataset_enum="SHOAIB_2013",
            source_kind="shoaib2013_local",
            source_root=shoaib2013_dir,
            streams=(
                StreamSpec(
                    stream_id="pocket",
                    source_columns=("timestamp", "ax", "ay", "az"),
                    placement_label="front_pocket",
                    unit_scale_to_ms2=1.0,
                ),
            ),
            # Shoaib 2013 (UbiComp): 4 participants, right jeans pocket, Samsung Galaxy S2.
            # Sampled at 50 Hz. Activities: Walking, Running, Sitting, Standing,
            # Upstairs, Downstairs (Shoaib 2013, "Human Activity Recognition Using
            # Heterogeneous Sensors", UbiComp Adjunct).
            activity_map={
                "walking": "walk",
                "running": "run",
                "sitting": "sit",
                "standing": "stand",
                "upstairs": "stairs",
                "downstairs": "stairs",
            },
            holdout_kind="none",
            holdout_values=(),
        ),
        "shoaib_sensors": DatasetSpec(
            dataset_id="shoaib_sensors",
            dataset_enum="SHOAIB_SENSORS",
            source_kind="shoaib_sensors_local",
            source_root=shoaib_sensors_dir,
            streams=(
                StreamSpec(
                    stream_id="left_pocket",
                    source_columns=("left_ts", "left_ax", "left_ay", "left_az"),
                    placement_label="front_pocket",
                    unit_scale_to_ms2=1.0,
                ),
                StreamSpec(
                    stream_id="right_pocket",
                    source_columns=("right_ts", "right_ax", "right_ay", "right_az"),
                    placement_label="front_pocket",
                    unit_scale_to_ms2=1.0,
                ),
            ),
            # Shoaib et al. Sensors 2016: 10 participants, left+right jeans pocket,
            # Samsung Galaxy S2 at 50 Hz. Activities: walking, standing, jogging,
            # sitting, biking, upstairs, downstairs (Shoaib et al. 2016,
            # "Complex human activity recognition using smartphone and wrist-worn
            # motion sensors", Sensors 16(4)).
            activity_map={
                "walking": "walk",
                "jogging": "run",
                "sitting": "sit",
                "standing": "stand",
                "biking": "locomotion-other",
                "upstairs": "stairs",
                "downstairs": "stairs",
            },
            holdout_kind="numeric_list",
            holdout_values=("2", "5", "8"),
        ),
        "ut_complex": DatasetSpec(
            dataset_id="ut_complex",
            dataset_enum="UT_COMPLEX",
            source_kind="ut_complex_local",
            source_root=ut_complex_dir,
            streams=(
                StreamSpec(
                    stream_id="pocket",
                    source_columns=("timestamp", "ax", "ay", "az"),
                    placement_label="front_pocket",
                    unit_scale_to_ms2=1.0,
                ),
            ),
            # Shoaib et al. Sensors 2016 (complex version): smartphone at pocket,
            # 50 Hz. Activity codes 11111-11123. Only locomotion and posture codes
            # are mapped; fine-motor codes (type, write, coffee, etc.) are excluded
            # (Shoaib et al. 2016, "Complex human activity recognition", Sensors 16(4)).
            activity_map={
                "11111": "walk",
                "11112": "stand",
                "11113": "run",
                "11114": "sit",
                "11116": "stairs",
                "11117": "stairs",
                "11115": None,   # biking -- excluded (no bike label in our model)
                "11118": None,   # typing
                "11119": None,   # writing
                "11120": None,   # coffee
                "11121": None,   # talking
                "11122": None,   # smoking
                "11123": None,   # eating
            },
            holdout_kind="none",
            holdout_values=(),
        ),
    }


def resolve_selected_specs(
    all_specs: dict[str, DatasetSpec],
    raw_names: Sequence[str],
) -> list[DatasetSpec]:
    """Resolve dataset names from CLI input."""

    if not raw_names or (len(raw_names) == 1 and raw_names[0].upper() == "ALL"):
        ordered_keys = (
            "wisdm",
            "motion_sense",
            "real_world",
            "iphone_sweep",
            "unimib_shar",
            "uma_fall",
            "wisdm_v2",
            "shoaib_2013",
            "shoaib_sensors",
            "ut_complex",
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
        x_values = table.column("x").to_numpy(zero_copy_only=False).astype(np.float64)
        y_values = table.column("y").to_numpy(zero_copy_only=False).astype(np.float64)
        z_values = table.column("z").to_numpy(zero_copy_only=False).astype(np.float64)

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


def _resolve_unimib_root(source_root: Path) -> Path:
    """Resolve UniMiB root whether caller points to outer or inner directory."""

    direct_data_dir = source_root / "data"
    nested_data_dir = source_root / "UniMiB-SHAR" / "data"
    if direct_data_dir.exists():
        return source_root
    if nested_data_dir.exists():
        return source_root / "UniMiB-SHAR"
    raise ValueError(f"Missing UniMiB-SHAR data directory under {source_root}")


def _extract_unimib_activity_codes(unimib_data_dir: Path) -> dict[int, str]:
    """Read UniMiB activity short codes indexed from 1..17."""

    import scipy.io as sio

    names_mat = sio.loadmat(unimib_data_dir / "acc_names.mat")
    if "acc_names" not in names_mat:
        raise ValueError("acc_names.mat does not contain 'acc_names'")

    names_array = names_mat["acc_names"]
    if names_array.ndim != 2 or names_array.shape[0] < 2:
        raise ValueError("acc_names.mat has unexpected shape")

    code_lookup: dict[int, str] = {}
    short_name_row = names_array[1]
    for idx, raw_name in enumerate(short_name_row, start=1):
        value = raw_name
        while hasattr(value, "shape") and getattr(value, "size", 1) == 1:
            value = value.item()
        code_lookup[idx] = str(value).strip()
    return code_lookup


def _unimib_stream_id_for_trial(trial_index: int, activity_code: str) -> str:
    """Infer right/left pocket stream from UniMiB trial index and activity family."""

    # UniMiB readme: ADLs have 2 trials (1 right, 2 left); falls have 6 trials
    # (1..3 right, 4..6 left). We keep the generic rule explicit.
    if activity_code.startswith("Falling") or activity_code in {
        "HittingObstacle",
        "Syncope",
    }:
        return "front_pocket_right" if trial_index <= 3 else "front_pocket_left"
    return "front_pocket_right" if trial_index == 1 else "front_pocket_left"


def process_unimib_local(
    spec: DatasetSpec,
    writer_registry: WriterRegistry,
    window_manifest_rows: list[dict[str, object]],
    session_manifest_rows: list[dict[str, object]],
) -> None:
    """Build canonical windows from local UniMiB-SHAR MAT files."""

    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError(
            "scipy is required to process UniMiB-SHAR (.mat files). "
            "Install it with: pip install scipy"
        ) from exc

    unimib_root = _resolve_unimib_root(spec.source_root)
    unimib_data_dir = unimib_root / "data"
    ensure_exists(unimib_data_dir / "acc_data.mat", "UniMiB acc_data.mat")
    ensure_exists(unimib_data_dir / "acc_labels.mat", "UniMiB acc_labels.mat")
    ensure_exists(unimib_data_dir / "acc_names.mat", "UniMiB acc_names.mat")

    code_lookup = _extract_unimib_activity_codes(unimib_data_dir)

    data_mat = sio.loadmat(unimib_data_dir / "acc_data.mat")
    labels_mat = sio.loadmat(unimib_data_dir / "acc_labels.mat")
    if "acc_data" not in data_mat or "acc_labels" not in labels_mat:
        raise ValueError("UniMiB MAT files missing expected variables acc_data/acc_labels")

    acc_data = np.asarray(data_mat["acc_data"], dtype=np.float64)
    acc_labels = np.asarray(labels_mat["acc_labels"], dtype=np.int64)
    if acc_data.ndim != 2 or acc_data.shape[1] != 453:
        raise ValueError(f"Unexpected UniMiB acc_data shape: {acc_data.shape}")
    if acc_labels.ndim != 2 or acc_labels.shape[1] < 3 or acc_labels.shape[0] != acc_data.shape[0]:
        raise ValueError(f"Unexpected UniMiB acc_labels shape: {acc_labels.shape}")

    holdout_subjects = resolve_holdout_subjects(
        spec,
        {str(value) for value in np.unique(acc_labels[:, 1]).tolist()},
    )

    center_start = (151 - WINDOW_SAMPLES) // 2
    center_end = center_start + WINDOW_SAMPLES

    for row_index in range(acc_data.shape[0]):
        activity_id = int(acc_labels[row_index, 0])
        subject_local_id = str(int(acc_labels[row_index, 1]))
        trial_index = int(acc_labels[row_index, 2])

        activity_code = code_lookup.get(activity_id)
        if activity_code is None:
            continue

        activity_label = resolve_activity_label(spec.activity_map, activity_code)
        if activity_label is None:
            continue

        sample_row = acc_data[row_index]
        x_values = sample_row[0:151]
        y_values = sample_row[151:302]
        z_values = sample_row[302:453]

        if not (
            np.all(np.isfinite(x_values))
            and np.all(np.isfinite(y_values))
            and np.all(np.isfinite(z_values))
        ):
            continue

        session_meta = SessionMeta(
            dataset_id=spec.dataset_id,
            dataset_enum=spec.dataset_enum,
            stream_id=_unimib_stream_id_for_trial(trial_index, activity_code),
            subject_local_id=subject_local_id,
            source_session_id=(
                f"subject:{subject_local_id}:activity:{activity_code}:"
                f"trial:{trial_index}:row:{row_index}"
            ),
            activity_name_raw=activity_code,
            activity_label=activity_label,
            placement_label="front_pocket",
            split="test" if subject_local_id in holdout_subjects else "train",
        )
        process_resampled_session(
            session_meta=session_meta,
            x_values=x_values[center_start:center_end].astype(np.float32, copy=False),
            y_values=y_values[center_start:center_end].astype(np.float32, copy=False),
            z_values=z_values[center_start:center_end].astype(np.float32, copy=False),
            writer_registry=writer_registry,
            window_manifest_rows=window_manifest_rows,
            session_manifest_rows=session_manifest_rows,
        )


def process_wisdm_v2_local(
    spec: DatasetSpec,
    writer_registry: WriterRegistry,
    window_manifest_rows: list[dict[str, object]],
    session_manifest_rows: list[dict[str, object]],
) -> None:
    """Build canonical windows from local WISDM v2 raw phone accelerometer files.

    Each file ``data_{subject_id}_accel_phone.txt`` contains all 18 activities for
    one subject.  Lines have the format::

        subject_id,activity_code,timestamp_ns,x,y,z;

    Values are already in m/s² (gravity component ≈ 9.81 at rest).  Timestamps
    are nanoseconds (device-local clock) and are converted to elapsed seconds
    before resampling to 50 Hz.
    """

    accel_dir = spec.source_root / "raw" / "phone" / "accel"
    ensure_exists(accel_dir, "WISDM v2 phone accel directory")

    # Resolve holdout subjects from all filenames before processing any file.
    all_subject_ids: set[str] = set()
    for txt_file in accel_dir.glob("data_*_accel_phone.txt"):
        parts = txt_file.stem.split("_")  # ['data', '1600', 'accel', 'phone']
        if len(parts) >= 2:
            all_subject_ids.add(parts[1])

    holdout_subjects = resolve_holdout_subjects(spec, all_subject_ids)

    stream = spec.streams[0]

    for txt_file in sorted(accel_dir.glob("data_*_accel_phone.txt")):
        stem_parts = txt_file.stem.split("_")  # ['data', '1600', 'accel', 'phone']
        if len(stem_parts) < 2:
            continue
        subject_local_id = stem_parts[1]
        split = "test" if subject_local_id in holdout_subjects else "train"

        # Group records by activity code within this subject's file.
        records_by_activity: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)

        with txt_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                line = line.rstrip(";")
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 6:
                    continue
                try:
                    activity_code = parts[1].lower()
                    timestamp_ns = int(parts[2])
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                except (ValueError, IndexError):
                    continue
                records_by_activity[activity_code].append((timestamp_ns, x, y, z))

        for activity_code, records in records_by_activity.items():
            activity_label = resolve_activity_label(spec.activity_map, activity_code)
            if activity_label is None:
                continue

            records.sort(key=lambda r: r[0])
            timestamps_ns = np.array([r[0] for r in records], dtype=np.float64)
            x_values = np.array([r[1] for r in records], dtype=np.float64) * stream.unit_scale_to_ms2
            y_values = np.array([r[2] for r in records], dtype=np.float64) * stream.unit_scale_to_ms2
            z_values = np.array([r[3] for r in records], dtype=np.float64) * stream.unit_scale_to_ms2

            time_seconds = timestamps_ns / 1e9
            time_seconds = time_seconds - time_seconds[0]

            time_seconds, x_values, y_values, z_values = sanitize_numeric_timeseries(
                time_seconds=time_seconds,
                x_values=x_values,
                y_values=y_values,
                z_values=z_values,
            )
            if time_seconds.size < 2:
                continue

            preprocessed_segments = preprocess_numeric_segments(
                time_seconds=time_seconds,
                x_values=x_values,
                y_values=y_values,
                z_values=z_values,
            )
            segment_count = len(preprocessed_segments)
            base_source_session_id = f"subject:{subject_local_id}:activity:{activity_code}"
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
                    activity_name_raw=activity_code,
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


def process_shoaib2013_local(
    spec: DatasetSpec,
    writer_registry: WriterRegistry,
    window_manifest_rows: list[dict[str, object]],
    session_manifest_rows: list[dict[str, object]],
) -> None:
    """Build canonical windows from the Shoaib 2013 Pocket.xlsx file.

    The file contains all 4 participants concatenated with no participant column.
    Timestamps are in milliseconds at 50 Hz.  Rows are sorted by timestamp and
    contiguous label runs are treated as independent sessions; the file contains
    one large recording gap (~4300 s) between two halves.  Because there is no
    explicit participant identifier, every row is assigned to a single synthetic
    subject ("1") and no holdout split is possible — all windows go to
    ``split="train_only"``.
    """
    pocket_xlsx = spec.source_root / "Pocket.xlsx"
    ensure_exists(pocket_xlsx, "Shoaib 2013 Pocket.xlsx")

    stream = spec.streams[0]

    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required to read Shoaib 2013 Pocket.xlsx. "
            "Install it with: pip install openpyxl"
        ) from exc

    df = pd.read_excel(pocket_xlsx, header=0, engine="openpyxl")
    df.columns = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "label"]
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Group contiguous label runs into sessions
    df["_run"] = (df["label"] != df["label"].shift()).cumsum()
    for (label_raw, run_id), group in df.groupby(["label", "_run"], sort=False):
        activity_label = resolve_activity_label(spec.activity_map, str(label_raw).lower())
        if activity_label is None:
            continue

        ts_ms = group["timestamp"].to_numpy(dtype=np.float64)
        time_seconds = ts_ms / 1000.0
        time_seconds = time_seconds - time_seconds[0]
        x_values = group["ax"].to_numpy(dtype=np.float64) * stream.unit_scale_to_ms2
        y_values = group["ay"].to_numpy(dtype=np.float64) * stream.unit_scale_to_ms2
        z_values = group["az"].to_numpy(dtype=np.float64) * stream.unit_scale_to_ms2

        time_seconds, x_values, y_values, z_values = sanitize_numeric_timeseries(
            time_seconds=time_seconds,
            x_values=x_values,
            y_values=y_values,
            z_values=z_values,
        )
        if time_seconds.size < 2:
            continue

        preprocessed_segments = preprocess_numeric_segments(
            time_seconds=time_seconds,
            x_values=x_values,
            y_values=y_values,
            z_values=z_values,
        )
        segment_count = len(preprocessed_segments)
        base_source_session_id = f"run:{run_id}:activity:{str(label_raw).lower()}"
        for preprocessed in preprocessed_segments:
            if preprocessed.total.time_seconds.size < 2:
                continue
            session_meta = SessionMeta(
                dataset_id=spec.dataset_id,
                dataset_enum=spec.dataset_enum,
                stream_id=stream.stream_id,
                subject_local_id="1",
                source_session_id=_build_segment_source_session_id(
                    base_source_session_id,
                    preprocessed.segment_index,
                    segment_count,
                ),
                activity_name_raw=str(label_raw).lower(),
                activity_label=activity_label,
                placement_label=stream.placement_label,
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


def process_shoaib_sensors_local(
    spec: DatasetSpec,
    writer_registry: WriterRegistry,
    window_manifest_rows: list[dict[str, object]],
    session_manifest_rows: list[dict[str, object]],
) -> None:
    """Build canonical windows from Shoaib Sensors per-participant CSV files.

    Each file (``Participant_N.csv``) has a 2-row header: row 0 holds the position
    label (Left_pocket, Right_pocket, …) and row 1 holds per-column names.  The
    last non-NaN column contains the activity string for that row.  Left pocket
    occupies columns 0–12 (skip col 13 = NaN separator); right pocket occupies
    columns 14–26.  Both are sampled at 50 Hz in m/s².
    """
    ensure_exists(spec.source_root, "Shoaib Sensors dataset directory")

    # Participant files are named Participant_1.csv … Participant_10.csv
    all_subject_ids = {
        p.stem.split("_")[1]
        for p in spec.source_root.glob("Participant_*.csv")
    }
    holdout_subjects = resolve_holdout_subjects(spec, all_subject_ids)

    # Stream → column index mapping (0-based into the data portion, after skipping header)
    STREAM_COLS = {
        "left_pocket":  {"ts": 0,  "ax": 1,  "ay": 2,  "az": 3},
        "right_pocket": {"ts": 14, "ax": 15, "ay": 16, "az": 17},
    }

    for csv_path in sorted(spec.source_root.glob("Participant_*.csv")):
        subject_local_id = csv_path.stem.split("_")[1]
        split = "test" if subject_local_id in holdout_subjects else "train"

        raw = pd.read_csv(csv_path, header=None, skiprows=2, low_memory=False)
        # Activity label is the last non-NaN value in each row (column 69)
        activity_col = raw.iloc[:, -1]

        for stream in spec.streams:
            cols = STREAM_COLS[stream.stream_id]
            ts_col = raw.iloc[:, cols["ts"]]
            ax_col = raw.iloc[:, cols["ax"]]
            ay_col = raw.iloc[:, cols["ay"]]
            az_col = raw.iloc[:, cols["az"]]

            # Build a sub-dataframe aligned with activities
            sub = pd.DataFrame({
                "ts": pd.to_numeric(ts_col, errors="coerce"),
                "ax": pd.to_numeric(ax_col, errors="coerce"),
                "ay": pd.to_numeric(ay_col, errors="coerce"),
                "az": pd.to_numeric(az_col, errors="coerce"),
                "label": activity_col,
            }).dropna(subset=["ts", "ax", "ay", "az", "label"])

            sub["_run"] = (sub["label"] != sub["label"].shift()).cumsum()
            for (label_raw, run_id), group in sub.groupby(["label", "_run"], sort=False):
                activity_label = resolve_activity_label(spec.activity_map, str(label_raw).lower())
                if activity_label is None:
                    continue

                x_values = group["ax"].to_numpy(dtype=np.float64) * stream.unit_scale_to_ms2
                y_values = group["ay"].to_numpy(dtype=np.float64) * stream.unit_scale_to_ms2
                z_values = group["az"].to_numpy(dtype=np.float64) * stream.unit_scale_to_ms2
                # Timestamps in this dataset are a constant value; synthesize time
                # from row index at the known 50 Hz sample rate instead.
                time_seconds = np.arange(len(x_values), dtype=np.float64) / TARGET_SAMPLE_RATE_HZ

                time_seconds, x_values, y_values, z_values = sanitize_numeric_timeseries(
                    time_seconds=time_seconds,
                    x_values=x_values,
                    y_values=y_values,
                    z_values=z_values,
                )
                if time_seconds.size < 2:
                    continue

                preprocessed_segments = preprocess_numeric_segments(
                    time_seconds=time_seconds,
                    x_values=x_values,
                    y_values=y_values,
                    z_values=z_values,
                )
                segment_count = len(preprocessed_segments)
                base_source_session_id = (
                    f"subject:{subject_local_id}:stream:{stream.stream_id}"
                    f":run:{run_id}:activity:{str(label_raw).lower()}"
                )
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
                        activity_name_raw=str(label_raw).lower(),
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


def process_ut_complex_local(
    spec: DatasetSpec,
    writer_registry: WriterRegistry,
    window_manifest_rows: list[dict[str, object]],
    session_manifest_rows: list[dict[str, object]],
) -> None:
    """Build canonical windows from the UT Complex pocket CSV.

    The file ``smartphoneatpocket.csv`` has no header.  Columns are:
    timestamp (ms), acc_x, acc_y, acc_z, lin_x, lin_y, lin_z,
    gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z, activity_code (integer).

    Accelerometer values are in m/s².  The file combines multiple participants
    with no participant column; the two large timestamp gaps (~30 s and ~10 s)
    separate recording sessions.  All rows are assigned to split="train_only"
    because there is no subject identifier for a proper holdout split.
    """
    pocket_csv = spec.source_root / "smartphoneatpocket.csv"
    ensure_exists(pocket_csv, "UT Complex smartphoneatpocket.csv")

    stream = spec.streams[0]

    df = pd.read_csv(pocket_csv, header=None, dtype={13: str})
    df.columns = [
        "ts", "ax", "ay", "az",
        "lx", "ly", "lz",
        "gx", "gy", "gz",
        "mx", "my", "mz",
        "label",
    ]
    df = df.sort_values("ts").reset_index(drop=True)
    df["label"] = df["label"].astype(str).str.strip()

    # Group by contiguous (label, run) blocks; timestamp gaps create natural boundaries
    df["_run"] = (df["label"] != df["label"].shift()).cumsum()
    for (label_code, run_id), group in df.groupby(["label", "_run"], sort=False):
        activity_label = resolve_activity_label(spec.activity_map, str(label_code))
        if activity_label is None:
            continue

        x_values = group["ax"].to_numpy(dtype=np.float64) * stream.unit_scale_to_ms2
        y_values = group["ay"].to_numpy(dtype=np.float64) * stream.unit_scale_to_ms2
        z_values = group["az"].to_numpy(dtype=np.float64) * stream.unit_scale_to_ms2
        # Timestamps in this dataset are a constant value; synthesize time
        # from row index at the known 50 Hz sample rate instead.
        time_seconds = np.arange(len(x_values), dtype=np.float64) / TARGET_SAMPLE_RATE_HZ

        time_seconds, x_values, y_values, z_values = sanitize_numeric_timeseries(
            time_seconds=time_seconds,
            x_values=x_values,
            y_values=y_values,
            z_values=z_values,
        )
        if time_seconds.size < 2:
            continue

        preprocessed_segments = preprocess_numeric_segments(
            time_seconds=time_seconds,
            x_values=x_values,
            y_values=y_values,
            z_values=z_values,
        )
        segment_count = len(preprocessed_segments)
        base_source_session_id = f"run:{run_id}:activity:{label_code}"
        for preprocessed in preprocessed_segments:
            if preprocessed.total.time_seconds.size < 2:
                continue
            session_meta = SessionMeta(
                dataset_id=spec.dataset_id,
                dataset_enum=spec.dataset_enum,
                stream_id=stream.stream_id,
                subject_local_id="1",
                source_session_id=_build_segment_source_session_id(
                    base_source_session_id,
                    preprocessed.segment_index,
                    segment_count,
                ),
                activity_name_raw=str(label_code),
                activity_label=activity_label,
                placement_label=stream.placement_label,
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
            elif spec.source_kind == "unimib_local":
                process_unimib_local(
                    spec=spec,
                    writer_registry=writer_registry,
                    window_manifest_rows=window_manifest_rows,
                    session_manifest_rows=session_manifest_rows,
                )
            elif spec.source_kind == "wisdm_v2_local":
                process_wisdm_v2_local(
                    spec=spec,
                    writer_registry=writer_registry,
                    window_manifest_rows=window_manifest_rows,
                    session_manifest_rows=session_manifest_rows,
                )
            elif spec.source_kind == "shoaib2013_local":
                process_shoaib2013_local(
                    spec=spec,
                    writer_registry=writer_registry,
                    window_manifest_rows=window_manifest_rows,
                    session_manifest_rows=session_manifest_rows,
                )
            elif spec.source_kind == "shoaib_sensors_local":
                process_shoaib_sensors_local(
                    spec=spec,
                    writer_registry=writer_registry,
                    window_manifest_rows=window_manifest_rows,
                    session_manifest_rows=session_manifest_rows,
                )
            elif spec.source_kind == "ut_complex_local":
                process_ut_complex_local(
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
        unimib_dir=args.unimib_dir,
        wisdm_v2_dir=args.wisdm_v2_dir,
        shoaib2013_dir=args.shoaib2013_dir,
        shoaib_sensors_dir=args.shoaib_sensors_dir,
        ut_complex_dir=args.ut_complex_dir,
    )
    selected_specs = resolve_selected_specs(specs, args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_contract_build(selected_specs=selected_specs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()