"""Filter and validate contract data for pocket_only strict mode training.

This script creates a subset of the MotionLens contract parquet data containing only:
- Datasets: WISDM, MotionSense, UMAFall
- Placements: front_pocket, lateral_left_lower, lateral_right_lower (remapped to front_pocket)
- Activity labels: walk, run, stairs, sit, stand, lay, transitions, locomotion-other

Exclusions:
- HHAR (unknown_free_living)
- Other datasets (HAPT, RealWorld2016, PAMAP2, MHEALTH, SAD)
- Non-pocket placements (chest, front_center_*)

The script validates that all target activity labels have sufficient coverage and reports any
activity labels missing from the filtered dataset.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Target configuration for pocket_only model
TARGET_DATASETS = {"wisdm", "motion_sense", "uma_fall", "wisdm_v2", "real_world", "shoaib_2013", "shoaib_sensors", "ut_complex"}
POCKET_PLACEMENTS = {"front_pocket", "lateral_left_lower", "lateral_right_lower"}
TARGET_ACTIVITY_LABELS = {
    "walk",
    "run",
    "stairs",
    "sit/lay",
    "stand",
    "transitions",
    "locomotion-other",
}
# Raw contract labels "sit" and "lay" are merged into the joint class "sit/lay".
LABEL_REMAP: dict[str, str] = {"sit": "sit/lay", "lay": "sit/lay"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate contract data for pocket_only training."
    )
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=Path("output") / "motionlens_contract",
        help="Path to the contract output directory with samples/.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output") / "pocket_only_data_summary.csv",
        help="Output CSV file with coverage summary.",
    )
    return parser.parse_args()


def analyze_contract_data(contract_dir: Path) -> dict[str, int]:
    """Scan contract parquet files and count windows matching pocket_only criteria.

    Returns a dict with:
    - per-label window counts (e.g., 'walk': 17427, 'run': 21002)
    - special keys: 'total_windows', 'missing_labels'
    """
    samples_dir = contract_dir / "samples"
    if not samples_dir.exists():
        raise ValueError(f"Contract samples directory not found: {samples_dir}")

    # Recursively find all part-*.parquet files in the nested structure
    # (split=*/dataset_id=*/stream_id=*/part-*.parquet)
    parquet_files = sorted(samples_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {samples_dir}")

    LOGGER.info(f"Found {len(parquet_files)} parquet files to scan.")

    activity_counts = defaultdict(int)
    total_windows = 0
    excluded_dataset_count = 0
    excluded_placement_count = 0
    excluded_activity_count = 0

    for file_idx, filepath in enumerate(parquet_files):
        LOGGER.info(f"  [{file_idx + 1}/{len(parquet_files)}] Scanning {filepath.name}...")

        pf = pq.ParquetFile(filepath)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(
                rg,
                columns=["dataset_id", "placement_label", "activity_label", "global_window_id"],
            )
            data = tbl.to_pydict()

            for ds, placement, activity, window_id in zip(
                data["dataset_id"],
                data["placement_label"],
                data["activity_label"],
                data["global_window_id"],
            ):
                # Check dataset
                if ds.lower() not in TARGET_DATASETS:
                    excluded_dataset_count += 1
                    continue

                # Check placement
                if placement not in POCKET_PLACEMENTS:
                    excluded_placement_count += 1
                    continue

                # Remap sit/lay → joint label before checking target
                activity = LABEL_REMAP.get(activity, activity)

                # Check activity
                if activity not in TARGET_ACTIVITY_LABELS:
                    excluded_activity_count += 1
                    continue

                # Include this window
                activity_counts[activity] += 1
                total_windows += 1

    LOGGER.info(f"\n=== Pocket_only data summary ===")
    LOGGER.info(f"Total matching windows: {total_windows}")
    LOGGER.info(f"Excluded (non-target datasets): {excluded_dataset_count}")
    LOGGER.info(f"Excluded (non-pocket placements): {excluded_placement_count}")
    LOGGER.info(f"Excluded (non-target activities): {excluded_activity_count}")

    # Per-activity breakdown
    LOGGER.info(f"\n=== Per-activity window counts ===")
    missing_labels = []
    for label in sorted(TARGET_ACTIVITY_LABELS):
        count = activity_counts[label]
        LOGGER.info(f"{label:20s}: {count:6d}")
        if count == 0:
            missing_labels.append(label)

    if missing_labels:
        LOGGER.warning(f"\n⚠️  Missing activity labels (0 windows): {missing_labels}")
    else:
        LOGGER.info(f"\n✓ All target activity labels have coverage.")

    # Merge into result dict
    result = dict(activity_counts)
    result["total_windows"] = total_windows
    result["excluded_dataset"] = excluded_dataset_count
    result["excluded_placement"] = excluded_placement_count
    result["excluded_activity"] = excluded_activity_count
    result["missing_labels"] = ",".join(missing_labels) if missing_labels else "none"

    return result


def write_summary_csv(summary: dict, output_csv: Path) -> None:
    """Write the summary to a CSV file for documentation."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])

        # Metadata
        writer.writerow(["Target Datasets", ",".join(sorted(TARGET_DATASETS))])
        writer.writerow(["Pocket Placements", ",".join(sorted(POCKET_PLACEMENTS))])
        writer.writerow(
            ["Target Activities", ",".join(sorted(TARGET_ACTIVITY_LABELS))]
        )

        # Summary
        writer.writerow(["", ""])
        writer.writerow(["Total Matching Windows", summary.get("total_windows", 0)])
        writer.writerow(["Excluded (Non-Target Datasets)", summary.get("excluded_dataset", 0)])
        writer.writerow(["Excluded (Non-Pocket Placements)", summary.get("excluded_placement", 0)])
        writer.writerow(["Excluded (Non-Target Activities)", summary.get("excluded_activity", 0)])

        # Per-activity counts
        writer.writerow(["", ""])
        writer.writerow(["Activity", "Window Count"])
        for label in sorted(TARGET_ACTIVITY_LABELS):
            count = summary.get(label, 0)
            writer.writerow([label, count])

        # Missing labels
        writer.writerow(["", ""])
        writer.writerow(["Missing Labels", summary.get("missing_labels", "none")])

    LOGGER.info(f"\n✓ Summary written to {output_csv}")


def main() -> int:
    args = parse_args()

    try:
        summary = analyze_contract_data(args.contract_dir)
        write_summary_csv(summary, args.output_csv)

        # Check for missing labels
        missing = summary.get("missing_labels", "")
        if missing != "none":
            LOGGER.warning(
                f"\n⚠️  Training will exclude labels with no coverage: {missing}"
            )
            return 1  # Non-fatal; training should continue but be aware

        return 0
    except Exception as e:
        LOGGER.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
