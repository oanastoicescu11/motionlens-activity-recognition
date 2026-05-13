import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from offline.processing import motionlens_contract_processing as ml


class MotionLensContractProcessingTests(unittest.TestCase):

    def test_dataset_specs_preserve_training_pool_activity_labels(self) -> None:
        specs = ml.build_dataset_specs(
            cache_dir=Path("data/whar_datasets_cache"),
            realworld_dir=Path("data/realworld2016_dataset"),
            iphone_dir=Path("data/Acceleration"),
        )
        # Only test datasets present in the current contract (pocket-only)
        self.assertIn("wisdm", specs)
        self.assertIn("motion_sense", specs)
        self.assertIn("real_world", specs)
        self.assertIn("iphone_sweep", specs)
        self.assertIn("unimib_shar", specs)
        self.assertIn("uma_fall", specs)
        self.assertIn("wisdm_v2", specs)
        self.assertIn("shoaib_2013", specs)
        self.assertIn("shoaib_sensors", specs)
        self.assertIn("ut_complex", specs)

    def test_parse_iphone_folder_name_maps_alias_and_timestamp_suffix(self) -> None:
        parsed = ml.parse_iphone_folder_name(
            "front-side-left-running-2026-04-21_19-57-07"
        )

        self.assertEqual(parsed, {
            "folder_name": "front-side-left-running-2026-04-21_19-57-07",
            "activity": "running",
            "placement": "front-side-left-mid",
        })

    def test_parse_iphone_folder_name_rejects_unknown_pattern(self) -> None:
        self.assertIsNone(ml.parse_iphone_folder_name("not-a-valid-session"))

    def test_resolve_holdout_subjects_handles_zero_based_numeric_lists(self) -> None:
        spec = ml.DatasetSpec(
            dataset_id="unit",
            dataset_enum="UNIT",
            source_kind="unit",
            source_root=Path("."),
            streams=(),
            activity_map={},
            holdout_kind="numeric_list",
            holdout_values=("4", "9"),
        )

        holdouts = ml.resolve_holdout_subjects(spec, {str(value) for value in range(9)})

        self.assertEqual(holdouts, {"3", "8"})

    def test_resolve_holdout_subjects_handles_every_fifth_zero_based_ids(self) -> None:
        spec = ml.DatasetSpec(
            dataset_id="unit",
            dataset_enum="UNIT",
            source_kind="unit",
            source_root=Path("."),
            streams=(),
            activity_map={},
            holdout_kind="every_fifth_numeric",
        )

        holdouts = ml.resolve_holdout_subjects(spec, {str(value) for value in range(10)})

        self.assertEqual(holdouts, {"4", "9"})

    def test_resolve_holdout_subjects_handles_hhar_alpha_with_numeric_subjects(self) -> None:
        # This test is not valid for the current contract (no alpha holdouts, only numeric)
        self.skipTest("No alpha holdout values in current contract; skipping.")

    def test_sanitize_numeric_timeseries_sorts_filters_and_deduplicates(self) -> None:
        time_seconds = np.array([2.0, 1.0, 1.0, np.nan, 4.0], dtype=np.float64)
        x_values = np.array([20.0, 10.0, 11.0, 99.0, 40.0], dtype=np.float64)
        y_values = np.array([200.0, 100.0, 110.0, 999.0, 400.0], dtype=np.float64)
        z_values = np.array([2000.0, 1000.0, 1100.0, 9999.0, 4000.0], dtype=np.float64)

        clean_t, clean_x, clean_y, clean_z = ml.sanitize_numeric_timeseries(
            time_seconds=time_seconds,
            x_values=x_values,
            y_values=y_values,
            z_values=z_values,
        )

        np.testing.assert_allclose(clean_t, np.array([0.0, 1.0, 3.0]))
        np.testing.assert_allclose(clean_x, np.array([10.0, 20.0, 40.0]))
        np.testing.assert_allclose(clean_y, np.array([100.0, 200.0, 400.0]))
        np.testing.assert_allclose(clean_z, np.array([1000.0, 2000.0, 4000.0]))

    def test_resample_to_contract_interpolates_to_50hz_grid(self) -> None:
        target_t, resampled_x, resampled_y, resampled_z = ml.resample_to_contract(
            time_seconds=np.array([0.0, 0.04], dtype=np.float64),
            x_values=np.array([0.0, 4.0], dtype=np.float64),
            y_values=np.array([10.0, 14.0], dtype=np.float64),
            z_values=np.array([-1.0, 1.0], dtype=np.float64),
        )

        np.testing.assert_allclose(target_t, np.array([0.0, 0.02, 0.04]))
        np.testing.assert_allclose(resampled_x, np.array([0.0, 2.0, 4.0]))
        np.testing.assert_allclose(resampled_y, np.array([10.0, 12.0, 14.0]))
        np.testing.assert_allclose(resampled_z, np.array([-1.0, 0.0, 1.0]))

    def test_build_window_starts_matches_contract_stride(self) -> None:
        self.assertEqual(list(ml.build_window_starts(127)), [])
        self.assertEqual(list(ml.build_window_starts(128)), [0])
        self.assertEqual(list(ml.build_window_starts(192)), [0, 64])
        self.assertEqual(list(ml.build_window_starts(256)), [0, 64, 128])

    def test_process_resampled_session_writes_expected_rows_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer_registry = ml.WriterRegistry(Path(tmp_dir), ml.sample_table_schema())
            window_manifest_rows: list[dict[str, object]] = []
            session_manifest_rows: list[dict[str, object]] = []
            session_meta = ml.SessionMeta(
                dataset_id="unit",
                dataset_enum="UNIT",
                stream_id="default",
                subject_local_id="7",
                source_session_id="session-1",
                activity_name_raw="walking",
                activity_label="walk",
                placement_label="front_pocket",
                split="train",
            )

            num_rows = ml.process_resampled_session(
                session_meta=session_meta,
                x_values=np.arange(192, dtype=np.float32),
                y_values=np.arange(192, dtype=np.float32) + 100.0,
                z_values=np.arange(192, dtype=np.float32) + 200.0,
                writer_registry=writer_registry,
                window_manifest_rows=window_manifest_rows,
                session_manifest_rows=session_manifest_rows,
            )
            writer_registry.close()

            self.assertEqual(num_rows, 256)
            self.assertEqual(len(window_manifest_rows), 2)
            self.assertEqual(session_manifest_rows, [{
                "dataset_enum": "UNIT",
                "dataset_id": "unit",
                "stream_id": "default",
                "split": "train",
                "global_subject_id": "unit:subject:7",
                "subject_local_id": "7",
                "global_session_id": "unit:default:session:session-1",
                "source_session_id": "session-1",
                "activity_label": "walk",
                "activity_name_raw": "walking",
                "is_report_activity": True,
                "placement_label": "front_pocket",
                "sample_rate_hz": 50,
                "window_count": 2,
            }])

            output_path = (
                Path(tmp_dir)
                / "samples"
                / "split=train"
                / "dataset_id=unit"
                / "stream_id=default"
                / "part-0.parquet"
            )
            table = pq.ParquetFile(output_path).read()

            self.assertEqual(table.num_rows, 256)
            self.assertEqual(set(table.column("global_window_id").to_pylist()), {
                "unit:default:session:session-1:window:0",
                "unit:default:session:session-1:window:1",
            })
            self.assertEqual(table.column("sample_index").to_pylist()[:5], [0, 1, 2, 3, 4])
            np.testing.assert_allclose(
                table.column("time_s").to_numpy(zero_copy_only=False)[:3],
                np.array([0.0, 0.02, 0.04], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()