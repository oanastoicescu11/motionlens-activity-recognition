"""Worker pipeline integration tests."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel

from app.backend.storage.memory import InMemoryStore
from app.worker._steps import _load_model
from app.worker.consumer import consume_once
from app.worker.pipeline import process_task


class WorkerPipelineTests(unittest.TestCase):
    """Test worker queue consumption and task processing."""

    def test_consume_once_returns_false_when_queue_empty(self):
        """consume_once returns False if no task available."""
        store = InMemoryStore()
        processed = consume_once(store)
        self.assertFalse(processed)

    def test_consume_once_processes_task_and_publishes_snapshot(self):
        """consume_once pops task, processes it, and stores inference snapshot."""
        store = InMemoryStore()
        task = {
            "session_id": "sess-1",
            "message_id": 1,
            "points_added": 100,
            "received_at_ns": 1000000,
        }
        store.enqueue_inference_task(task)

        processed = consume_once(store)
        self.assertTrue(processed)

        snapshot = store.get_latest_inference("sess-1")
        self.assertIsNotNone(snapshot)
        # Snapshot is stored as dict (serialized from Pydantic model)
        self.assertIn(snapshot["status"], {"success", "queued", "error"})
        self.assertEqual(snapshot["session_id"], "sess-1")
        self.assertEqual(snapshot["message_id"], 1)

    def test_process_task_returns_pydantic_snapshot(self):
        """process_task generates a Pydantic snapshot with required fields."""
        task = {
            "session_id": "sess-2",
            "message_id": 42,
            "points_added": 50,
        }
        snapshot = process_task(task)
        # Now returns Pydantic model, not dict
        self.assertIsInstance(snapshot, BaseModel)
        # Access fields directly from Pydantic model
        self.assertIn(snapshot.status, {"success", "queued", "error", "queued-processed"})
        self.assertEqual(snapshot.session_id, "sess-2")
        self.assertEqual(snapshot.message_id, 42)
        # Can serialize to dict for backward compatibility
        snapshot_dict = snapshot.model_dump()
        self.assertIn("status", snapshot_dict)
        self.assertIn("updated_at_ns", snapshot_dict)

    def test_load_model_exposes_feature_names_for_live_insights(self):
        """Loaded runtime artifacts include the feature schema used by live insights."""
        artifacts, bundle_meta = _load_model()

        self.assertTrue(artifacts.feature_names)
        self.assertIn("body_mag_energy", artifacts.feature_names)
        self.assertIn("gravity_tilt_start_end_delta", artifacts.feature_names)
        self.assertEqual(artifacts.feature_names, bundle_meta["feature_names"])

    def test_load_model_requires_runtime_bundle(self):
        """Live runtime must fail clearly when the deployment bundle is absent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_steps_path = Path(temp_dir) / "src" / "app" / "worker" / "_steps.py"
            fake_steps_path.parent.mkdir(parents=True)
            fake_steps_path.touch()

            with patch("app.worker._steps.__file__", str(fake_steps_path)):
                with self.assertRaises(FileNotFoundError) as context:
                    _load_model()

        error_message = str(context.exception)
        self.assertIn("inference_bundle.joblib", error_message)
        self.assertIn("output/", error_message)

    def test_load_model_uses_bundle_without_secondary_files(self):
        """The deployment contract only needs the bundled runtime artifact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_steps_path = Path(temp_dir) / "src" / "app" / "worker" / "_steps.py"
            fake_steps_path.parent.mkdir(parents=True)
            fake_steps_path.touch()

            artifacts_dir = Path(temp_dir) / "src" / "artifacts" / "model"
            artifacts_dir.mkdir(parents=True)
            bundle_path = artifacts_dir / "inference_bundle.joblib"
            bundle = {
                "base_model": {"kind": "bundle-only"},
                "label_order": ["walk"],
                "feature_names": ["body_mag_energy"],
                "placement_labels": [],
                "temporal_decoder": "none",
                "log_init_probs": np.asarray([0.0], dtype=np.float64),
                "log_trans_probs": np.asarray([[0.0]], dtype=np.float64),
                "use_hierarchical": False,
                "coarse_model": None,
                "fine_models": None,
                "static_specialist_model": None,
                "static_specialist_labels": None,
                "static_specialist_threshold": 0.35,
                "static_specialist_blend": 0.6,
                "target_sample_rate_hz": 50.0,
            }
            joblib.dump(bundle, bundle_path)

            with patch("app.worker._steps.__file__", str(fake_steps_path)):
                artifacts, bundle_meta = _load_model()

        self.assertEqual(artifacts.base_model, {"kind": "bundle-only"})
        self.assertEqual(artifacts.feature_names, ["body_mag_energy"])
        self.assertEqual(bundle_meta["feature_names"], ["body_mag_energy"])
        self.assertEqual(bundle_meta["target_sample_rate_hz"], 50.0)


if __name__ == "__main__":
    unittest.main()
