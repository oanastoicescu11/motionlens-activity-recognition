"""Worker pipeline integration tests."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel

from app.backend.storage.memory import InMemoryStore
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


if __name__ == "__main__":
    unittest.main()
