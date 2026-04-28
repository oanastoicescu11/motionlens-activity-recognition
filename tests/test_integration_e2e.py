"""End-to-end integration tests for ingest â†’ queue â†’ worker â†’ read pipeline."""

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.core.auth import issue_session_credentials
from app.backend.ingest_routes import process_ingest_batch
from app.backend.models.session import SessionMeta
from app.backend.storage.memory import InMemoryStore
from app.worker.consumer import consume_once


class IntegrationE2ETests(unittest.TestCase):
    """Full ingest â†’ queue â†’ worker â†’ inference read pipeline."""

    def setUp(self):
        self.store = InMemoryStore()
        creds = issue_session_credentials()
        self.meta = SessionMeta(
            session_id="sess-1",
            owner_id="owner-1",
            viewer_token=creds.viewer_token,
            write_token=creds.write_token,
            device_id="dev-1",
            expires_at_ns=10**30,
        )
        self.store.create_session(self.meta)

    def test_full_pipeline_ingest_to_inference_read(self):
        """Full flow: ingest batch â†’ dedupe check â†’ queue task â†’ worker consume â†’ inference available."""
        # 1. Ingest 128 samples at 50 Hz (2.56 seconds) - minimum for feature extraction
        # Generate realistic accelerometer samples simulating a walking motion
        base_time_ms = 1698501144401
        sample_interval_ms = 20  # 20 ms at 50 Hz
        
        samples = []
        for i in range(128):
            # Simulate walking with sinusoidal pattern
            time_s = i * 0.02  # seconds
            walking_freq_hz = 2.0  # typical walking cadence
            amplitude = 5.0
            x_val = amplitude * math.sin(2 * math.pi * walking_freq_hz * time_s)
            y_val = amplitude * math.cos(2 * math.pi * walking_freq_hz * time_s) + 10.0  # gravity component
            z_val = 0.5 * math.sin(2 * math.pi * walking_freq_hz * time_s * 2) + 10.0  # double cadence + gravity
            
            samples.append({
                "timestampMs": base_time_ms + i * sample_interval_ms,
                "acc": {"x": x_val, "y": y_val, "z": z_val},
            })
        
        batch_body = {
            "source": "web-devicemotion",
            "messageId": 1,
            "sessionId": "sess-1",
            "deviceId": "dev-1",
            "samples": samples,
        }
        ingest_result = process_ingest_batch(
            self.store,
            write_token=self.meta.write_token,
            raw_body=batch_body,
            now_ns=1000,
        )
        self.assertEqual(ingest_result["accepted_points"], 128)
        self.assertTrue(ingest_result["enqueued"])
        self.assertFalse(ingest_result["dropped_duplicate"])

        # 2. Verify signal is stored immediately
        points = self.store.read_raw_points("sess-1")
        self.assertEqual(len(points), 128)
        self.assertEqual(points[0][0], base_time_ms * 1_000_000)  # time_ns
        self.assertAlmostEqual(points[0][1], 0.0, places=1)  # x (first sine sample â‰ˆ 0)

        # 3. Verify inference task was queued
        task = self.store.pop_inference_task()
        self.assertIsNotNone(task)
        self.assertEqual(task["session_id"], "sess-1")
        self.assertEqual(task["message_id"], 1)
        self.assertEqual(task["points_added"], 128)

        # 4. Verify no inference snapshot yet
        snapshot_before = self.store.get_latest_inference("sess-1")
        self.assertIsNone(snapshot_before)

        # 5. Re-queue and consume via worker
        self.store.enqueue_inference_task(task)
        processed = consume_once(self.store)
        self.assertTrue(processed)

        # 6. Verify inference snapshot is now available
        snapshot_after = self.store.get_latest_inference("sess-1")
        self.assertIsNotNone(snapshot_after)
        if snapshot_after.get("status") == "error":
            print(f"INFERENCE ERROR: {snapshot_after.get('error')}")
        self.assertEqual(snapshot_after["status"], "success")
        self.assertEqual(snapshot_after["session_id"], "sess-1")
        self.assertIn("activity", snapshot_after)
        self.assertIn("confidence", snapshot_after)
        self.assertIsInstance(snapshot_after["confidence"], float)
        self.assertGreater(snapshot_after["confidence"], 0.0)
        self.assertLess(snapshot_after["confidence"], 1.0)

    def test_duplicate_ingest_does_not_enqueue_again(self):
        """Duplicate message is dropped and not enqueued for inference."""
        batch_body = {
            "source": "web-devicemotion",
            "messageId": 7,
            "sessionId": "sess-1",
            "deviceId": "dev-1",
            "samples": [
                {"timestampMs": 100, "acc": {"x": 1.0, "y": 2.0, "z": 3.0}}
            ],
        }

        # First ingest
        first = process_ingest_batch(
            self.store,
            write_token=self.meta.write_token,
            raw_body=batch_body,
            now_ns=1000,
        )
        self.assertTrue(first["enqueued"])
        first_task = self.store.pop_inference_task()
        self.assertIsNotNone(first_task)

        # Duplicate ingest
        second = process_ingest_batch(
            self.store,
            write_token=self.meta.write_token,
            raw_body=batch_body,
            now_ns=1001,
        )
        self.assertFalse(second["enqueued"])
        self.assertTrue(second["dropped_duplicate"])

        # Queue should be empty
        second_task = self.store.pop_inference_task()
        self.assertIsNone(second_task)

    def test_signal_available_while_inference_processing(self):
        """Reader can fetch signal immediately while inference is queued."""
        batch_body = {
            "source": "web-devicemotion",
            "messageId": 1,
            "sessionId": "sess-1",
            "deviceId": "dev-1",
            "samples": [
                {"timestampMs": 100, "acc": {"x": 0.5, "y": 1.5, "z": 2.5}},
                {"timestampMs": 101, "acc": {"x": 0.6, "y": 1.6, "z": 2.6}},
            ],
        }

        process_ingest_batch(
            self.store,
            write_token=self.meta.write_token,
            raw_body=batch_body,
            now_ns=1000,
        )

        # Signal is immediately available
        points = self.store.read_raw_points("sess-1", limit=10)
        self.assertEqual(len(points), 2)

        # Inference is not yet available (still queued)
        inference = self.store.get_latest_inference("sess-1")
        self.assertIsNone(inference)


if __name__ == "__main__":
    unittest.main()
