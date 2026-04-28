import unittest

from app.backend.core.auth import issue_session_credentials
from app.backend.ingest_routes import process_ingest_batch
from app.backend.models.session import SessionMeta
from app.backend.storage.memory import InMemoryStore


def _body(message_id: int = 1) -> dict:
    return {
        "source": "web-devicemotion",
        "messageId": message_id,
        "sessionId": "sess-1",
        "deviceId": "dev-1",
        "placementLabel": "front_pocket",
        "samples": [
            {
                "timestampMs": 100,
                "acc": {"x": 1.0, "y": 2.0, "z": 3.0},
            }
        ],
    }


class IngestDedupeTests(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_duplicate_message_is_dropped(self) -> None:
        first = process_ingest_batch(
            self.store,
            write_token=self.meta.write_token,
            raw_body=_body(7),
            now_ns=1000,
        )
        queued = self.store.pop_inference_task()
        self.assertIsNotNone(queued)
        self.assertEqual(queued["placement_label"], "front_pocket")
        self.store.enqueue_inference_task(queued)
        second = process_ingest_batch(
            self.store,
            write_token=self.meta.write_token,
            raw_body=_body(7),
            now_ns=1001,
        )

        self.assertEqual(first["accepted_points"], 1)
        self.assertEqual(second["accepted_points"], 0)
        self.assertTrue(second["dropped_duplicate"])


if __name__ == "__main__":
    unittest.main()
