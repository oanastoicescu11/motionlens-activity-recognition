import unittest

from app.backend.core.auth import issue_session_credentials
from app.backend.ingest_routes import process_ingest_batch
from app.backend.models.session import SessionMeta
from app.backend.storage.memory import InMemoryStore


class SessionIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryStore()
        a = issue_session_credentials()
        b = issue_session_credentials()
        self.meta_a = SessionMeta(
            session_id="sess-a",
            owner_id="owner-a",
            viewer_token=a.viewer_token,
            write_token=a.write_token,
            device_id="dev-a",
            expires_at_ns=10**30,
        )
        self.meta_b = SessionMeta(
            session_id="sess-b",
            owner_id="owner-b",
            viewer_token=b.viewer_token,
            write_token=b.write_token,
            device_id="dev-b",
            expires_at_ns=10**30,
        )
        self.store.create_session(self.meta_a)
        self.store.create_session(self.meta_b)

    def test_cross_session_write_is_forbidden(self) -> None:
        body = {
            "messageId": 1,
            "sessionId": "sess-a",
            "deviceId": "dev-a",
            "samples": [
                {"timestampMs": 100, "acc": {"x": 0.1, "y": 0.2, "z": 0.3}}
            ],
        }

        with self.assertRaises(PermissionError):
            process_ingest_batch(
                self.store,
                write_token=self.meta_b.write_token,
                raw_body=body,
                now_ns=100,
            )

    def test_cross_session_web_write_is_forbidden(self) -> None:
        body = {
            "source": "web-devicemotion",
            "messageId": 1,
            "sessionId": "sess-a",
            "deviceId": "dev-a",
            "samples": [
                {"timestampMs": 100, "acc": {"x": 0.1, "y": 0.2, "z": 0.3}}
            ],
        }

        with self.assertRaises(PermissionError):
            process_ingest_batch(
                self.store,
                write_token=self.meta_b.write_token,
                raw_body=body,
                now_ns=100,
            )


if __name__ == "__main__":
    unittest.main()
