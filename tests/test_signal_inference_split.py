import unittest

from app.backend.core.auth import issue_session_credentials
from app.backend.models.session import SessionMeta
from app.backend.storage.memory import InMemoryStore


class SignalInferenceSplitTests(unittest.TestCase):
    def test_signal_available_when_inference_missing(self) -> None:
        store = InMemoryStore()
        creds = issue_session_credentials()
        meta = SessionMeta(
            session_id="sess-1",
            owner_id="owner-1",
            viewer_token=creds.viewer_token,
            write_token=creds.write_token,
            device_id="dev-1",
            expires_at_ns=10**30,
        )
        store.create_session(meta)

        store.append_raw_points("sess-1", [(100, 0.1, 0.2, 0.3)])

        points = store.read_raw_points("sess-1")
        latest = store.get_latest_inference("sess-1")

        self.assertEqual(len(points), 1)
        self.assertIsNone(latest)


if __name__ == "__main__":
    unittest.main()
