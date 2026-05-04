"""Test that write tokens are bound to specific session + device pairs."""

import unittest

from app.backend.core.auth import issue_session_credentials
from app.backend.ingest_routes import process_ingest_batch
from app.backend.models.session import SessionMeta
from app.backend.storage.memory import InMemoryStore


def _batch(session_id: str, device_id: str, message_id: int = 1) -> dict:
    """Create a browser DeviceMotion batch payload."""
    return {
        "source": "web-devicemotion",
        "messageId": message_id,
        "sessionId": session_id,
        "deviceId": device_id,
        "samples": [
            {
                "timestampMs": 100,
                "acc": {"x": 1.0, "y": 2.0, "z": 3.0},
            }
        ],
    }


class DeviceTokenBindingTests(unittest.TestCase):
    """Verify write tokens are cryptographically bound to session + device."""

    def setUp(self) -> None:
        self.store = InMemoryStore()

    def test_write_token_bound_to_session_id(self):
        """Write token only works for its assigned session, not any other session."""
        # Session A with device 1
        creds_a = issue_session_credentials()
        meta_a = SessionMeta(
            session_id="sess-a",
            owner_id="owner-a",
            viewer_token=creds_a.viewer_token,
            write_token=creds_a.write_token,
            device_id="dev-1",
            expires_at_ns=10**30,
        )
        self.store.create_session(meta_a)

        # Session B with device 1
        creds_b = issue_session_credentials()
        meta_b = SessionMeta(
            session_id="sess-b",
            owner_id="owner-b",
            viewer_token=creds_b.viewer_token,
            write_token=creds_b.write_token,
            device_id="dev-1",
            expires_at_ns=10**30,
        )
        self.store.create_session(meta_b)

        # Try to use token from session A to ingest for session B
        with self.assertRaises(PermissionError) as ctx:
            process_ingest_batch(
                self.store,
                write_token=meta_a.write_token,
                raw_body=_batch("sess-b", "dev-1"),  # Wrong session
                now_ns=1000,
            )
        self.assertIn("sessionId does not match", str(ctx.exception))

    def test_write_token_bound_to_device_id(self):
        """Write token only works for its assigned device, not any other device."""
        # Session with device A
        creds_a = issue_session_credentials()
        meta_a = SessionMeta(
            session_id="sess-1",
            owner_id="owner",
            viewer_token=creds_a.viewer_token,
            write_token=creds_a.write_token,
            device_id="dev-a",
            expires_at_ns=10**30,
        )
        self.store.create_session(meta_a)

        # Try to use token to ingest from device B
        with self.assertRaises(PermissionError) as ctx:
            process_ingest_batch(
                self.store,
                write_token=meta_a.write_token,
                raw_body=_batch("sess-1", "dev-b"),  # Wrong device
                now_ns=1000,
            )
        self.assertIn("deviceId does not match", str(ctx.exception))

    def test_write_token_only_accepts_matching_session_and_device_pair(self):
        """Write token ONLY accepts both matching session AND matching device."""
        creds = issue_session_credentials()
        meta = SessionMeta(
            session_id="sess-xyz",
            owner_id="owner",
            viewer_token=creds.viewer_token,
            write_token=creds.write_token,
            device_id="iphone-12",
            expires_at_ns=10**30,
        )
        self.store.create_session(meta)

        # Correct session + device: should work
        result = process_ingest_batch(
            self.store,
            write_token=meta.write_token,
            raw_body=_batch("sess-xyz", "iphone-12"),
            now_ns=1000,
        )
        self.assertEqual(result["accepted_points"], 1)

        # Wrong session, right device: should fail
        with self.assertRaises(PermissionError) as ctx:
            process_ingest_batch(
                self.store,
                write_token=meta.write_token,
                raw_body=_batch("sess-wrong", "iphone-12"),
                now_ns=1000,
            )
        self.assertIn("sessionId does not match", str(ctx.exception))

        # Right session, wrong device: should fail
        with self.assertRaises(PermissionError) as ctx:
            process_ingest_batch(
                self.store,
                write_token=meta.write_token,
                raw_body=_batch("sess-xyz", "android-phone"),
                now_ns=1000,
            )
        self.assertIn("deviceId does not match", str(ctx.exception))

    def test_multiple_devices_each_get_separate_sessions_and_tokens(self):
        """Each device gets its own session with its own tokens (not shared sessions)."""
        owner = "owner-1"

        # Device A gets session 1
        creds_a = issue_session_credentials()
        meta_a = SessionMeta(
            session_id="sess-a",
            owner_id=owner,
            viewer_token=creds_a.viewer_token,
            write_token=creds_a.write_token,
            device_id="phone-a",
            expires_at_ns=10**30,
        )
        self.store.create_session(meta_a)

        # Device B gets session 2
        creds_b = issue_session_credentials()
        meta_b = SessionMeta(
            session_id="sess-b",
            owner_id=owner,
            viewer_token=creds_b.viewer_token,
            write_token=creds_b.write_token,
            device_id="phone-b",
            expires_at_ns=10**30,
        )
        self.store.create_session(meta_b)

        # Device A can ingest with its token and session
        result_a = process_ingest_batch(
            self.store,
            write_token=meta_a.write_token,
            raw_body=_batch("sess-a", "phone-a"),
            now_ns=1000,
        )
        self.assertEqual(result_a["accepted_points"], 1)

        # Device B can ingest with its token and session
        result_b = process_ingest_batch(
            self.store,
            write_token=meta_b.write_token,
            raw_body=_batch("sess-b", "phone-b"),
            now_ns=1000,
        )
        self.assertEqual(result_b["accepted_points"], 1)

        # Device A CANNOT use device B's token (even in its own session)
        with self.assertRaises(PermissionError) as ctx:
            process_ingest_batch(
                self.store,
                write_token=meta_b.write_token,  # Token from device B
                raw_body=_batch("sess-a", "phone-a"),  # Ingest for device A
                now_ns=1000,
            )
        self.assertIn("sessionId does not match", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
