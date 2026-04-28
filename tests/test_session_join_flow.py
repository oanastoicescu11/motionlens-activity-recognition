import unittest

from app.backend.storage.memory import InMemoryStore
from app.backend.session_routes import join_hybrid_session, start_hybrid_session


class SessionJoinFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryStore()

    def test_desktop_start_then_join_attaches_writer(self) -> None:
        started = start_hybrid_session(
            self.store,
            owner_id="owner-1",
            mode="desktop",
            ttl_seconds=300,
            join_ttl_seconds=90,
        )
        self.assertEqual(started["mode"], "desktop")
        self.assertIn("join_token", started)
        self.assertNotIn("write_token", started)

        joined = join_hybrid_session(
            self.store,
            join_token=started["join_token"],
            device_id="iphone-1",
        )
        self.assertEqual(joined["session_id"], started["session_id"])
        self.assertIn("write_token", joined)
        self.assertEqual(joined["viewer_token"], started["viewer_token"])

        # join token is one-time use
        with self.assertRaises(PermissionError):
            join_hybrid_session(
                self.store,
                join_token=started["join_token"],
                device_id="iphone-2",
            )

    def test_join_rejects_invalid_token(self) -> None:
        started = start_hybrid_session(
            self.store,
            owner_id="owner-1",
            mode="desktop",
        )

        with self.assertRaises(PermissionError):
            join_hybrid_session(
                self.store,
                join_token=f"{started['join_token']}-invalid",
                device_id="iphone-1",
            )

    def test_mobile_start_returns_writer_immediately(self) -> None:
        started = start_hybrid_session(
            self.store,
            owner_id="owner-1",
            mode="mobile",
            device_id="iphone-1",
        )
        self.assertEqual(started["mode"], "mobile")
        self.assertIn("write_token", started)
        self.assertIn("viewer_token", started)


if __name__ == "__main__":
    unittest.main()
