import unittest

from app.backend.storage.memory import InMemoryStore
from app.backend.session_routes import (
    finalize_session_summary,
    join_hybrid_session,
    read_session_summary,
    start_hybrid_session,
)


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

    def test_finalize_summary_marks_session_and_stores_denoised_recap(self) -> None:
        started = start_hybrid_session(
            self.store,
            owner_id="owner-1",
            mode="desktop",
            ttl_seconds=300,
            join_ttl_seconds=90,
        )

        session_id = started["session_id"]
        self.store.append_inference_history(
            session_id,
            {
                "message_id": 1,
                "received_at_ns": 1_000_000_000,
                "status": "success",
                "activity": "sit/lay",
                "confidence": 0.96,
                "cadence_spm": 0.0,
                "signal_quality_score": 88.0,
                "intensity_level": "still",
                "intensity_ratio": 1.0,
                "posture_change_detected": False,
                "burst_detected": False,
            },
        )
        self.store.append_inference_history(
            session_id,
            {
                "message_id": 2,
                "received_at_ns": 1_700_000_000,
                "status": "success",
                "activity": "walk",
                "confidence": 0.42,
                "cadence_spm": 103.0,
                "signal_quality_score": 44.0,
                "intensity_level": "moderate",
                "intensity_ratio": 2.0,
                "posture_change_detected": False,
                "burst_detected": False,
            },
        )
        self.store.append_inference_history(
            session_id,
            {
                "message_id": 3,
                "received_at_ns": 2_400_000_000,
                "status": "success",
                "activity": "sit/lay",
                "confidence": 0.95,
                "cadence_spm": 0.0,
                "signal_quality_score": 87.0,
                "intensity_level": "still",
                "intensity_ratio": 1.0,
                "posture_change_detected": False,
                "burst_detected": False,
            },
        )

        summary = finalize_session_summary(self.store, session_id=session_id, now_ns=5_000_000_000)

        self.assertEqual(summary["status"], "finalized")
        self.assertGreater(summary["noise_bouts_folded"], 0)
        self.assertGreater(summary["stationary_seconds"], summary["walking_seconds"])
        self.assertEqual(summary["dominant_activity"], "sit/lay")
        self.assertIsNotNone(read_session_summary(self.store, session_id=session_id))
        meta = self.store.get_session(session_id)
        self.assertIsNotNone(meta)
        assert meta is not None
        self.assertEqual(meta.finalized_at_ns, 5_000_000_000)

    def test_finalize_summary_waits_briefly_for_delayed_history(self) -> None:
        class DelayedHistoryStore(InMemoryStore):
            def __init__(self) -> None:
                super().__init__()
                self._history_reads = 0

            def read_inference_history(self, session_id: str, limit: int = 0) -> list[dict[str, object]]:
                self._history_reads += 1
                if self._history_reads == 1:
                    return []
                return super().read_inference_history(session_id, limit=limit)

        store = DelayedHistoryStore()
        started = start_hybrid_session(
            store,
            owner_id="owner-1",
            mode="desktop",
            ttl_seconds=300,
            join_ttl_seconds=90,
        )

        session_id = started["session_id"]
        store.append_raw_points(session_id, [(1_000_000_000, 0.1, 0.2, 9.8)])
        store.append_inference_history(
            session_id,
            {
                "message_id": 1,
                "received_at_ns": 1_000_000_000,
                "status": "success",
                "activity": "walk",
                "confidence": 0.96,
                "cadence_spm": 110.0,
                "signal_quality_score": 88.0,
                "intensity_level": "moderate",
                "intensity_ratio": 2.0,
                "posture_change_detected": False,
                "burst_detected": False,
            },
        )

        summary = finalize_session_summary(
            store,
            session_id=session_id,
            now_ns=5_000_000_000,
            history_wait_polls=2,
            history_wait_seconds=0.0,
        )

        self.assertEqual(summary["status"], "finalized")
        self.assertEqual(summary["dominant_activity"], "walk")

    def test_finalize_summary_uses_latest_snapshot_when_history_missing(self) -> None:
        started = start_hybrid_session(
            self.store,
            owner_id="owner-1",
            mode="desktop",
            ttl_seconds=300,
            join_ttl_seconds=90,
        )

        session_id = started["session_id"]
        self.store.append_raw_points(session_id, [(1_000_000_000, 0.1, 0.2, 9.8)])
        self.store.set_latest_inference(
            session_id,
            {
                "session_id": session_id,
                "message_id": 3,
                "status": "success",
                "activity": "walk",
                "confidence": 0.93,
                "cadence_spm": 112.0,
                "updated_at_ns": 2_000_000_000,
                "insights": {
                    "signal_quality_score": 86.0,
                    "intensity_level": "moderate",
                    "intensity_ratio": 2.1,
                    "posture_change_detected": False,
                    "burst_detected": False,
                },
            },
        )

        summary = finalize_session_summary(
            self.store,
            session_id=session_id,
            now_ns=5_000_000_000,
            history_wait_polls=0,
            history_wait_seconds=0.0,
        )

        self.assertEqual(summary["status"], "finalized")
        self.assertEqual(summary["dominant_activity"], "walk")
        self.assertGreater(summary["total_duration_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
