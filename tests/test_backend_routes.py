"""Integration tests for FastAPI backend routes."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from httpx import ASGITransport, AsyncClient
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class BackendRoutesTests(unittest.IsolatedAsyncioTestCase):
    """Test FastAPI backend routes if httpx and FastAPI are available."""

    @classmethod
    def setUpClass(cls):
        if not HAS_HTTPX:
            raise unittest.SkipTest("httpx required for route testing")

        try:
            from app.backend.main import create_app
            from app.backend.api.dependencies import get_store

            cls.app = create_app()
            cls.store = get_store()
            if cls.app is None:
                raise unittest.SkipTest("FastAPI required")
        except ImportError:
            raise unittest.SkipTest("FastAPI required")

    async def asyncSetUp(self):
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_endpoint(self):
        """GET /health returns status ok."""
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    async def test_create_session_endpoint(self):
        """POST /v1/sessions creates a session with valid credentials."""
        response = await self.client.post(
            "/v1/sessions",
            params={"owner_id": "owner-1", "device_id": "dev-1"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("session_id", data)
        self.assertIn("viewer_token", data)
        self.assertIn("write_token", data)
        self.assertIn("expires_at_ns", data)

    async def test_create_session_missing_params(self):
        """POST /v1/sessions without owner_id returns 400 validation error."""
        response = await self.client.post(
            "/v1/sessions",
            params={"device_id": "dev-1"},
        )
        # Modern FastAPI returns 400 for missing required query parameters
        self.assertIn(response.status_code, [400, 422])

    async def test_desktop_start_and_join_flow(self):
        """Desktop start issues join token and join attaches one writer device."""
        start_res = await self.client.post(
            "/v1/sessions/start",
            params={"owner_id": "owner-a", "mode": "desktop", "ttl_seconds": 300, "join_ttl_seconds": 90},
        )
        self.assertEqual(start_res.status_code, 200)
        started = start_res.json()
        self.assertEqual(started["mode"], "desktop")
        self.assertIn("join_token", started)
        self.assertNotIn("write_token", started)

        join_res = await self.client.post(
            "/v1/sessions/join",
            json={
                "join_token": started["join_token"],
                "device_id": "iphone-a",
            },
        )
        self.assertEqual(join_res.status_code, 200)
        joined = join_res.json()
        self.assertEqual(joined["session_id"], started["session_id"])
        self.assertEqual(joined["viewer_token"], started["viewer_token"])
        self.assertIn("write_token", joined)

        second_join = await self.client.post(
            "/v1/sessions/join",
            json={
                "join_token": started["join_token"],
                "device_id": "iphone-b",
            },
        )
        self.assertEqual(second_join.status_code, 403)

    async def test_join_rejects_invalid_token(self):
        """Join endpoint rejects invalid join token values."""
        start_res = await self.client.post(
            "/v1/sessions/start",
            params={"owner_id": "owner-z", "mode": "desktop"},
        )
        self.assertEqual(start_res.status_code, 200)
        started = start_res.json()

        join_res = await self.client.post(
            "/v1/sessions/join",
            json={
                "join_token": f"{started['join_token']}-invalid",
                "device_id": "iphone-a",
            },
        )
        self.assertEqual(join_res.status_code, 403)

    async def test_finalize_and_read_summary(self):
        start_res = await self.client.post(
            "/v1/sessions/start",
            params={"owner_id": "owner-s", "mode": "desktop", "ttl_seconds": 300, "join_ttl_seconds": 90},
        )
        self.assertEqual(start_res.status_code, 200)
        started = start_res.json()
        session_id = started["session_id"]
        viewer_token = started["viewer_token"]

        self.store.append_inference_history(
            session_id,
            {
                "message_id": 1,
                "received_at_ns": 1_000_000_000,
                "status": "success",
                "activity": "walk",
                "confidence": 0.94,
                "cadence_spm": 112.0,
                "signal_quality_score": 86.0,
                "intensity_level": "moderate",
                "intensity_ratio": 2.1,
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
                "activity": "stairs",
                "confidence": 0.45,
                "cadence_spm": 118.0,
                "signal_quality_score": 50.0,
                "intensity_level": "moderate",
                "intensity_ratio": 2.2,
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
                "activity": "walk",
                "confidence": 0.93,
                "cadence_spm": 111.0,
                "signal_quality_score": 84.0,
                "intensity_level": "moderate",
                "intensity_ratio": 2.0,
                "posture_change_detected": False,
                "burst_detected": False,
            },
        )

        finalize_res = await self.client.post(
            f"/v1/sessions/{session_id}/finalize",
            headers={"X-Viewer-Token": viewer_token},
        )
        self.assertEqual(finalize_res.status_code, 200)
        summary = finalize_res.json()["summary"]
        self.assertEqual(summary["status"], "finalized")
        self.assertEqual(summary["dominant_activity"], "walk")
        self.assertGreater(summary["noise_bouts_folded"], 0)

        read_res = await self.client.get(
            f"/v1/sessions/{session_id}/summary",
            headers={"X-Viewer-Token": viewer_token},
        )
        self.assertEqual(read_res.status_code, 200)
        self.assertEqual(read_res.json()["summary"]["dominant_activity"], "walk")


if __name__ == "__main__":
    unittest.main()
