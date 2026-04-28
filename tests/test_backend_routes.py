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

            cls.app = create_app()
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


if __name__ == "__main__":
    unittest.main()
