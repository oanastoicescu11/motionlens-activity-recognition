"""Session management route handlers.

Business logic lives in ``session_routes.py``; this module wires it to FastAPI.
"""

from __future__ import annotations

from typing import Any

from ...session_routes import (
    _authorize_viewer,
    create_session_record,
    finalize_session_summary,
    join_hybrid_session,
    read_session_summary,
    start_hybrid_session,
)
from ...storage import SessionStore

try:
    from fastapi import APIRouter, Header, HTTPException
except Exception:  # pragma: no cover
    APIRouter = None  # type: ignore[assignment]
    Header = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]


def build_sessions_router(store: SessionStore):
    if APIRouter is None:
        return None  # pragma: no cover

    router = APIRouter(prefix="/v1", tags=["sessions"])

    @router.post("/sessions/start")
    def start(
        owner_id: str,
        mode: str,
        device_id: str | None = None,
        ttl_seconds: int = 300,
        join_ttl_seconds: int = 90,
    ) -> dict[str, Any]:
        if not owner_id:
            raise HTTPException(status_code=400, detail="owner_id is required.")
        try:
            return start_hybrid_session(
                store,
                owner_id=owner_id,
                mode=mode,
                device_id=device_id,
                ttl_seconds=ttl_seconds,
                join_ttl_seconds=join_ttl_seconds,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/sessions/join")
    def join(body: dict[str, Any]) -> dict[str, Any]:
        join_token = str(body.get("join_token", ""))
        device_id = str(body.get("device_id", ""))
        if not join_token or not device_id:
            raise HTTPException(status_code=400, detail="join_token and device_id are required.")
        try:
            return join_hybrid_session(store, join_token=join_token, device_id=device_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @router.post("/sessions")
    def create(owner_id: str, device_id: str, ttl_seconds: int = 300) -> dict[str, Any]:
        if not owner_id or not device_id:
            raise HTTPException(status_code=400, detail="owner_id and device_id are required.")
        return create_session_record(
            store,
            owner_id=owner_id,
            device_id=device_id,
            ttl_seconds=ttl_seconds,
        )

    @router.get("/sessions/{session_id}/signal")
    def read_signal(
        session_id: str,
        viewer_token: str = Header(alias="X-Viewer-Token"),
        limit: int = 1000,
    ) -> dict[str, Any]:
        try:
            _authorize_viewer(store, viewer_token=viewer_token, session_id=session_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        points = store.read_raw_points(session_id, limit=limit)
        return {"session_id": session_id, "points": points}

    @router.get("/sessions/{session_id}/inference")
    def read_inference(
        session_id: str,
        viewer_token: str = Header(alias="X-Viewer-Token"),
    ) -> dict[str, Any]:
        try:
            _authorize_viewer(store, viewer_token=viewer_token, session_id=session_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        snapshot = store.get_latest_inference(session_id)
        return {"session_id": session_id, "inference": snapshot}

    @router.post("/sessions/{session_id}/finalize")
    def finalize(
        session_id: str,
        viewer_token: str = Header(alias="X-Viewer-Token"),
    ) -> dict[str, Any]:
        try:
            _authorize_viewer(store, viewer_token=viewer_token, session_id=session_id)
            summary = finalize_session_summary(store, session_id=session_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return {"session_id": session_id, "summary": summary}

    @router.get("/sessions/{session_id}/summary")
    def read_summary(
        session_id: str,
        viewer_token: str = Header(alias="X-Viewer-Token"),
    ) -> dict[str, Any]:
        try:
            _authorize_viewer(store, viewer_token=viewer_token, session_id=session_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return {"session_id": session_id, "summary": read_session_summary(store, session_id=session_id)}

    return router
