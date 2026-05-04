"""Ingest route handler.

Business logic lives in ``ingest_routes.process_ingest_batch``; this module
wires it to FastAPI.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...ingest_routes import process_ingest_batch
from ...storage import SessionStore

try:
    from fastapi import APIRouter, HTTPException
except Exception:  # pragma: no cover
    APIRouter = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]


def build_ingest_router(store: SessionStore):
    if APIRouter is None:
        return None  # pragma: no cover

    router = APIRouter(prefix="/v1", tags=["ingest"])

    @router.post("/ingest/{write_token}")
    def ingest(write_token: str, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return process_ingest_batch(store, write_token=write_token, raw_body=body)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
