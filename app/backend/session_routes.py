"""Session business logic — lifecycle helpers used by routes and tests.

FastAPI route handlers live in ``api/routes/sessions.py``.
"""

from __future__ import annotations

import secrets
import time
import uuid
from typing import Any

from .core.auth import issue_session_credentials
from .models.session import SessionMeta
from .models.protocols import SessionStore


MAX_SESSION_TTL_SECONDS = 30 * 60
MAX_JOIN_TTL_SECONDS = 5 * 60


def _clamp_ttl(value: int, *, max_seconds: int) -> int:
    return min(max(1, int(value)), int(max_seconds))


def create_session_record(
    store: SessionStore,
    *,
    owner_id: str,
    device_id: str,
    ttl_seconds: int = 1800,
) -> dict[str, Any]:
    """Create one isolated session with device-specific write token.
    
    The write_token is cryptographically bound to both:
    - session_id: Only this session can use the token
    - device_id: Only this device can use the token
    
    Ingest requests must provide matching session_id and device_id in the payload,
    or they will be rejected with PermissionError.
    """

    creds = issue_session_credentials()
    session_id = str(uuid.uuid4())
    expires_at_ns = int(time.time_ns() + int(ttl_seconds) * 1_000_000_000)
    meta = SessionMeta(
        session_id=session_id,
        owner_id=owner_id,
        viewer_token=creds.viewer_token,
        write_token=creds.write_token,
        device_id=device_id,
        expires_at_ns=expires_at_ns,
    )
    store.create_session(meta)
    return {
        "session_id": session_id,
        "viewer_token": creds.viewer_token,
        "write_token": creds.write_token,
        "expires_at_ns": expires_at_ns,
    }


def start_hybrid_session(
    store: SessionStore,
    *,
    owner_id: str,
    mode: str,
    device_id: str | None = None,
    ttl_seconds: int = 300,
    join_ttl_seconds: int = 90,
) -> dict[str, Any]:
    """Create a short-lived session for mobile-direct or desktop-QR flow."""

    session_id = str(uuid.uuid4())
    ttl_seconds = _clamp_ttl(ttl_seconds, max_seconds=MAX_SESSION_TTL_SECONDS)
    expires_at_ns = int(time.time_ns() + ttl_seconds * 1_000_000_000)

    if mode == "mobile":
        if not device_id:
            raise ValueError("device_id is required for mobile mode.")
        creds = issue_session_credentials()
        meta = SessionMeta(
            session_id=session_id,
            owner_id=owner_id,
            viewer_token=creds.viewer_token,
            write_token=creds.write_token,
            device_id=device_id,
            expires_at_ns=expires_at_ns,
        )
        store.create_session(meta)
        return {
            "mode": mode,
            "session_id": session_id,
            "viewer_token": creds.viewer_token,
            "write_token": creds.write_token,
            "expires_at_ns": expires_at_ns,
        }

    if mode == "desktop":
        viewer_token = issue_session_credentials().viewer_token
        # Writer is intentionally not active until phone joins.
        meta = SessionMeta(
            session_id=session_id,
            owner_id=owner_id,
            viewer_token=viewer_token,
            write_token="",
            device_id="",
            expires_at_ns=expires_at_ns,
        )
        store.create_session(meta)

        join_token = secrets.token_urlsafe(24)
        join_expires_at_ns = int(
            min(
                expires_at_ns,
                time.time_ns() + _clamp_ttl(join_ttl_seconds, max_seconds=MAX_JOIN_TTL_SECONDS) * 1_000_000_000,
            )
        )
        store.create_join_token(
            join_token=join_token,
            session_id=session_id,
            expires_at_ns=join_expires_at_ns,
        )
        return {
            "mode": mode,
            "session_id": session_id,
            "viewer_token": viewer_token,
            "join_token": join_token,
            "join_expires_at_ns": join_expires_at_ns,
            "expires_at_ns": expires_at_ns,
        }

    raise ValueError("mode must be either 'mobile' or 'desktop'.")


def join_hybrid_session(
    store: SessionStore,
    *,
    join_token: str,
    device_id: str,
) -> dict[str, Any]:
    """Consume one-time join token and attach a writer device to the session."""

    session_id = store.consume_join_token(join_token)
    if session_id is None:
        raise PermissionError("Invalid or expired join token.")

    meta = store.get_session(session_id)
    if meta is None:
        raise PermissionError("Session not found.")
    if store.expired(session_id):
        raise PermissionError("Session expired.")
    if meta.device_id:
        raise PermissionError("Session capture device already attached.")

    write_token = issue_session_credentials().write_token
    updated = store.attach_writer(
        session_id,
        device_id=device_id,
        write_token=write_token,
    )
    if updated is None:
        raise PermissionError("Unable to attach writer device.")

    return {
        "session_id": updated.session_id,
        "viewer_token": updated.viewer_token,
        "write_token": updated.write_token,
        "expires_at_ns": updated.expires_at_ns,
    }


def _authorize_viewer(store: SessionStore, viewer_token: str, session_id: str) -> SessionMeta:
    meta = store.session_from_viewer_token(viewer_token)
    if meta is None:
        raise PermissionError("Invalid viewer token.")
    if meta.session_id != session_id:
        raise PermissionError("Cross-session read is forbidden.")
    if store.expired(session_id):
        raise PermissionError("Session expired.")
    return meta
