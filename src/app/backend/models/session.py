"""Session data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionMeta:
    """Session metadata tracked by the backend."""

    session_id: str
    owner_id: str
    viewer_token: str
    write_token: str
    device_id: str
    expires_at_ns: int
    last_message_id: int = -1
    finalized_at_ns: int = 0