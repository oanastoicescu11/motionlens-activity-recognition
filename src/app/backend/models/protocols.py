"""Protocol definitions."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .session import SessionMeta


class SessionStore(Protocol):
    """Protocol shared by in-memory and Redis-backed stores."""

    def create_session(self, meta: SessionMeta) -> None:
        ...

    def get_session(self, session_id: str) -> SessionMeta | None:
        ...

    def session_from_viewer_token(self, viewer_token: str) -> SessionMeta | None:
        ...

    def session_from_write_token(self, write_token: str) -> SessionMeta | None:
        ...

    def attach_writer(self, session_id: str, *, device_id: str, write_token: str) -> SessionMeta | None:
        ...

    def finalize_session(self, session_id: str, *, finalized_at_ns: int) -> SessionMeta | None:
        ...

    def create_join_token(self, *, join_token: str, session_id: str, expires_at_ns: int) -> None:
        ...

    def consume_join_token(self, join_token: str, *, now_ns: int | None = None) -> str | None:
        ...

    def dedupe_seen(self, session_id: str, device_id: str, message_id: int) -> bool:
        ...

    def append_raw_points(self, session_id: str, points: list[tuple[int, float, float, float]]) -> None:
        ...

    def read_raw_points(self, session_id: str, limit: int = 1000) -> list[tuple[int, float, float, float]]:
        ...

    def enqueue_inference_task(self, task: dict[str, Any]) -> None:
        ...

    def pop_inference_task(self) -> dict[str, Any] | None:
        ...

    def set_latest_inference(self, session_id: str, snapshot: dict[str, Any]) -> None:
        ...

    def get_latest_inference(self, session_id: str) -> dict[str, Any] | None:
        ...

    def append_inference_history(self, session_id: str, snapshot: Mapping[str, Any]) -> None:
        ...

    def read_inference_history(self, session_id: str, limit: int = 0) -> list[dict[str, Any]]:
        ...

    def set_session_summary(self, session_id: str, summary: Mapping[str, Any] | None) -> None:
        ...

    def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        ...

    def set_decoder_state(self, session_id: str, state: Mapping[str, Any] | None) -> None:
        ...

    def get_decoder_state(self, session_id: str) -> dict[str, Any] | None:
        ...

    def update_last_message_id(self, session_id: str, message_id: int) -> None:
        ...

    def expired(self, session_id: str, now_ns: int | None = None) -> bool:
        ...

