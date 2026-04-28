"""In-memory session store — used for tests and single-process local dev."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Mapping

from ..models.session import SessionMeta


class InMemoryStore:
    """Thread-safe (GIL-protected) in-memory store with Redis-like key semantics."""

    def __init__(
        self,
        *,
        max_raw_points: int = 6000,
        max_queue_tasks: int = 2000,
        sweep_interval_seconds: float = 10.0,
    ) -> None:
        self._sessions: dict[str, SessionMeta] = {}
        self._viewer_to_session: dict[str, str] = {}
        self._write_to_session: dict[str, str] = {}
        self._dedupe: set[tuple[str, str, int]] = set()
        self._raw: dict[str, deque[tuple[int, float, float, float]]] = {}
        self._queue: deque[dict[str, Any]] = deque(maxlen=max(1, int(max_queue_tasks)))
        self._latest_inference: dict[str, dict[str, Any]] = {}
        self._decoder_state: dict[str, dict[str, Any]] = {}
        self._join_tokens: dict[str, tuple[str, int]] = {}
        self._max_raw_points = int(max_raw_points)
        self._sweep_interval_ns = int(max(0.5, float(sweep_interval_seconds)) * 1_000_000_000)
        self._last_sweep_ns = 0

    # ------------------------------------------------------------------
    # Internal sweep
    # ------------------------------------------------------------------

    def _sweep_expired(self, *, now_ns: int | None = None, force: bool = False) -> None:
        current_ns = time.time_ns() if now_ns is None else int(now_ns)
        if not force and (current_ns - self._last_sweep_ns) < self._sweep_interval_ns:
            return
        self._last_sweep_ns = current_ns

        expired_sessions = [
            sid for sid, meta in self._sessions.items()
            if current_ns >= int(meta.expires_at_ns)
        ]
        for sid in expired_sessions:
            meta = self._sessions.pop(sid, None)
            if meta is not None:
                self._viewer_to_session.pop(meta.viewer_token, None)
                if meta.write_token:
                    self._write_to_session.pop(meta.write_token, None)
            self._raw.pop(sid, None)
            self._latest_inference.pop(sid, None)
            self._decoder_state.pop(sid, None)
            self._dedupe = {item for item in self._dedupe if item[0] != sid}
            self._queue = deque(
                [task for task in self._queue if str(task.get("session_id", "")) != sid],
                maxlen=self._queue.maxlen,
            )

        expired_join_tokens = [
            token for token, (_, exp) in self._join_tokens.items()
            if current_ns >= int(exp)
        ]
        for token in expired_join_tokens:
            self._join_tokens.pop(token, None)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, meta: SessionMeta) -> None:
        self._sweep_expired()
        self._sessions[meta.session_id] = meta
        self._viewer_to_session[meta.viewer_token] = meta.session_id
        if meta.write_token:
            self._write_to_session[meta.write_token] = meta.session_id
        self._raw[meta.session_id] = deque(maxlen=self._max_raw_points)

    def get_session(self, session_id: str) -> SessionMeta | None:
        return self._sessions.get(session_id)

    def session_from_viewer_token(self, viewer_token: str) -> SessionMeta | None:
        session_id = self._viewer_to_session.get(viewer_token)
        return self._sessions.get(session_id) if session_id else None

    def session_from_write_token(self, write_token: str) -> SessionMeta | None:
        session_id = self._write_to_session.get(write_token)
        return self._sessions.get(session_id) if session_id else None

    def attach_writer(self, session_id: str, *, device_id: str, write_token: str) -> SessionMeta | None:
        self._sweep_expired()
        meta = self._sessions.get(session_id)
        if meta is None or not device_id or not write_token:
            return None
        if meta.write_token:
            self._write_to_session.pop(meta.write_token, None)
        meta.device_id = device_id
        meta.write_token = write_token
        self._write_to_session[write_token] = session_id
        return meta

    def expired(self, session_id: str, now_ns: int | None = None) -> bool:
        meta = self._sessions.get(session_id)
        if meta is None:
            return True
        current_ns = time.time_ns() if now_ns is None else int(now_ns)
        return current_ns >= meta.expires_at_ns

    # ------------------------------------------------------------------
    # Join tokens
    # ------------------------------------------------------------------

    def create_join_token(self, *, join_token: str, session_id: str, expires_at_ns: int) -> None:
        self._sweep_expired()
        self._join_tokens[join_token] = (session_id, int(expires_at_ns))

    def consume_join_token(self, join_token: str, *, now_ns: int | None = None) -> str | None:
        payload = self._join_tokens.pop(join_token, None)
        if payload is None:
            return None
        session_id, expires_at_ns = payload
        current_ns = time.time_ns() if now_ns is None else int(now_ns)
        return session_id if current_ns < expires_at_ns else None

    # ------------------------------------------------------------------
    # Deduplcation
    # ------------------------------------------------------------------

    def dedupe_seen(self, session_id: str, device_id: str, message_id: int) -> bool:
        self._sweep_expired()
        key = (session_id, device_id, int(message_id))
        if key in self._dedupe:
            return True
        self._dedupe.add(key)
        return False

    # ------------------------------------------------------------------
    # Raw signal points
    # ------------------------------------------------------------------

    def append_raw_points(
        self, session_id: str, points: list[tuple[int, float, float, float]]
    ) -> None:
        self._sweep_expired()
        if session_id not in self._raw:
            self._raw[session_id] = deque(maxlen=self._max_raw_points)
        self._raw[session_id].extend(points)

    def read_raw_points(
        self, session_id: str, limit: int = 1000
    ) -> list[tuple[int, float, float, float]]:
        values = self._raw.get(session_id)
        if not values:
            return []
        return list(values) if limit <= 0 else list(values)[-limit:]

    # ------------------------------------------------------------------
    # Inference queue
    # ------------------------------------------------------------------

    def enqueue_inference_task(self, task: dict[str, Any]) -> None:
        self._sweep_expired()
        self._queue.append(task)

    def pop_inference_task(self) -> dict[str, Any] | None:
        return self._queue.popleft() if self._queue else None

    # ------------------------------------------------------------------
    # Inference snapshots & decoder state
    # ------------------------------------------------------------------

    def set_latest_inference(self, session_id: str, snapshot: dict[str, Any]) -> None:
        self._latest_inference[session_id] = dict(snapshot)

    def get_latest_inference(self, session_id: str) -> dict[str, Any] | None:
        return self._latest_inference.get(session_id)

    def set_decoder_state(self, session_id: str, state: Mapping[str, Any] | None) -> None:
        if state is None:
            self._decoder_state.pop(session_id, None)
        else:
            self._decoder_state[session_id] = dict(state)

    def get_decoder_state(self, session_id: str) -> dict[str, Any] | None:
        value = self._decoder_state.get(session_id)
        return dict(value) if value is not None else None

    # ------------------------------------------------------------------
    # Message ordering
    # ------------------------------------------------------------------

    def update_last_message_id(self, session_id: str, message_id: int) -> None:
        meta = self._sessions.get(session_id)
        if meta is not None:
            meta.last_message_id = int(message_id)
