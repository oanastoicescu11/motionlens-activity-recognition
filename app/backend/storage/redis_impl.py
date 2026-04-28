"""Redis-backed session store — used in multi-process production deployments."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from ..models.session import SessionMeta

try:
    import redis as _redis_module
except Exception:  # pragma: no cover
    _redis_module = None  # type: ignore[assignment]


class RedisStore:
    """Redis-backed implementation for backend + worker across multiple processes."""

    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = "mlive",
        max_raw_points: int = 6000,
        max_queue_tasks: int = 2000,
    ) -> None:
        self._client = client
        self._prefix = key_prefix.strip() or "mlive"
        self._max_raw_points = int(max_raw_points)
        self._max_queue_tasks = max(1, int(max_queue_tasks))

    @classmethod
    def from_env(cls) -> "RedisStore":
        if _redis_module is None:
            raise RuntimeError(
                "Redis backend requested but 'redis' package is not installed. "
                "Install with: pip install redis"
            )
        import os
        redis_url = os.getenv("MLIVE_REDIS_URL", "redis://localhost:6379/0")
        key_prefix = os.getenv("MLIVE_REDIS_PREFIX", "mlive")
        max_raw_points = int(os.getenv("MLIVE_MAX_RAW_POINTS", "6000"))
        max_queue_tasks = int(os.getenv("MLIVE_MAX_QUEUE_TASKS", "2000"))

        client = _redis_module.Redis.from_url(redis_url, decode_responses=True)
        try:
            client.ping()
        except Exception as exc:
            raise RuntimeError(f"Unable to connect to Redis at {redis_url}: {exc}") from exc

        return cls(
            client=client,
            key_prefix=key_prefix,
            max_raw_points=max_raw_points,
            max_queue_tasks=max_queue_tasks,
        )

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _key(self, suffix: str) -> str:
        return f"{self._prefix}:{suffix}"

    def _session_key(self, session_id: str) -> str:
        return self._key(f"session:{session_id}")

    def _viewer_token_key(self, viewer_token: str) -> str:
        return self._key(f"token:viewer:{viewer_token}")

    def _write_token_key(self, write_token: str) -> str:
        return self._key(f"token:write:{write_token}")

    def _join_token_key(self, join_token: str) -> str:
        return self._key(f"token:join:{join_token}")

    def _dedupe_key(self, session_id: str, device_id: str) -> str:
        return self._key(f"dedupe:{session_id}:{device_id}")

    def _raw_key(self, session_id: str) -> str:
        return self._key(f"raw:{session_id}")

    def _queue_key(self) -> str:
        return self._key("queue:inference")

    def _latest_inference_key(self, session_id: str) -> str:
        return self._key(f"inference:{session_id}")

    def _decoder_state_key(self, session_id: str) -> str:
        return self._key(f"decoder:{session_id}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_ttl_seconds(self, expires_at_ns: int) -> int:
        return max(int((int(expires_at_ns) - time.time_ns()) / 1_000_000_000), 1)

    def _decode_session_meta(self, payload: dict[str, str] | None) -> SessionMeta | None:
        if not payload:
            return None
        try:
            return SessionMeta(
                session_id=str(payload["session_id"]),
                owner_id=str(payload["owner_id"]),
                viewer_token=str(payload["viewer_token"]),
                write_token=str(payload["write_token"]),
                device_id=str(payload["device_id"]),
                expires_at_ns=int(payload["expires_at_ns"]),
                last_message_id=int(payload.get("last_message_id", "-1")),
            )
        except Exception:
            return None

    def _get_session_meta(self, session_id: str) -> SessionMeta | None:
        return self._decode_session_meta(self._client.hgetall(self._session_key(session_id)))

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, meta: SessionMeta) -> None:
        ttl = self._session_ttl_seconds(meta.expires_at_ns)
        payload = {
            "session_id": meta.session_id,
            "owner_id": meta.owner_id,
            "viewer_token": meta.viewer_token,
            "write_token": meta.write_token,
            "device_id": meta.device_id,
            "expires_at_ns": str(meta.expires_at_ns),
            "last_message_id": str(meta.last_message_id),
        }
        self._client.hset(self._session_key(meta.session_id), mapping=payload)
        self._client.expire(self._session_key(meta.session_id), ttl)
        self._client.setex(self._viewer_token_key(meta.viewer_token), ttl, meta.session_id)
        if meta.write_token:
            self._client.setex(self._write_token_key(meta.write_token), ttl, meta.session_id)

    def get_session(self, session_id: str) -> SessionMeta | None:
        return self._get_session_meta(session_id)

    def session_from_viewer_token(self, viewer_token: str) -> SessionMeta | None:
        session_id = self._client.get(self._viewer_token_key(viewer_token))
        return self._get_session_meta(str(session_id)) if session_id else None

    def session_from_write_token(self, write_token: str) -> SessionMeta | None:
        session_id = self._client.get(self._write_token_key(write_token))
        return self._get_session_meta(str(session_id)) if session_id else None

    def attach_writer(self, session_id: str, *, device_id: str, write_token: str) -> SessionMeta | None:
        if not device_id or not write_token:
            return None
        meta = self._get_session_meta(session_id)
        if meta is None:
            return None
        ttl = self._session_ttl_seconds(meta.expires_at_ns)
        if meta.write_token:
            self._client.delete(self._write_token_key(meta.write_token))
        self._client.hset(
            self._session_key(session_id),
            mapping={"device_id": device_id, "write_token": write_token},
        )
        self._client.setex(self._write_token_key(write_token), ttl, session_id)
        return self._get_session_meta(session_id)

    def expired(self, session_id: str, now_ns: int | None = None) -> bool:
        meta = self._get_session_meta(session_id)
        if meta is None:
            return True
        current_ns = time.time_ns() if now_ns is None else int(now_ns)
        return current_ns >= meta.expires_at_ns

    # ------------------------------------------------------------------
    # Join tokens
    # ------------------------------------------------------------------

    def create_join_token(self, *, join_token: str, session_id: str, expires_at_ns: int) -> None:
        ttl = self._session_ttl_seconds(expires_at_ns)
        self._client.setex(self._join_token_key(join_token), ttl, session_id)

    def consume_join_token(self, join_token: str, *, now_ns: int | None = None) -> str | None:
        value = self._client.execute_command("GETDEL", self._join_token_key(join_token))
        return str(value) if value else None

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def dedupe_seen(self, session_id: str, device_id: str, message_id: int) -> bool:
        key = self._dedupe_key(session_id, device_id)
        added = int(self._client.sadd(key, str(int(message_id))))
        self._client.expire(key, max(3600, 6 * 3600))
        return added == 0

    # ------------------------------------------------------------------
    # Raw signal points
    # ------------------------------------------------------------------

    def append_raw_points(
        self, session_id: str, points: list[tuple[int, float, float, float]]
    ) -> None:
        if not points:
            return
        key = self._raw_key(session_id)
        encoded = [json.dumps([int(t), float(x), float(y), float(z)]) for t, x, y, z in points]
        self._client.rpush(key, *encoded)
        self._client.ltrim(key, -self._max_raw_points, -1)
        meta = self._get_session_meta(session_id)
        if meta is not None:
            self._client.expire(key, self._session_ttl_seconds(meta.expires_at_ns))

    def read_raw_points(
        self, session_id: str, limit: int = 1000
    ) -> list[tuple[int, float, float, float]]:
        key = self._raw_key(session_id)
        raw_values = (
            self._client.lrange(key, 0, -1)
            if limit <= 0
            else self._client.lrange(key, -int(limit), -1)
        )
        out: list[tuple[int, float, float, float]] = []
        for raw in raw_values:
            try:
                t, x, y, z = json.loads(raw)
                out.append((int(t), float(x), float(y), float(z)))
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------
    # Inference queue
    # ------------------------------------------------------------------

    def enqueue_inference_task(self, task: dict[str, Any]) -> None:
        queue_key = self._queue_key()
        if int(self._client.llen(queue_key)) >= self._max_queue_tasks:
            self._client.lpop(queue_key)
        self._client.rpush(queue_key, json.dumps(task))

    def pop_inference_task(self) -> dict[str, Any] | None:
        raw = self._client.lpop(self._queue_key())
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    # ------------------------------------------------------------------
    # Inference snapshots & decoder state
    # ------------------------------------------------------------------

    def set_latest_inference(self, session_id: str, snapshot: dict[str, Any]) -> None:
        meta = self._get_session_meta(session_id)
        if meta is None:
            return
        self._client.setex(
            self._latest_inference_key(session_id),
            self._session_ttl_seconds(meta.expires_at_ns),
            json.dumps(snapshot),
        )

    def get_latest_inference(self, session_id: str) -> dict[str, Any] | None:
        raw = self._client.get(self._latest_inference_key(session_id))
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def set_decoder_state(self, session_id: str, state: Mapping[str, Any] | None) -> None:
        key = self._decoder_state_key(session_id)
        if state is None:
            self._client.delete(key)
            return
        meta = self._get_session_meta(session_id)
        if meta is None:
            return
        self._client.setex(
            key,
            self._session_ttl_seconds(meta.expires_at_ns),
            json.dumps(dict(state)),
        )

    def get_decoder_state(self, session_id: str) -> dict[str, Any] | None:
        raw = self._client.get(self._decoder_state_key(session_id))
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    # ------------------------------------------------------------------
    # Message ordering
    # ------------------------------------------------------------------

    def update_last_message_id(self, session_id: str, message_id: int) -> None:
        self._client.hset(
            self._session_key(session_id),
            mapping={"last_message_id": str(int(message_id))},
        )
