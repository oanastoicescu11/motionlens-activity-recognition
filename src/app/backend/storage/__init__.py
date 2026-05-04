"""Storage package — in-memory and Redis implementations."""

from __future__ import annotations

import os

from .memory import InMemoryStore
from .redis_impl import RedisStore
from ..models.protocols import SessionStore

__all__ = ["InMemoryStore", "RedisStore", "SessionStore", "get_runtime_store"]


def get_runtime_store() -> SessionStore:
    """Create the runtime store from environment settings.

    - ``MLIVE_STORE_BACKEND=memory`` (default): in-process store for local dev / tests.
    - ``MLIVE_STORE_BACKEND=redis``: shared store for multi-process deployments.
    """
    backend = os.getenv("MLIVE_STORE_BACKEND", "memory").strip().lower()
    if backend == "redis":
        return RedisStore.from_env()
    if backend == "memory":
        return InMemoryStore(
            max_raw_points=int(os.getenv("MLIVE_MAX_RAW_POINTS", "6000")),
            max_queue_tasks=int(os.getenv("MLIVE_MAX_QUEUE_TASKS", "2000")),
        )
    raise ValueError(
        f"Unsupported MLIVE_STORE_BACKEND={backend!r}. Use 'memory' or 'redis'."
    )
