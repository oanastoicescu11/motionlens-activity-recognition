"""Storage package — in-memory and Redis implementations."""

from __future__ import annotations

from .memory import InMemoryStore
from .redis_impl import RedisStore
from ..config import settings
from ..models.protocols import SessionStore

__all__ = ["InMemoryStore", "RedisStore", "SessionStore", "get_runtime_store"]


def get_runtime_store() -> SessionStore:
    """Create the runtime store from environment settings.

    - ``MLIVE_STORE_BACKEND=memory`` (default): in-process store for local dev / tests.
    - ``MLIVE_STORE_BACKEND=redis``: shared store for multi-process deployments.
    """
    backend = settings.store_backend
    if backend == "redis":
        return RedisStore.from_env()
    if backend == "memory":
        return InMemoryStore(
            max_raw_points=settings.max_raw_points,
            max_queue_tasks=settings.max_queue_tasks,
            max_history_entries=settings.max_history_entries,
        )
    raise ValueError(
        f"Unsupported MLIVE_STORE_BACKEND={backend!r}. Use 'memory' or 'redis'."
    )
