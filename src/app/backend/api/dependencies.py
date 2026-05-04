"""FastAPI dependency providers."""

from __future__ import annotations

from ..storage import SessionStore, get_runtime_store

# Module-level singleton so all requests share one store instance.
_store: SessionStore | None = None


def get_store() -> SessionStore:
    """Return the runtime store singleton (lazy-initialised)."""
    global _store
    if _store is None:
        _store = get_runtime_store()
    return _store


def set_store(store: SessionStore) -> None:
    """Override the store singleton — used in tests and app startup."""
    global _store
    _store = store
