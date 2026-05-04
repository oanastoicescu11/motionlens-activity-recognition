"""Centralised configuration read from environment variables.

This module also loads a repo-root ``.env`` file when present so both the
FastAPI backend and the Streamlit UI resolve the same local/deployment
configuration without each component reimplementing env handling.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(int(default))).strip().lower() in _TRUTHY_VALUES


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _normalize_http_base_url(url: str, default: str) -> str:
    cleaned = (url or default).strip() or default
    return cleaned.rstrip("/")


load_dotenv(_REPO_ROOT / ".env", override=False)


class Settings:
    """Runtime settings resolved from environment variables at import time."""

    def __init__(self) -> None:
        self.store_backend: str = os.getenv("MLIVE_STORE_BACKEND", "memory").strip().lower()
        self.redis_url: str = os.getenv("MLIVE_REDIS_URL", "redis://localhost:6379/0")
        self.redis_prefix: str = os.getenv("MLIVE_REDIS_PREFIX", "mlive")
        self.max_raw_points: int = _env_int("MLIVE_MAX_RAW_POINTS", 6000)
        self.max_queue_tasks: int = _env_int("MLIVE_MAX_QUEUE_TASKS", 2000)
        self.max_history_entries: int = _env_int("MLIVE_MAX_HISTORY_ENTRIES", 2000)
        self.worker_poll_seconds: float = max(0.01, _env_float("MLIVE_WORKER_POLL_SECONDS", 0.05))
        self.backend_base_url: str = _normalize_http_base_url(
            os.getenv("MLIVE_BACKEND_BASE_URL", "http://localhost:8000"),
            default="http://localhost:8000",
        )
        self.public_backend_url: str = _normalize_http_base_url(
            os.getenv("MLIVE_PUBLIC_BACKEND_URL", self.backend_base_url),
            default=self.backend_base_url,
        )
        self.allow_insecure_local: bool = _env_bool("MLIVE_ALLOW_INSECURE_LOCAL", False)


# Module-level singleton — imported everywhere as ``from .config import settings``.
settings = Settings()
