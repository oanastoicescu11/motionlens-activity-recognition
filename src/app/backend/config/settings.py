"""Centralised configuration read from environment variables.

All ``MLIVE_*`` environment variables are read here so the rest of the
codebase never calls ``os.getenv`` directly.
"""

from __future__ import annotations

import os


class Settings:
    """Runtime settings resolved from environment variables at import time."""

    def __init__(self) -> None:
        self.store_backend: str = os.getenv("MLIVE_STORE_BACKEND", "memory").strip().lower()
        self.redis_url: str = os.getenv("MLIVE_REDIS_URL", "redis://localhost:6379/0")
        self.redis_prefix: str = os.getenv("MLIVE_REDIS_PREFIX", "mlive")
        self.max_raw_points: int = int(os.getenv("MLIVE_MAX_RAW_POINTS", "6000"))
        self.max_queue_tasks: int = int(os.getenv("MLIVE_MAX_QUEUE_TASKS", "2000"))
        self.allow_insecure_local: bool = (
            os.getenv("MLIVE_ALLOW_INSECURE_LOCAL", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )


# Module-level singleton — imported everywhere as ``from .config import settings``.
settings = Settings()
