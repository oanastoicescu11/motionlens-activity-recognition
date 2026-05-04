"""Worker process entrypoint."""

from __future__ import annotations

import multiprocessing as mp
import time

from app.backend.config import settings
from app.backend.storage import get_runtime_store
from app.worker.consumer import consume_once


DEFAULT_WORKER_POLL_SECONDS = 0.05


def _worker_loop(poll_seconds: float = DEFAULT_WORKER_POLL_SECONDS) -> None:
    store = get_runtime_store()
    while True:
        processed = consume_once(store)
        if not processed:
            time.sleep(poll_seconds)


def start_workers(num_workers: int = 2, poll_seconds: float = DEFAULT_WORKER_POLL_SECONDS) -> list[mp.Process]:
    """Start multiprocessing worker processes."""

    processes: list[mp.Process] = []
    for _ in range(max(1, int(num_workers))):
        proc = mp.Process(target=_worker_loop, args=(poll_seconds,), daemon=True)
        proc.start()
        processes.append(proc)
    return processes


def worker_poll_seconds_from_env(default: float = DEFAULT_WORKER_POLL_SECONDS) -> float:
    """Read worker poll cadence from environment with a safe lower bound."""
    configured = settings.worker_poll_seconds
    if configured > 0.0:
        return max(0.01, configured)
    return max(0.01, float(default))
