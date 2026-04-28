"""Worker process entrypoint."""

from __future__ import annotations

import multiprocessing as mp
import time

from app.backend.storage import get_runtime_store
from app.worker.consumer import consume_once


def _worker_loop(poll_seconds: float = 0.1) -> None:
    store = get_runtime_store()
    while True:
        processed = consume_once(store)
        if not processed:
            time.sleep(poll_seconds)


def start_workers(num_workers: int = 2, poll_seconds: float = 0.1) -> list[mp.Process]:
    """Start multiprocessing worker processes."""

    processes: list[mp.Process] = []
    for _ in range(max(1, int(num_workers))):
        proc = mp.Process(target=_worker_loop, args=(poll_seconds,), daemon=True)
        proc.start()
        processes.append(proc)
    return processes
