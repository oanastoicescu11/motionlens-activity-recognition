"""FastAPI application entrypoint for live ingestion backend."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from .api.dependencies import set_store
from .api.routes.health import build_health_router
from .api.routes.ingest import build_ingest_router
from .api.routes.mobile import build_mobile_capture_router
from .api.routes.sessions import build_sessions_router
from .config.settings import settings
from .storage import get_runtime_store

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    Request = object  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


def _start_inline_worker(store, stop_event: threading.Event) -> threading.Thread:
    """Run inference queue drain in a daemon thread (memory-mode only)."""
    from app.worker.consumer import consume_once

    def _loop() -> None:
        while not stop_event.is_set():
            try:
                drained = consume_once(store)
            except Exception:
                LOGGER.exception("Inline worker loop failed while draining inference queue")
                drained = False
            if not drained:
                time.sleep(0.1)

    t = threading.Thread(target=_loop, name="inline-worker", daemon=True)
    t.start()
    return t


def create_app() -> "FastAPI":
    """Build backend application with HTTPS enforcement middleware."""

    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed. Install backend deps with: "
            "pip install fastapi uvicorn"
        )

    store = get_runtime_store()
    set_store(store)
    _stop_event = threading.Event()
    _worker_thread: threading.Thread | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal _worker_thread
        if settings.store_backend == "memory":
            _worker_thread = _start_inline_worker(store, _stop_event)
        yield
        _stop_event.set()

    app = FastAPI(title="MotionLens Live Backend", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def enforce_https(request: Request, call_next):
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        if proto != "https" and not settings.allow_insecure_local:
            return JSONResponse(status_code=426, content={"detail": "HTTPS is required."})
        return await call_next(request)

    for router_factory, kwargs in [
        (build_health_router, {}),
        (build_ingest_router, {"store": store}),
        (build_sessions_router, {"store": store}),
        (build_mobile_capture_router, {}),
    ]:
        router = router_factory(**kwargs)
        if router is not None:
            app.include_router(router)

    return app


app = create_app()
