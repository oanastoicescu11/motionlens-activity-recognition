"""Worker queue consumer utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.backend.models.protocols import SessionStore
from app.worker.pipeline import process_task


_LOG_ENABLED = os.getenv("MLIVE_INFERENCE_LOG_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
_DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "output" / "live_inference" / "inference_snapshots.jsonl"
_LOG_PATH = Path(os.getenv("MLIVE_INFERENCE_LOG_PATH", str(_DEFAULT_LOG_PATH)))


def _serialize_snapshot(snapshot: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Serialize snapshot to dict for JSON logging and storage.

    Uses .model_dump(mode="json") for Pydantic models,
    passes through plain dicts for backward compatibility.
    """
    if isinstance(snapshot, BaseModel):
        return snapshot.model_dump(mode="json")
    return snapshot


def _append_inference_log(task: dict[str, Any], snapshot: BaseModel | dict[str, Any]) -> None:
    """Append one worker snapshot record to a JSONL log file (best effort)."""
    if not _LOG_ENABLED:
        return

    serialized = _serialize_snapshot(snapshot)
    record = {
        "session_id": task.get("session_id"),
        "message_id": task.get("message_id"),
        "received_at_ns": task.get("received_at_ns"),
        "snapshot": serialized,
    }
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        # Logging must never break inference processing.
        return


def consume_once(store: SessionStore) -> bool:
    """Consume one task if available and publish latest snapshot."""

    task = store.pop_inference_task()
    if task is None:
        return False

    snapshot = process_task(task, store=store)
    _append_inference_log(task, snapshot)
    session_id = str(task.get("session_id", ""))
    if session_id:
        # Serialize to dict for storage (backward compatible with SessionStore protocol)
        serialized = _serialize_snapshot(snapshot)
        store.set_latest_inference(session_id, serialized)
    return True
