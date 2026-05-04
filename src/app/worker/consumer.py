"""Worker queue consumer utilities."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.backend.models.protocols import SessionStore
from app.worker.pipeline import process_task


def _serialize_snapshot(snapshot: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Serialize snapshot to dict for storage.

    Uses .model_dump(mode="json") for Pydantic models,
    passes through plain dicts for backward compatibility.
    """
    if isinstance(snapshot, BaseModel):
        return snapshot.model_dump(mode="json")
    return snapshot


def _build_history_entry(task: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """Build a compact per-window history entry for session recap generation."""
    if snapshot.get("status") != "success":
        return None

    insights = snapshot.get("insights") or {}
    return {
        "message_id": int(task.get("message_id", 0) or 0),
        "received_at_ns": int(task.get("received_at_ns", 0) or 0),
        "updated_at_ns": int(snapshot.get("updated_at_ns", 0) or 0),
        "window_start_ns": snapshot.get("window_start_ns"),
        "window_end_ns": snapshot.get("window_end_ns"),
        "status": snapshot.get("status", "unknown"),
        "activity": str(snapshot.get("activity", "unknown")),
        "confidence": float(snapshot.get("confidence", 0.0) or 0.0),
        "cadence_spm": float(snapshot.get("cadence_spm", 0.0) or 0.0),
        "periodicity_strength": float(snapshot.get("periodicity_strength", 0.0) or 0.0),
        "signal_quality_score": float(insights.get("signal_quality_score", 0.0) or 0.0),
        "intensity_level": str(insights.get("intensity_level", "unknown")),
        "intensity_ratio": float(insights.get("intensity_ratio", 0.0) or 0.0),
        "posture_change_detected": bool(insights.get("posture_change_detected", False)),
        "burst_detected": bool(insights.get("burst_detected", False)),
    }


def consume_once(store: SessionStore) -> bool:
    """Consume one task if available and publish latest snapshot."""

    task = store.pop_inference_task()
    if task is None:
        return False

    snapshot = process_task(task, store=store)
    session_id = str(task.get("session_id", ""))
    if session_id:
        # Serialize to dict for storage (backward compatible with SessionStore protocol)
        serialized = _serialize_snapshot(snapshot)
        store.set_latest_inference(session_id, serialized)
        history_entry = _build_history_entry(task, serialized)
        if history_entry is not None:
            store.append_inference_history(session_id, history_entry)
    return True
