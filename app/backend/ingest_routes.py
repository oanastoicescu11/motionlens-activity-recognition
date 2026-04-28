"""Ingest business logic — ``process_ingest_batch`` used by routes and tests.

FastAPI route handlers live in ``api/routes/ingest.py``.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from .core.ingest_parser import extract_accelerometer_points, parse_ingest_batch
from .models.protocols import SessionStore


def process_ingest_batch(
    store: SessionStore,
    *,
    write_token: str,
    raw_body: Mapping[str, Any],
    now_ns: int | None = None,
) -> dict[str, Any]:
    """Process one ingest batch with device+session binding verification.
    
    Security Guarantee: write_token is bound to (session_id, device_id) pair.
    This function enforces that ingest batches must match both:
    1. Batch's sessionId must match write_token's session
    2. Batch's deviceId must match write_token's device
    
    Mismatch on either field → PermissionError (cross-session/device attacks blocked).
    """

    meta = store.session_from_write_token(write_token)
    if meta is None:
        raise PermissionError("Invalid writer token.")

    if store.expired(meta.session_id, now_ns=now_ns):
        raise PermissionError("Session expired.")

    batch = parse_ingest_batch(raw_body)

    if batch.session_id != meta.session_id:
        raise PermissionError("sessionId does not match writer token scope.")
    if batch.device_id != meta.device_id:
        raise PermissionError("deviceId does not match session device.")

    duplicate = store.dedupe_seen(batch.session_id, batch.device_id, batch.message_id)
    out_of_order = batch.message_id <= meta.last_message_id

    if duplicate:
        return {
            "accepted_points": 0,
            "dropped_duplicate": True,
            "out_of_order": out_of_order,
            "enqueued": False,
        }

    points = extract_accelerometer_points(batch)
    if points:
        store.append_raw_points(meta.session_id, points)

    store.update_last_message_id(meta.session_id, batch.message_id)

    enqueued = False
    if points:
        store.enqueue_inference_task(
            {
                "session_id": meta.session_id,
                "message_id": batch.message_id,
                "points_added": len(points),
                "placement_label": batch.placement_label,
                "received_at_ns": int(time.time_ns() if now_ns is None else now_ns),
            }
        )
        enqueued = True

    return {
        "accepted_points": len(points),
        "dropped_duplicate": False,
        "out_of_order": out_of_order,
        "enqueued": enqueued,
    }
