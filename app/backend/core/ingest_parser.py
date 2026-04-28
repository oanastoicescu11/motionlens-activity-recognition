"""Web DeviceMotion ingest parsing and validation utilities."""

from __future__ import annotations

from typing import Any, Mapping

from ..models.ingest import GyroValues, IngestBatch, MotionSample, XYZValues


MAX_SAMPLES_PER_BATCH = 512


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object.")
    return value


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _require_int_like(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"{field_name} must be an integer.")


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def _ms_to_ns(value_ms: int) -> int:
    return value_ms * 1_000_000


def parse_ingest_batch(raw: Mapping[str, Any]) -> IngestBatch:
    """Parse and validate a browser DeviceMotion ingest payload.

    Expected shape:
    - messageId/sessionId/deviceId at top-level
    - optional source field
    - samples[] items with timestampMs and acc{x,y,z}
    """

    msg = _require_mapping(raw, "body")
    message_id = _require_int(msg.get("messageId"), "messageId")
    session_id = _require_str(msg.get("sessionId"), "sessionId")
    device_id = _require_str(msg.get("deviceId"), "deviceId")
    placement_label_raw = msg.get("placementLabel")
    placement_label: str | None = None
    if placement_label_raw is not None:
        placement_label = _require_str(placement_label_raw, "placementLabel")

    source_raw = msg.get("source")
    if source_raw is not None:
        _require_str(source_raw, "source")

    samples_raw = msg.get("samples")
    if not isinstance(samples_raw, list):
        raise ValueError("samples must be an array.")
    if len(samples_raw) > MAX_SAMPLES_PER_BATCH:
        raise ValueError(f"samples exceeds max allowed per batch ({MAX_SAMPLES_PER_BATCH}).")

    samples: list[MotionSample] = []
    for index, entry in enumerate(samples_raw):
        sample = _require_mapping(entry, f"samples[{index}]")
        timestamp_ms = _require_int_like(sample.get("timestampMs"), f"samples[{index}].timestampMs")
        acc_raw = _require_mapping(sample.get("acc"), f"samples[{index}].acc")

        gyro_values: GyroValues | None = None
        gyro_raw = sample.get("gyro")
        if gyro_raw is not None:
            gyro_map = _require_mapping(gyro_raw, f"samples[{index}].gyro")
            gyro_values = GyroValues(
                alpha=_require_number(gyro_map.get("alpha"), f"samples[{index}].gyro.alpha"),
                beta=_require_number(gyro_map.get("beta"), f"samples[{index}].gyro.beta"),
                gamma=_require_number(gyro_map.get("gamma"), f"samples[{index}].gyro.gamma"),
            )

        samples.append(
            MotionSample(
                time_ns=_ms_to_ns(timestamp_ms),
                values=XYZValues(
                    x=_require_number(acc_raw.get("x"), f"samples[{index}].acc.x"),
                    y=_require_number(acc_raw.get("y"), f"samples[{index}].acc.y"),
                    z=_require_number(acc_raw.get("z"), f"samples[{index}].acc.z"),
                ),
                gyro=gyro_values,
            )
        )

    return IngestBatch(
        message_id=message_id,
        session_id=session_id,
        device_id=device_id,
        placement_label=placement_label,
        samples=tuple(samples),
    )


def extract_accelerometer_points(batch: IngestBatch) -> list[tuple[int, float, float, float]]:
    """Return accelerometer readings as (time_ns, x, y, z)."""

    return [
        (sample.time_ns, sample.values.x, sample.values.y, sample.values.z)
        for sample in batch.samples
    ]