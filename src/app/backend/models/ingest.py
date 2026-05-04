"""Web DeviceMotion ingest data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class XYZValues:
    """Tri-axis sensor values."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class GyroValues:
    """Rotation-rate values from DeviceMotion (deg/s)."""

    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class MotionSample:
    """One browser DeviceMotion sample."""

    time_ns: int
    values: XYZValues
    gyro: GyroValues | None = None


@dataclass(frozen=True)
class IngestBatch:
    """Top-level validated ingest message."""

    message_id: int
    session_id: str
    device_id: str
    samples: tuple[MotionSample, ...]

