"""Accelerometer preprocessing primitives shared by training and runtime inference."""

from .accelerometer import (
    AccelerometerSegment,
    GravitySplit,
    apply_gravity_split,
)

__all__ = [
    "AccelerometerSegment",
    "GravitySplit",
    "apply_gravity_split",
]
