"""Accelerometer preprocessing primitives shared by training and runtime inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


DEFAULT_TARGET_SAMPLE_RATE_HZ = 50
DEFAULT_MAX_INTERPOLATION_GAP_SECONDS = 0.25
DEFAULT_GRAVITY_CUTOFF_HZ = 0.3
DEFAULT_GRAVITY_FILTER_ORDER = 3


@dataclass(frozen=True)
class AccelerometerSegment:
    """One monotonic 3-axis accelerometer segment."""

    time_seconds: np.ndarray
    acc_x: np.ndarray
    acc_y: np.ndarray
    acc_z: np.ndarray

    def __len__(self) -> int:
        return int(self.time_seconds.size)


@dataclass(frozen=True)
class GravitySplit:
    """Per-axis gravity estimate and residual body acceleration."""

    gravity_x: np.ndarray
    gravity_y: np.ndarray
    gravity_z: np.ndarray
    body_x: np.ndarray
    body_y: np.ndarray
    body_z: np.ndarray


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _empty_segment() -> AccelerometerSegment:
    empty = np.empty(0, dtype=np.float64)
    return AccelerometerSegment(
        time_seconds=empty,
        acc_x=empty.copy(),
        acc_y=empty.copy(),
        acc_z=empty.copy(),
    )


def _build_segment(
    time_seconds: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> AccelerometerSegment:
    return AccelerometerSegment(
        time_seconds=_as_float_array(time_seconds),
        acc_x=_as_float_array(x_values),
        acc_y=_as_float_array(y_values),
        acc_z=_as_float_array(z_values),
    )


def _validate_shapes(segment: AccelerometerSegment) -> None:
    expected_shape = segment.time_seconds.shape
    if (
        segment.acc_x.shape != expected_shape
        or segment.acc_y.shape != expected_shape
        or segment.acc_z.shape != expected_shape
    ):
        raise ValueError("Time and acceleration arrays must share the same shape.")


def _estimate_gravity_axis(
    axis_values: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
    filter_order: int,
) -> np.ndarray:
    if axis_values.size == 0:
        return axis_values

    if axis_values.size < 3:
        return np.full_like(axis_values, axis_values.mean())

    sos = signal.butter(
        filter_order,
        cutoff_hz,
        btype="lowpass",
        fs=sample_rate_hz,
        output="sos",
    )
    try:
        return signal.sosfiltfilt(sos, axis_values)
    except ValueError:
        # Very short segments cannot support zero-phase padding. Fall back to
        # a constant gravity estimate rather than introducing phase distortion.
        return np.full_like(axis_values, axis_values.mean())


def apply_gravity_split(
    segment: AccelerometerSegment,
    sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ,
    cutoff_hz: float = DEFAULT_GRAVITY_CUTOFF_HZ,
    filter_order: int = DEFAULT_GRAVITY_FILTER_ORDER,
) -> GravitySplit:
    """Estimate gravity with a low-pass filter and derive body acceleration."""

    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if cutoff_hz <= 0.0 or cutoff_hz >= (sample_rate_hz / 2.0):
        raise ValueError("cutoff_hz must be between 0 and the Nyquist frequency.")
    if filter_order < 1:
        raise ValueError("filter_order must be at least 1.")

    _validate_shapes(segment)
    gravity_x = _estimate_gravity_axis(segment.acc_x, sample_rate_hz, cutoff_hz, filter_order)
    gravity_y = _estimate_gravity_axis(segment.acc_y, sample_rate_hz, cutoff_hz, filter_order)
    gravity_z = _estimate_gravity_axis(segment.acc_z, sample_rate_hz, cutoff_hz, filter_order)
    return GravitySplit(
        gravity_x=gravity_x,
        gravity_y=gravity_y,
        gravity_z=gravity_z,
        body_x=segment.acc_x - gravity_x,
        body_y=segment.acc_y - gravity_y,
        body_z=segment.acc_z - gravity_z,
    )


__all__ = [
    "AccelerometerSegment",
    "GravitySplit",
    "apply_gravity_split",
    "DEFAULT_GRAVITY_CUTOFF_HZ",
    "DEFAULT_GRAVITY_FILTER_ORDER",
    "DEFAULT_MAX_INTERPOLATION_GAP_SECONDS",
    "DEFAULT_TARGET_SAMPLE_RATE_HZ",
]
