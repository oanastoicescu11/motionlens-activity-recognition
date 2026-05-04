"""Training-specific accelerometer preprocessing — gap splitting, resampling, and multi-segment preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from core.preprocessing.accelerometer import (
    AccelerometerSegment,
    GravitySplit,
    _as_float_array,
    _empty_segment,
    _build_segment,
    _validate_shapes,
    _estimate_gravity_axis,
    apply_gravity_split,
    DEFAULT_TARGET_SAMPLE_RATE_HZ,
    DEFAULT_MAX_INTERPOLATION_GAP_SECONDS,
    DEFAULT_GRAVITY_CUTOFF_HZ,
    DEFAULT_GRAVITY_FILTER_ORDER,
)


@dataclass(frozen=True)
class PreprocessedAccelerometerSegment:
    """Canonical total, gravity, and body streams for one segment."""

    total: AccelerometerSegment
    gravity: AccelerometerSegment
    body: AccelerometerSegment
    segment_index: int


def sanitize_numeric_timeseries(
    time_seconds: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> AccelerometerSegment:
    """Sort, de-duplicate, and remove invalid numeric rows."""

    segment = _build_segment(time_seconds, x_values, y_values, z_values)
    _validate_shapes(segment)

    mask = (
        np.isfinite(segment.time_seconds)
        & np.isfinite(segment.acc_x)
        & np.isfinite(segment.acc_y)
        & np.isfinite(segment.acc_z)
    )
    if not np.any(mask):
        return _empty_segment()

    time_seconds = segment.time_seconds[mask]
    x_values = segment.acc_x[mask]
    y_values = segment.acc_y[mask]
    z_values = segment.acc_z[mask]

    if time_seconds.size < 2:
        if time_seconds.size:
            time_seconds = time_seconds - time_seconds[0]
        return _build_segment(time_seconds, x_values, y_values, z_values)

    order = np.argsort(time_seconds, kind="stable")
    time_seconds = time_seconds[order]
    x_values = x_values[order]
    y_values = y_values[order]
    z_values = z_values[order]

    unique_mask = np.ones(time_seconds.shape[0], dtype=bool)
    unique_mask[1:] = np.diff(time_seconds) > 0

    time_seconds = time_seconds[unique_mask]
    x_values = x_values[unique_mask]
    y_values = y_values[unique_mask]
    z_values = z_values[unique_mask]

    if time_seconds.size:
        time_seconds = time_seconds - time_seconds[0]

    return _build_segment(time_seconds, x_values, y_values, z_values)


def sanitize_timestamp_timeseries(
    timestamps: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
) -> AccelerometerSegment:
    """Convert raw timestamps to elapsed seconds and sanitize the series."""

    timestamp_values = np.asarray(timestamps)
    if np.issubdtype(timestamp_values.dtype, np.datetime64):
        time_seconds = timestamp_values.astype("datetime64[ns]").astype(np.int64) / 1_000_000_000.0
    else:
        time_seconds = timestamp_values.astype(np.float64)
    return sanitize_numeric_timeseries(time_seconds, x_values, y_values, z_values)


def split_timeseries_by_gaps(
    segment: AccelerometerSegment,
    max_gap_seconds: float = DEFAULT_MAX_INTERPOLATION_GAP_SECONDS,
) -> list[AccelerometerSegment]:
    """Split a sanitized segment wherever the source gap exceeds the threshold."""

    if max_gap_seconds <= 0.0:
        raise ValueError("max_gap_seconds must be positive.")

    _validate_shapes(segment)
    if len(segment) == 0:
        return []
    if len(segment) == 1:
        return [segment]

    split_points = np.flatnonzero(np.diff(segment.time_seconds) > max_gap_seconds) + 1
    if split_points.size == 0:
        return [segment]

    segments: list[AccelerometerSegment] = []
    start = 0
    for stop in (*split_points.tolist(), len(segment)):
        piece_time = segment.time_seconds[start:stop]
        if piece_time.size:
            segments.append(
                _build_segment(
                    piece_time - piece_time[0],
                    segment.acc_x[start:stop],
                    segment.acc_y[start:stop],
                    segment.acc_z[start:stop],
                )
            )
        start = stop
    return segments


def resample_accelerometer_segment(
    segment: AccelerometerSegment,
    target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ,
) -> AccelerometerSegment:
    """Resample a segment onto a fixed-rate time grid."""

    if target_sample_rate_hz <= 0.0:
        raise ValueError("target_sample_rate_hz must be positive.")

    _validate_shapes(segment)
    if len(segment) < 2:
        return segment

    last_time = float(segment.time_seconds[-1])
    if last_time <= 0.0:
        return _empty_segment()

    seconds_per_sample = 1.0 / target_sample_rate_hz
    target_time = np.arange(0.0, last_time + (seconds_per_sample * 0.5), seconds_per_sample)
    return _build_segment(
        target_time,
        np.interp(target_time, segment.time_seconds, segment.acc_x),
        np.interp(target_time, segment.time_seconds, segment.acc_y),
        np.interp(target_time, segment.time_seconds, segment.acc_z),
    )


def preprocess_sanitized_segment(
    segment: AccelerometerSegment,
    target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ,
    max_gap_seconds: float = DEFAULT_MAX_INTERPOLATION_GAP_SECONDS,
    gravity_cutoff_hz: float = DEFAULT_GRAVITY_CUTOFF_HZ,
    gravity_filter_order: int = DEFAULT_GRAVITY_FILTER_ORDER,
) -> list[PreprocessedAccelerometerSegment]:
    """Split on long gaps, resample, and derive gravity/body streams."""

    preprocessed_segments: list[PreprocessedAccelerometerSegment] = []
    for segment_index, source_segment in enumerate(
        split_timeseries_by_gaps(segment, max_gap_seconds=max_gap_seconds)
    ):
        total_segment = resample_accelerometer_segment(
            source_segment,
            target_sample_rate_hz=target_sample_rate_hz,
        )
        gravity_split = apply_gravity_split(
            total_segment,
            sample_rate_hz=target_sample_rate_hz,
            cutoff_hz=gravity_cutoff_hz,
            filter_order=gravity_filter_order,
        )
        preprocessed_segments.append(
            PreprocessedAccelerometerSegment(
                total=total_segment,
                gravity=_build_segment(
                    total_segment.time_seconds,
                    gravity_split.gravity_x,
                    gravity_split.gravity_y,
                    gravity_split.gravity_z,
                ),
                body=_build_segment(
                    total_segment.time_seconds,
                    gravity_split.body_x,
                    gravity_split.body_y,
                    gravity_split.body_z,
                ),
                segment_index=segment_index,
            )
        )
    return preprocessed_segments


def preprocess_numeric_timeseries(
    time_seconds: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ,
    max_gap_seconds: float = DEFAULT_MAX_INTERPOLATION_GAP_SECONDS,
    gravity_cutoff_hz: float = DEFAULT_GRAVITY_CUTOFF_HZ,
    gravity_filter_order: int = DEFAULT_GRAVITY_FILTER_ORDER,
) -> list[PreprocessedAccelerometerSegment]:
    """Sanitize a numeric series and convert it into canonical segments."""

    segment = sanitize_numeric_timeseries(time_seconds, x_values, y_values, z_values)
    return preprocess_sanitized_segment(
        segment,
        target_sample_rate_hz=target_sample_rate_hz,
        max_gap_seconds=max_gap_seconds,
        gravity_cutoff_hz=gravity_cutoff_hz,
        gravity_filter_order=gravity_filter_order,
    )


def preprocess_timestamp_timeseries(
    timestamps: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    target_sample_rate_hz: float = DEFAULT_TARGET_SAMPLE_RATE_HZ,
    max_gap_seconds: float = DEFAULT_MAX_INTERPOLATION_GAP_SECONDS,
    gravity_cutoff_hz: float = DEFAULT_GRAVITY_CUTOFF_HZ,
    gravity_filter_order: int = DEFAULT_GRAVITY_FILTER_ORDER,
) -> list[PreprocessedAccelerometerSegment]:
    """Sanitize a timestamped series and convert it into canonical segments."""

    segment = sanitize_timestamp_timeseries(timestamps, x_values, y_values, z_values)
    return preprocess_sanitized_segment(
        segment,
        target_sample_rate_hz=target_sample_rate_hz,
        max_gap_seconds=max_gap_seconds,
        gravity_cutoff_hz=gravity_cutoff_hz,
        gravity_filter_order=gravity_filter_order,
    )


__all__ = [
    "AccelerometerSegment",
    "GravitySplit",
    "PreprocessedAccelerometerSegment",
    "apply_gravity_split",
    "preprocess_numeric_timeseries",
    "preprocess_sanitized_segment",
    "preprocess_timestamp_timeseries",
    "resample_accelerometer_segment",
    "sanitize_numeric_timeseries",
    "sanitize_timestamp_timeseries",
    "split_timeseries_by_gaps",
]
