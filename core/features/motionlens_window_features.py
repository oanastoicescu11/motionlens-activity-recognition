"""MotionLens window feature extraction shared by training and streaming."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pywt as _pywt
from scipy.signal import find_peaks as _find_peaks
from scipy.signal import stft as _compute_stft

from core.preprocessing import AccelerometerSegment, apply_gravity_split


AXES = ("x", "y", "z")


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0))


def _safe_skew(values: np.ndarray) -> float:
    std = _safe_std(values)
    if std < 1e-8:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean((centered / std) ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    std = _safe_std(values)
    if std < 1e-8:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean((centered / std) ** 4) - 3.0)


def _median_abs_dev(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _zero_crossing_rate(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.mean(np.diff(np.signbit(values)) != 0))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return 0.0
    if _safe_std(a) < 1e-8 or _safe_std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _band_energy(power: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def _spectral_entropy(power: np.ndarray) -> float:
    total_power = float(np.sum(power))
    if total_power <= 0.0:
        return 0.0
    probs = power / total_power
    probs = probs[probs > 0.0]
    if probs.size <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(entropy / math.log(probs.size))


def _dominant_frequency(power: np.ndarray, freqs: np.ndarray) -> float:
    if power.size == 0:
        return 0.0
    valid = freqs > 0.0
    if not np.any(valid):
        return 0.0
    idx = np.argmax(power[valid])
    return float(freqs[valid][idx])


def _gravity_angle_stability(gravity_x: np.ndarray, gravity_y: np.ndarray, gravity_z: np.ndarray) -> float:
    gravity_vectors = np.column_stack((gravity_x, gravity_y, gravity_z))
    mean_vec = np.mean(gravity_vectors, axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    norms = np.linalg.norm(gravity_vectors, axis=1)
    valid = (norms > 1e-8) & (mean_norm > 1e-8)
    if not np.any(valid):
        return 0.0

    dots = np.sum(gravity_vectors[valid] * mean_vec, axis=1)
    cos_angles = np.clip(dots / (norms[valid] * mean_norm), -1.0, 1.0)
    angles = np.arccos(cos_angles)
    return float(np.std(angles, ddof=0))


def _spectral_centroid(power: np.ndarray, freqs: np.ndarray) -> float:
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    return float(np.sum(freqs * power) / total)


def _spectral_spread(power: np.ndarray, freqs: np.ndarray) -> float:
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    centroid = _spectral_centroid(power, freqs)
    return float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total))


def _spectral_rolloff(power: np.ndarray, freqs: np.ndarray, roll_percent: float = 0.85) -> float:
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    cumsum = np.cumsum(power)
    idx = int(np.searchsorted(cumsum, roll_percent * total))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _spectral_flatness(power: np.ndarray) -> float:
    p = power[power > 0.0]
    if p.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(p))) / (np.mean(p) + 1e-8))


def _autocorr_features(signal: np.ndarray, fs: float, min_hz: float = 0.5, max_hz: float = 4.0) -> tuple[float, float, float, float]:
    x = signal - np.mean(signal)
    if x.size < 3 or _safe_std(x) < 1e-8:
        return 0.0, 0.0, 0.0, 0.0

    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    ac = ac / (ac[0] + 1e-8)

    min_lag = max(1, int(fs / max_hz))
    max_lag = min(len(ac) - 1, int(fs / min_hz))
    if max_lag <= min_lag:
        return 0.0, 0.0, 0.0, 0.0

    search = ac[min_lag : max_lag + 1]
    best = int(np.argmax(search)) + min_lag
    peak = float(ac[best])
    lag_s = float(best / fs)
    freq_hz = float(1.0 / lag_s) if lag_s > 0.0 else 0.0
    periodicity_ratio = float(peak / (np.mean(np.abs(search)) + 1e-8))
    return peak, lag_s, freq_hz, periodicity_ratio


def _peak_features(signal: np.ndarray, fs: float) -> tuple[float, float, float, float, float, float, float]:
    x = signal - np.mean(signal)
    if x.size < 3 or _safe_std(x) < 1e-8:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    min_distance = max(1, int(0.25 * fs))
    prominence = 0.25 * _safe_std(x)
    peaks, props = _find_peaks(x, distance=min_distance, prominence=prominence)
    count = float(len(peaks))
    duration = float(len(x) / fs) if fs > 0.0 else 0.0
    rate = float(count / duration) if duration > 0.0 else 0.0

    prominences = np.asarray(props.get("prominences", np.asarray([], dtype=np.float64)), dtype=np.float64)
    prom_mean = float(np.mean(prominences)) if prominences.size else 0.0
    prom_std = float(np.std(prominences, ddof=0)) if prominences.size else 0.0

    if len(peaks) >= 2:
        intervals = np.diff(peaks) / fs
        interval_mean = float(np.mean(intervals))
        interval_std = float(np.std(intervals, ddof=0))
        interval_cv = float(interval_std / (interval_mean + 1e-8))
    else:
        interval_mean = 0.0
        interval_std = 0.0
        interval_cv = 0.0

    return count, rate, prom_mean, prom_std, interval_mean, interval_std, interval_cv


def _ar_coeffs(signal: np.ndarray, order: int = 4) -> list[float]:
    x = signal - np.mean(signal)
    if x.size <= order or _safe_std(x) < 1e-8:
        return [0.0] * order

    y = x[order:]
    x_lagged = np.column_stack([x[order - k - 1 : -k - 1] for k in range(order)])
    try:
        coeffs, *_ = np.linalg.lstsq(x_lagged, y, rcond=None)
        return [float(c) for c in coeffs]
    except np.linalg.LinAlgError:
        return [0.0] * order


def _max_crosscorr_lag(a: np.ndarray, b: np.ndarray, fs: float, max_lag_s: float = 0.5) -> tuple[float, float]:
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    a_std = _safe_std(a_centered)
    b_std = _safe_std(b_centered)
    if a_std < 1e-8 or b_std < 1e-8:
        return 0.0, 0.0

    max_lag = int(max_lag_s * fs)
    corr = np.correlate(a_centered, b_centered, mode="full")
    lags = np.arange(-len(a_centered) + 1, len(a_centered))

    mask = np.abs(lags) <= max_lag
    corr = corr[mask]
    lags = lags[mask]
    corr = corr / ((len(a_centered) * a_std * b_std) + 1e-8)
    best_idx = int(np.argmax(np.abs(corr)))
    return float(corr[best_idx]), float(lags[best_idx] / fs)


def _phase_at_frequency(a: np.ndarray, b: np.ndarray, fs: float, freq_hz: float) -> float:
    if freq_hz <= 0.0 or a.size < 2 or b.size < 2:
        return 0.0
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    if _safe_std(a_centered) < 1e-8 or _safe_std(b_centered) < 1e-8:
        return 0.0

    a_fft = np.fft.rfft(a_centered)
    b_fft = np.fft.rfft(b_centered)
    freqs = np.fft.rfftfreq(a_centered.size, d=1.0 / fs)
    if freqs.size <= 1:
        return 0.0
    idx = int(np.argmin(np.abs(freqs - freq_hz)))
    idx = max(1, min(idx, freqs.size - 1))
    phase = np.angle(b_fft[idx]) - np.angle(a_fft[idx])
    return float(np.arctan2(np.sin(phase), np.cos(phase)))


def _stft_band_energy_stats(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    nperseg: int = 32,
    noverlap: int = 16,
) -> tuple[float, float]:
    if signal.size < 2:
        return 0.0, 0.0
    nperseg_eff = min(max(8, nperseg), signal.size)
    noverlap_eff = min(noverlap, nperseg_eff - 1)
    nfft_eff = max(128, nperseg_eff)
    stft_freqs, _, Zxx = _compute_stft(
        signal,
        fs=fs,
        nperseg=nperseg_eff,
        noverlap=noverlap_eff,
        nfft=nfft_eff,
        boundary=None,
    )
    power = np.abs(Zxx) ** 2
    mask = (stft_freqs >= low_hz) & (stft_freqs <= high_hz)
    if not np.any(mask) or power.shape[1] == 0:
        return 0.0, 0.0
    band_energy_over_time = np.sum(power[mask, :], axis=0)
    return float(np.mean(band_energy_over_time)), float(np.std(band_energy_over_time, ddof=0))


def _stft_dom_freq_std(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 32,
    noverlap: int = 16,
) -> float:
    if signal.size < 2:
        return 0.0
    nperseg_eff = min(max(8, nperseg), signal.size)
    noverlap_eff = min(noverlap, nperseg_eff - 1)
    nfft_eff = max(128, nperseg_eff)
    stft_freqs, _, Zxx = _compute_stft(
        signal,
        fs=fs,
        nperseg=nperseg_eff,
        noverlap=noverlap_eff,
        nfft=nfft_eff,
        boundary=None,
    )
    power = np.abs(Zxx) ** 2
    if power.shape[1] == 0:
        return 0.0
    valid = stft_freqs > 0.0
    if not np.any(valid):
        return 0.0
    dom_freq_per_frame = stft_freqs[valid][np.argmax(power[valid, :], axis=0)]
    return float(np.std(dom_freq_per_frame, ddof=0))


def _dwt_subband_energies(signal: np.ndarray, wavelet: str = "db4", level: int = 4) -> list[float]:
    coeffs = _pywt.wavedec(signal, wavelet, level=level)
    approx_energy = float(np.sum(coeffs[0] ** 2))
    detail_energies = [float(np.sum(c ** 2)) for c in reversed(coeffs[1:])]
    total = sum(detail_energies) + approx_energy
    if total <= 0.0:
        return [0.0] * 6
    normed = [e / total for e in detail_energies]
    normed_approx = approx_energy / total
    high = normed[0] + normed[1] + normed[2]
    low = normed[3] + normed_approx
    ratio = high / (low + 1e-8)
    return normed + [normed_approx, ratio]


def window_feature_names(placement_labels: Sequence[str], axes: Sequence[str] = AXES) -> list[str]:
    names: list[str] = []
    for axis in axes:
        names.extend(
            [
                f"body_{axis}_mean",
                f"body_{axis}_std",
                f"body_{axis}_rms",
                f"body_{axis}_mad",
                f"body_{axis}_min",
                f"body_{axis}_max",
                f"body_{axis}_zcr",
                f"body_{axis}_skew",
                f"body_{axis}_kurt",
            ]
        )

    names.extend(
        [
            "body_corr_xy",
            "body_corr_xz",
            "body_corr_yz",
            "body_sma",
            "body_x_energy",
            "body_y_energy",
            "body_z_energy",
            "body_energy_xy_ratio",
            "body_energy_xz_ratio",
            "body_energy_yz_ratio",
            "jerk_x_rms",
            "jerk_y_rms",
            "jerk_z_rms",
            "jerk_mag_mean",
            "jerk_mag_std",
            "jerk_mag_rms",
            "jerk_mag_p90",
            "jerk_mag_iqr",
            "jerk_mag_dom_freq_hz",
            "body_mag_dom_freq_hz",
            "body_mag_spectral_entropy",
            "body_mag_energy",
            "body_mag_energy_0p5_4_hz",
            "body_mag_energy_0p1_0p5_hz",
            "body_mag_p10",
            "body_mag_p50",
            "body_mag_p90",
            "body_mag_iqr",
            "body_mag_range",
            "body_mag_coeff_var",
            "body_mag_median_abs_dev",
            "body_mag_skew",
            "body_mag_kurt",
            "body_mag_autocorr_peak",
            "body_mag_autocorr_lag_s",
            "body_mag_autocorr_peak_freq_hz",
            "body_mag_autocorr_periodicity_ratio",
            "body_mag_peak_count",
            "body_mag_peak_rate_hz",
            "body_mag_peak_prominence_mean",
            "body_mag_peak_prominence_std",
            "body_mag_interpeak_interval_mean",
            "body_mag_interpeak_interval_std",
            "body_mag_interpeak_interval_cv",
            "gravity_mag_mean",
            "gravity_x_mean",
            "gravity_y_mean",
            "gravity_z_mean",
            "gravity_x_std",
            "gravity_y_std",
            "gravity_z_std",
            "gravity_tilt_mean_rad",
            "gravity_tilt_std_rad",
            "gravity_pitch_mean",
            "gravity_pitch_std",
            "gravity_roll_mean",
            "gravity_roll_std",
            "gravity_angle_stability_std",
            "body_vertical_mean",
            "body_vertical_std",
            "body_vertical_rms",
            "body_vertical_energy",
            "body_horizontal_rms",
            "body_horizontal_energy",
            "vertical_horizontal_energy_ratio",
            "body_mag_first_half_mean",
            "body_mag_second_half_mean",
            "body_mag_half_mean_delta",
            "body_mag_first_half_energy",
            "body_mag_second_half_energy",
            "body_mag_half_energy_ratio",
            "gravity_tilt_start_end_delta",
            "body_mag_linear_trend",
            "body_x_dom_freq_hz",
            "body_y_dom_freq_hz",
            "body_z_dom_freq_hz",
            "body_mag_total_power",
            "body_mag_energy_4_8_hz",
            "body_mag_energy_8_20_hz",
            "body_mag_spectral_centroid",
            "body_mag_locomotion_ratio",
            "body_mag_rel_energy_0p1_0p5",
            "body_mag_rel_energy_0p5_1p5",
            "body_mag_rel_energy_1p5_3",
            "body_mag_rel_energy_3_6",
            "body_mag_spectral_spread",
            "body_mag_spectral_rolloff_85",
            "body_mag_spectral_flatness",
            "body_x_ar_1",
            "body_x_ar_2",
            "body_x_ar_3",
            "body_x_ar_4",
            "body_y_ar_1",
            "body_y_ar_2",
            "body_y_ar_3",
            "body_y_ar_4",
            "body_z_ar_1",
            "body_z_ar_2",
            "body_z_ar_3",
            "body_z_ar_4",
            "body_mag_ar_1",
            "body_mag_ar_2",
            "body_mag_ar_3",
            "body_mag_ar_4",
            "body_xy_crosscorr_max",
            "body_xy_crosscorr_lag_s",
            "body_xz_crosscorr_max",
            "body_xz_crosscorr_lag_s",
            "body_yz_crosscorr_max",
            "body_yz_crosscorr_lag_s",
            "body_xy_phase_at_dom_freq",
            "body_xz_phase_at_dom_freq",
            "body_yz_phase_at_dom_freq",
            "stft_locomotion_band_mean",
            "stft_locomotion_band_std",
            "stft_cadence_band_mean",
            "stft_cadence_band_std",
            "stft_dom_freq_std",
            "dwt_d1_energy",
            "dwt_d2_energy",
            "dwt_d3_energy",
            "dwt_d4_energy",
            "dwt_a4_energy",
            "dwt_high_low_ratio",
        ]
    )

    for placement in placement_labels:
        names.append(f"placement_{placement}")
    return names


def compute_window_features(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    placement_label: str,
    placement_labels: Sequence[str],
    *,
    target_sample_rate_hz: float,
) -> np.ndarray:
    segment = AccelerometerSegment(
        time_seconds=np.arange(acc_x.size, dtype=np.float64) / target_sample_rate_hz,
        acc_x=acc_x.astype(np.float64, copy=False),
        acc_y=acc_y.astype(np.float64, copy=False),
        acc_z=acc_z.astype(np.float64, copy=False),
    )
    split = apply_gravity_split(segment, sample_rate_hz=target_sample_rate_hz)

    body_axes = {
        "x": split.body_x,
        "y": split.body_y,
        "z": split.body_z,
    }

    features: list[float] = []
    for axis in AXES:
        values = body_axes[axis]
        features.extend(
            [
                float(np.mean(values)),
                _safe_std(values),
                float(np.sqrt(np.mean(values ** 2))),
                float(np.mean(np.abs(values - np.mean(values)))),
                float(np.min(values)),
                float(np.max(values)),
                _zero_crossing_rate(values),
                _safe_skew(values),
                _safe_kurtosis(values),
            ]
        )

    features.extend(
        [
            _safe_corr(body_axes["x"], body_axes["y"]),
            _safe_corr(body_axes["x"], body_axes["z"]),
            _safe_corr(body_axes["y"], body_axes["z"]),
            float(np.mean(np.abs(body_axes["x"]) + np.abs(body_axes["y"]) + np.abs(body_axes["z"]))),
            float(np.mean(body_axes["x"] ** 2)),
            float(np.mean(body_axes["y"] ** 2)),
            float(np.mean(body_axes["z"] ** 2)),
        ]
    )

    body_x_energy = float(np.mean(body_axes["x"] ** 2))
    body_y_energy = float(np.mean(body_axes["y"] ** 2))
    body_z_energy = float(np.mean(body_axes["z"] ** 2))
    features.extend(
        [
            body_x_energy / (body_y_energy + 1e-8),
            body_x_energy / (body_z_energy + 1e-8),
            body_y_energy / (body_z_energy + 1e-8),
        ]
    )

    jerk_x = np.diff(split.body_x) * target_sample_rate_hz
    jerk_y = np.diff(split.body_y) * target_sample_rate_hz
    jerk_z = np.diff(split.body_z) * target_sample_rate_hz
    jerk_mag = np.sqrt(jerk_x ** 2 + jerk_y ** 2 + jerk_z ** 2)
    features.extend(
        [
            float(np.sqrt(np.mean(jerk_x ** 2))) if jerk_x.size else 0.0,
            float(np.sqrt(np.mean(jerk_y ** 2))) if jerk_y.size else 0.0,
            float(np.sqrt(np.mean(jerk_z ** 2))) if jerk_z.size else 0.0,
            float(np.mean(jerk_mag)) if jerk_mag.size else 0.0,
            _safe_std(jerk_mag) if jerk_mag.size else 0.0,
            float(np.sqrt(np.mean(jerk_mag ** 2))) if jerk_mag.size else 0.0,
            float(np.percentile(jerk_mag, 90.0)) if jerk_mag.size else 0.0,
            float(np.percentile(jerk_mag, 75.0) - np.percentile(jerk_mag, 25.0)) if jerk_mag.size else 0.0,
        ]
    )
    if jerk_mag.size >= 2:
        centered_jerk = jerk_mag - np.mean(jerk_mag)
        jerk_power = np.abs(np.fft.rfft(centered_jerk)) ** 2
        jerk_freqs = np.fft.rfftfreq(centered_jerk.size, d=1.0 / target_sample_rate_hz)
        jerk_dom_freq = _dominant_frequency(jerk_power, jerk_freqs)
    else:
        jerk_dom_freq = 0.0
    features.append(jerk_dom_freq)

    body_mag = np.sqrt(split.body_x ** 2 + split.body_y ** 2 + split.body_z ** 2)
    centered_mag = body_mag - np.mean(body_mag)
    fft_values = np.fft.rfft(centered_mag)
    freqs = np.fft.rfftfreq(centered_mag.size, d=1.0 / target_sample_rate_hz)
    power = np.abs(fft_values) ** 2

    body_mag_dom_freq = _dominant_frequency(power, freqs)
    features.extend(
        [
            body_mag_dom_freq,
            _spectral_entropy(power),
            float(np.mean(body_mag ** 2)),
            _band_energy(power, freqs, 0.5, 4.0),
            _band_energy(power, freqs, 0.1, 0.5),
            float(np.percentile(body_mag, 10.0)),
            float(np.percentile(body_mag, 50.0)),
            float(np.percentile(body_mag, 90.0)),
            float(np.percentile(body_mag, 75.0) - np.percentile(body_mag, 25.0)),
            float(np.max(body_mag) - np.min(body_mag)),
            _safe_std(body_mag) / (float(np.mean(body_mag)) + 1e-8),
            _median_abs_dev(body_mag),
            _safe_skew(body_mag),
            _safe_kurtosis(body_mag),
        ]
    )

    auto_peak, auto_lag_s, auto_freq_hz, auto_periodicity = _autocorr_features(
        body_mag,
        target_sample_rate_hz,
        min_hz=0.5,
        max_hz=4.0,
    )
    features.extend([auto_peak, auto_lag_s, auto_freq_hz, auto_periodicity])

    features.extend(list(_peak_features(centered_mag, target_sample_rate_hz)))

    gravity_mag = np.sqrt(split.gravity_x ** 2 + split.gravity_y ** 2 + split.gravity_z ** 2)
    safe_gravity_mag = np.maximum(gravity_mag, 1e-8)
    gravity_tilt = np.arccos(np.clip(np.abs(split.gravity_z) / safe_gravity_mag, 0.0, 1.0))
    gravity_pitch = np.arctan2(-split.gravity_x, np.sqrt(split.gravity_y ** 2 + split.gravity_z ** 2))
    gravity_roll = np.arctan2(split.gravity_y, split.gravity_z)
    features.extend(
        [
            float(np.mean(gravity_mag)),
            float(np.mean(split.gravity_x)),
            float(np.mean(split.gravity_y)),
            float(np.mean(split.gravity_z)),
            _safe_std(split.gravity_x),
            _safe_std(split.gravity_y),
            _safe_std(split.gravity_z),
            float(np.mean(gravity_tilt)),
            _safe_std(gravity_tilt),
            float(np.mean(gravity_pitch)),
            _safe_std(gravity_pitch),
            float(np.mean(gravity_roll)),
            _safe_std(gravity_roll),
            _gravity_angle_stability(split.gravity_x, split.gravity_y, split.gravity_z),
        ]
    )

    gravity = np.column_stack((split.gravity_x, split.gravity_y, split.gravity_z))
    body = np.column_stack((split.body_x, split.body_y, split.body_z))
    g_norm = np.linalg.norm(gravity, axis=1, keepdims=True)
    g_unit = gravity / np.maximum(g_norm, 1e-8)
    body_vertical = np.sum(body * g_unit, axis=1)
    body_horizontal_vec = body - body_vertical[:, None] * g_unit
    body_horizontal_mag = np.linalg.norm(body_horizontal_vec, axis=1)

    vertical_energy = float(np.mean(body_vertical ** 2)) if body_vertical.size else 0.0
    horizontal_energy = float(np.mean(body_horizontal_mag ** 2)) if body_horizontal_mag.size else 0.0
    features.extend(
        [
            float(np.mean(body_vertical)) if body_vertical.size else 0.0,
            _safe_std(body_vertical) if body_vertical.size else 0.0,
            float(np.sqrt(np.mean(body_vertical ** 2))) if body_vertical.size else 0.0,
            vertical_energy,
            float(np.sqrt(np.mean(body_horizontal_mag ** 2))) if body_horizontal_mag.size else 0.0,
            horizontal_energy,
            vertical_energy / (horizontal_energy + 1e-8),
        ]
    )

    mid = body_mag.size // 2
    first_half = body_mag[:mid]
    second_half = body_mag[mid:]
    first_half_mean = float(np.mean(first_half)) if first_half.size else 0.0
    second_half_mean = float(np.mean(second_half)) if second_half.size else 0.0
    first_half_energy = float(np.mean(first_half ** 2)) if first_half.size else 0.0
    second_half_energy = float(np.mean(second_half ** 2)) if second_half.size else 0.0
    body_mag_half_energy_ratio = second_half_energy / (first_half_energy + 1e-8)
    gravity_tilt_start_end_delta = (
        float(gravity_tilt[-1] - gravity_tilt[0]) if gravity_tilt.size >= 2 else 0.0
    )
    if body_mag.size >= 2:
        t = np.arange(body_mag.size, dtype=np.float64) / target_sample_rate_hz
        body_mag_linear_trend = float(np.polyfit(t, body_mag, deg=1)[0])
    else:
        body_mag_linear_trend = 0.0
    features.extend(
        [
            first_half_mean,
            second_half_mean,
            second_half_mean - first_half_mean,
            first_half_energy,
            second_half_energy,
            body_mag_half_energy_ratio,
            gravity_tilt_start_end_delta,
            body_mag_linear_trend,
        ]
    )

    for axis in AXES:
        ax_centered = body_axes[axis] - np.mean(body_axes[axis])
        ax_power = np.abs(np.fft.rfft(ax_centered)) ** 2
        features.append(_dominant_frequency(ax_power, freqs))

    total_power = float(np.sum(power))
    features.extend(
        [
            total_power,
            _band_energy(power, freqs, 4.0, 8.0),
            _band_energy(power, freqs, 8.0, 20.0),
            _spectral_centroid(power, freqs),
            _band_energy(power, freqs, 0.5, 4.0) / (total_power + 1e-8),
            _band_energy(power, freqs, 0.1, 0.5) / (total_power + 1e-8),
            _band_energy(power, freqs, 0.5, 1.5) / (total_power + 1e-8),
            _band_energy(power, freqs, 1.5, 3.0) / (total_power + 1e-8),
            _band_energy(power, freqs, 3.0, 6.0) / (total_power + 1e-8),
            _spectral_spread(power, freqs),
            _spectral_rolloff(power, freqs, roll_percent=0.85),
            _spectral_flatness(power),
        ]
    )

    # AR coefficients capture shape persistence beyond summary moments.
    features.extend(_ar_coeffs(body_axes["x"], order=4))
    features.extend(_ar_coeffs(body_axes["y"], order=4))
    features.extend(_ar_coeffs(body_axes["z"], order=4))
    features.extend(_ar_coeffs(body_mag, order=4))

    xy_corr, xy_lag = _max_crosscorr_lag(body_axes["x"], body_axes["y"], target_sample_rate_hz)
    xz_corr, xz_lag = _max_crosscorr_lag(body_axes["x"], body_axes["z"], target_sample_rate_hz)
    yz_corr, yz_lag = _max_crosscorr_lag(body_axes["y"], body_axes["z"], target_sample_rate_hz)
    features.extend([xy_corr, xy_lag, xz_corr, xz_lag, yz_corr, yz_lag])

    xy_phase = _phase_at_frequency(body_axes["x"], body_axes["y"], target_sample_rate_hz, body_mag_dom_freq)
    xz_phase = _phase_at_frequency(body_axes["x"], body_axes["z"], target_sample_rate_hz, body_mag_dom_freq)
    yz_phase = _phase_at_frequency(body_axes["y"], body_axes["z"], target_sample_rate_hz, body_mag_dom_freq)
    features.extend([xy_phase, xz_phase, yz_phase])

    loco_mean, loco_std = _stft_band_energy_stats(centered_mag, target_sample_rate_hz, 0.5, 4.0)
    cadence_mean, cadence_std = _stft_band_energy_stats(centered_mag, target_sample_rate_hz, 1.4, 2.5)
    features.extend(
        [
            loco_mean,
            loco_std,
            cadence_mean,
            cadence_std,
            _stft_dom_freq_std(centered_mag, target_sample_rate_hz),
        ]
    )

    features.extend(_dwt_subband_energies(centered_mag))

    placement_set = set(placement_labels)
    normalized_placement = placement_label if placement_label in placement_set else "unknown_free_living"
    for placement in placement_labels:
        features.append(1.0 if placement == normalized_placement else 0.0)

    return np.asarray(features, dtype=np.float32)
