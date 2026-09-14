"""Envelope extraction, smoothing, and thresholding to a binary waveform."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import lfilter


def envelope(x: np.ndarray) -> np.ndarray:
    """Magnitude envelope of a (complex or real) signal."""
    return np.abs(x)


def sliding_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centered sliding mean over ``window`` samples, O(N) via cumsums.

    Zero-padded at the edges (identical to ``np.convolve(x, ones(w)/w,
    'same')``, but linear-time). Accepts real or complex input.
    """
    x = np.asarray(x)
    out_dtype = np.complex128 if np.iscomplexobj(x) else np.float64
    if window <= 1:
        return x.astype(out_dtype)
    n = x.size
    csum = np.concatenate(([0], np.cumsum(x, dtype=out_dtype)))
    if window >= n:
        idx = np.arange(n)
        lo = np.clip(idx - window // 2, 0, n)
        hi = np.clip(idx - window // 2 + window, 0, n)
        return (csum[hi] - csum[lo]) / window
    # out[i] averages x[i - w//2 : i - w//2 + w]; the bulk is a plain
    # slice difference, only the edge regions need index arithmetic.
    out = np.empty(n, dtype=out_dtype)
    a = window // 2
    out[a:a + n - window + 1] = (csum[window:] - csum[:-window]) / window
    left = np.arange(a)
    out[:a] = csum[left + window - a] / window
    right = np.arange(a + n - window + 1, n)
    out[a + n - window + 1:] = (csum[n] - csum[right - a]) / window
    return out


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average (zero group delay), window in samples."""
    return sliding_mean(np.asarray(x, dtype=np.float64), window)


def lowpass_1pole(x: np.ndarray, sample_rate: float, cutoff_hz: float) -> np.ndarray:
    """Single-pole IIR low-pass (causal; introduces group delay).

    For envelope smoothing pick ``cutoff_hz`` around 10x the bit rate.
    """
    a = float(np.exp(-2.0 * np.pi * cutoff_hz / sample_rate))
    return lfilter([1.0 - a], [1.0, -a], x)


def percentile_est(x: np.ndarray, q, max_samples: int = 100_000):
    """Percentile estimate from a strided subsample of large arrays.

    Threshold estimation doesn't need exact order statistics over tens of
    megasamples; a regular subsample is statistically equivalent here and
    keeps the partition cost bounded.
    """
    x = np.asarray(x)
    step = max(1, x.size // max_samples)
    return np.percentile(x[::step], q)


def midpoint_threshold(env: np.ndarray, lo_pct: float = 10.0, hi_pct: float = 90.0) -> float:
    """Threshold halfway between the low and high envelope levels.

    Percentiles rather than min/max so noise spikes don't skew it.
    """
    lo, hi = percentile_est(env, [lo_pct, hi_pct])
    return 0.5 * (float(lo) + float(hi))


def _hysteresis_slice(env: np.ndarray, lo: float, hi: float) -> np.ndarray:
    state = np.full(env.shape, -1, dtype=np.int8)
    state[env >= hi] = 1
    state[env <= lo] = 0
    decided = state >= 0
    idx = np.where(decided, np.arange(env.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = state[idx]
    out[out < 0] = 0  # leading undecided region defaults to low
    return out.astype(np.uint8)


def to_binary(
    env: np.ndarray,
    threshold: Optional[float] = None,
    hysteresis: float = 0.0,
) -> np.ndarray:
    """Slice an envelope into a 0/1 waveform.

    ``hysteresis`` is a fraction of the threshold: the comparator switches
    high above ``threshold * (1 + h)`` and low below ``threshold * (1 - h)``,
    holding its previous state in between.
    """
    env = np.asarray(env, dtype=np.float64)
    if threshold is None:
        threshold = midpoint_threshold(env)
    if hysteresis <= 0.0:
        return (env > threshold).astype(np.uint8)
    return _hysteresis_slice(env, threshold * (1.0 - hysteresis), threshold * (1.0 + hysteresis))
