"""Envelope extraction, smoothing, and thresholding to a binary waveform."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import lfilter


def envelope(x: np.ndarray) -> np.ndarray:
    """Magnitude envelope of a (complex or real) signal."""
    return np.abs(x)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average (zero group delay), window in samples."""
    if window <= 1:
        return np.asarray(x, dtype=np.float64)
    kernel = np.full(window, 1.0 / window)
    return np.convolve(np.asarray(x, dtype=np.float64), kernel, mode="same")


def lowpass_1pole(x: np.ndarray, sample_rate: float, cutoff_hz: float) -> np.ndarray:
    """Single-pole IIR low-pass (causal; introduces group delay).

    For envelope smoothing pick ``cutoff_hz`` around 10x the bit rate.
    """
    a = float(np.exp(-2.0 * np.pi * cutoff_hz / sample_rate))
    return lfilter([1.0 - a], [1.0, -a], x)


def midpoint_threshold(env: np.ndarray, lo_pct: float = 10.0, hi_pct: float = 90.0) -> float:
    """Threshold halfway between the low and high envelope levels.

    Percentiles rather than min/max so noise spikes don't skew it.
    """
    lo = float(np.percentile(env, lo_pct))
    hi = float(np.percentile(env, hi_pct))
    return 0.5 * (lo + hi)


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
