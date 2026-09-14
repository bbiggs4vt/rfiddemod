"""Gain control / normalization."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

_EPS = 1e-12


def normalize(x: np.ndarray) -> np.ndarray:
    """Scale to unit peak magnitude (good enough for file-based processing)."""
    x = np.asarray(x)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < _EPS:
        return x
    out = x / peak
    return out.astype(np.complex64) if np.iscomplexobj(out) else out


def agc(x: np.ndarray, sample_rate: float, tau: float = 1e-3) -> np.ndarray:
    """Sliding AGC: divide by a one-pole tracked RMS (time constant ``tau``)."""
    x = np.asarray(x)
    a = float(np.exp(-1.0 / (tau * sample_rate)))
    power = lfilter([1.0 - a], [1.0, -a], np.abs(x) ** 2)
    gain = 1.0 / np.sqrt(np.maximum(power, _EPS))
    out = x * gain
    return out.astype(np.complex64) if np.iscomplexobj(out) else out
