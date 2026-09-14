"""Subcarrier extraction for HF tag replies.

HF tags reply on subcarriers of the 13.56 MHz carrier: fc/16 = 847.5 kHz
(ISO 14443A/B) or fc/32 = 423.75 kHz (ISO 15693). Mixing the complex
baseband down by the subcarrier frequency and averaging over whole
subcarrier cycles isolates the tag component; the reader's CW carrier
(sitting at DC) averages out over integer cycles.
"""

from __future__ import annotations

import numpy as np


def mix_baseband(
    x: np.ndarray,
    sample_rate: float,
    freq: float,
    avg_cycles: float = 1.0,
) -> np.ndarray:
    """Complex subcarrier baseband: mix down by ``freq`` and average."""
    x = np.asarray(x)
    n = np.arange(x.size)
    bb = x * np.exp(-2j * np.pi * freq * n / sample_rate)
    window = max(1, int(round(avg_cycles * sample_rate / freq)))
    kernel = np.full(window, 1.0 / window)
    out = np.convolve(bb, kernel, mode="same")
    # The partially-filled windows at the array edges do not cancel the
    # carrier (the average no longer spans whole subcarrier cycles) and
    # would otherwise look like huge signal spikes — zero them.
    out[:window] = 0.0
    if window > 1:
        out[-window:] = 0.0
    return out


def mix_magnitude(
    x: np.ndarray,
    sample_rate: float,
    freq: float,
    avg_cycles: float = 1.0,
) -> np.ndarray:
    """Subcarrier amplitude envelope (OOK detection)."""
    return np.abs(mix_baseband(x, sample_rate, freq, avg_cycles))
