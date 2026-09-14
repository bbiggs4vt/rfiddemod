"""Frequency shift so the reader carrier sits at DC."""

from __future__ import annotations

import numpy as np


def retune(
    x: np.ndarray,
    freq_offset: float,
    sample_rate: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Shift the component at ``freq_offset`` Hz down to DC.

    Implements ``x * exp(-j 2 pi f_off t)`` from the brief.
    """
    n = np.arange(len(x))
    rot = np.exp(-1j * (2.0 * np.pi * freq_offset * n / sample_rate + phase))
    return (np.asarray(x) * rot).astype(np.complex64)
