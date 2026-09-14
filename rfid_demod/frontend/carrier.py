"""Carrier cancellation.

Tag replies sit 30-60 dB below the reader carrier. v1 uses a slow DC
tracker (single-pole IIR) subtracted from the signal; an adaptive canceller
hook is reserved for build-order step 5 (needed for monostatic antennas
with heavy carrier leakage).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def dc_block(x: np.ndarray, sample_rate: float, cutoff_hz: float = 1e3) -> np.ndarray:
    """Subtract a slowly-tracked DC/carrier estimate (high-pass).

    ``cutoff_hz`` must sit well below the lowest modulation rate of
    interest so the tag sidebands survive (~1 kHz default per the brief).
    """
    a = float(np.exp(-2.0 * np.pi * cutoff_hz / sample_rate))
    dc = lfilter([1.0 - a], [1.0, -a], np.asarray(x))
    out = np.asarray(x) - dc
    return out.astype(np.complex64) if np.iscomplexobj(out) else out


class AdaptiveCanceller:
    """Hook for an adaptive carrier canceller (build-order step 5).

    Planned: LMS/RLS estimate of the leakage path using carrier-only gaps
    between frames as the reference.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Adaptive carrier cancellation lands in build-order step 5; "
            "use dc_block() until then."
        )
