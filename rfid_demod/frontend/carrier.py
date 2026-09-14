"""Carrier cancellation.

Tag replies sit 30-60 dB below the reader carrier. Two cancellers:

* :func:`dc_block` — slow DC tracker (single-pole IIR high-pass). Cheap
  and streaming-friendly, but it lags fast leakage drift and biases
  toward the modulation while a tag is replying.
* :func:`adaptive_cancel` — estimates the leakage from *quiet* segments
  (lowest-variance blocks, i.e. carrier-only) and interpolates the
  estimate across the modulated regions before subtracting. Tracks
  amplitude/phase drift of monostatic leakage without touching the
  modulation itself.
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


def adaptive_cancel(
    x: np.ndarray,
    sample_rate: float,
    segment_us: float = 100.0,
    quiet_fraction: float = 0.25,
) -> np.ndarray:
    """Subtract a leakage estimate interpolated between quiet segments.

    The capture is split into ~``segment_us`` blocks and the quietest
    fraction taken as carrier-only (RFID links are half-duplex, so such
    gaps exist between frames); their complex means sample the leakage,
    linearly interpolated across the modulated blocks. "Quiet" is judged
    by first-difference power, not variance: modulation makes
    sample-to-sample jumps while leakage drift is smooth, so this metric
    separates them even when the drift's within-segment excursion exceeds
    the tag's modulation depth. Pick ``segment_us`` above a symbol period
    (so modulated blocks contain transitions) and well below the drift
    period (so a block mean samples the leakage faithfully).
    """
    x = np.asarray(x)
    seg = max(16, int(round(segment_us * 1e-6 * sample_rate)))
    m = x.size // seg
    if m < 4:
        out = x - x.mean()
        return out.astype(np.complex64) if np.iscomplexobj(out) else out

    view = x[:m * seg].reshape(m, seg)
    means = view.mean(axis=1)
    activity = np.mean(np.abs(np.diff(view, axis=1)) ** 2, axis=1)
    quiet = activity <= np.quantile(activity, quiet_fraction)
    # A segment that only clips the edge of a burst can sneak under the
    # activity quantile while its mean is already biased by modulation:
    # disqualify neighbors of clearly-active segments (well above the
    # noise-floor median — a quantile cutoff would land inside the noise).
    active = activity > 3.0 * float(np.median(activity))
    near_active = np.convolve(active, [1, 1, 1], mode="same") > 0
    guarded = quiet & ~near_active
    if guarded.sum() >= 2:
        quiet = guarded
    if quiet.sum() < 2:
        quiet = activity <= np.median(activity)

    centers = (np.flatnonzero(quiet) + 0.5) * seg
    quiet_means = means[quiet]
    n = np.arange(x.size)
    estimate = np.interp(n, centers, quiet_means.real)
    if np.iscomplexobj(x):
        estimate = estimate + 1j * np.interp(n, centers, quiet_means.imag)
    out = x - estimate
    return out.astype(np.complex64) if np.iscomplexobj(out) else out


class AdaptiveCanceller:
    """Callable wrapper around :func:`adaptive_cancel` with fixed params."""

    def __init__(self, sample_rate: float, segment_us: float = 100.0,
                 quiet_fraction: float = 0.25):
        self.sample_rate = sample_rate
        self.segment_us = segment_us
        self.quiet_fraction = quiet_fraction

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return adaptive_cancel(x, self.sample_rate, self.segment_us,
                               self.quiet_fraction)
