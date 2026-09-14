"""Preamble correlation for frame alignment.

RFID decoders align once on a known preamble, then count fixed symbol
periods (carrier-derived timing) — no free-running clock recovery.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

_EPS = 1e-12


def normalized_correlation(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of a real signal with a template.

    Output ``c[i]`` in [-1, 1] scores how well ``signal[i : i + len(template)]``
    matches the (zero-meaned) template, invariant to signal gain and offset.
    """
    x = np.asarray(signal, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    n = t.size
    if n == 0 or x.size < n:
        return np.empty(0)

    t = t - t.mean()
    t_norm = np.linalg.norm(t)
    if t_norm < _EPS:
        raise ValueError("template is constant; correlation is undefined")

    cross = np.correlate(x, t, mode="valid")

    # Sliding window energy of the zero-meaned signal windows via cumsums.
    csum = np.concatenate(([0.0], np.cumsum(x)))
    csum2 = np.concatenate(([0.0], np.cumsum(x * x)))
    win_sum = csum[n:] - csum[:-n]
    win_sq = csum2[n:] - csum2[:-n]
    win_var = np.maximum(win_sq - win_sum * win_sum / n, 0.0)
    denom = np.sqrt(win_var) * t_norm

    return cross / np.maximum(denom, _EPS)


def find_preamble(
    signal: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.75,
) -> Tuple[int, float]:
    """Best template match in the signal.

    Returns ``(position, score)``; position is -1 when the best score is
    below ``threshold``.
    """
    c = normalized_correlation(signal, template)
    if c.size == 0:
        return -1, 0.0
    pos = int(np.argmax(c))
    score = float(c[pos])
    if score < threshold:
        return -1, score
    return pos, score
