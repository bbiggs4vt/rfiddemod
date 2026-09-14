"""Shared HF tag-decode helpers."""

from __future__ import annotations

import numpy as np


def window_sums(sc: np.ndarray) -> np.ndarray:
    """Cumulative sum with a leading 0: ``sums[b] - sums[a]`` is sum(sc[a:b]).

    Lets bit loops read window means in O(1) instead of ndarray.mean()
    per half-bit (the dominant cost at HF sample rates).
    """
    return np.concatenate(([0.0], np.cumsum(sc, dtype=np.float64)))


def align_half_split(
    sc: np.ndarray,
    coarse: int,
    half: float,
    nbits: int = 6,
) -> int:
    """Refine a bit-grid origin on a half-split (Manchester-like) envelope.

    Every bit puts energy in exactly one of its two halves, so the summed
    |first-half sum - second-half sum| over a few bits peaks when the grid
    is aligned to bit boundaries (a half-bit offset scores ~50%). Searches
    within +-1 half-bit around ``coarse``; fully vectorized.
    """
    n = sc.size
    step = max(1, int(half / 16))
    starts = coarse + np.arange(-int(half), int(half) + 1, step)
    starts = starts[starts >= 0]
    if starts.size == 0:
        return coarse

    nb = nbits
    while nb >= 2:
        span = int(round(2 * nb * half))
        kept = starts[starts + span <= n]
        if kept.size:
            break
        nb -= 1
    else:
        return coarse
    starts = kept

    k = np.arange(nb)
    off0 = np.round(2 * k * half).astype(np.int64)
    off1 = np.round((2 * k + 1) * half).astype(np.int64)
    off2 = np.round((2 * k + 2) * half).astype(np.int64)
    csum = window_sums(sc)
    a0 = starts[:, None] + off0
    a1 = starts[:, None] + off1
    a2 = starts[:, None] + off2
    first = csum[a1] - csum[a0]
    second = csum[a2] - csum[a1]
    score = np.abs(first - second).sum(axis=1)
    return int(starts[int(np.argmax(score))])
