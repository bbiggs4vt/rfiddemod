"""Shared HF tag-decode helpers."""

from __future__ import annotations

import numpy as np


def align_half_split(
    sc: np.ndarray,
    coarse: int,
    half: float,
    nbits: int = 6,
) -> int:
    """Refine a bit-grid origin on a half-split (Manchester-like) envelope.

    Every bit puts energy in exactly one of its two halves, so the summed
    |first-half mean - second-half mean| over a few bits peaks when the
    grid is aligned to bit boundaries (a half-bit offset scores ~50%).
    Searches within +-1 half-bit around ``coarse``.
    """
    step = max(1, int(half / 16))
    best_start, best_score = coarse, -1.0
    for delta in range(-int(half), int(half) + 1, step):
        s0 = coarse + delta
        if s0 < 0:
            continue
        nb = min(nbits, int((sc.size - s0) / (2 * half)))
        if nb < 2:
            continue
        score = 0.0
        for k in range(nb):
            a0 = s0 + int(round(2 * k * half))
            a1 = s0 + int(round((2 * k + 1) * half))
            a2 = s0 + int(round((2 * k + 2) * half))
            score += abs(float(sc[a0:a1].mean()) - float(sc[a1:a2].mean()))
        score /= nb
        if score > best_score:
            best_score, best_start = score, s0
    return best_start
