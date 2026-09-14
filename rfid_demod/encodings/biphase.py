"""Differential biphase (conditioned diphase), 2 chips per bit.

A level transition occurs at every bit boundary; a mid-bit transition
encodes '1' (biphase-mark, the T5577 "biphase" mode) or '0' when
``space=True`` (biphase-space — which is exactly FM0, see fm0.py).

Being differential, the code is polarity-free: decoding ignores absolute
levels and only looks at transitions.
"""

from __future__ import annotations

import numpy as np


def encode(
    bits: np.ndarray,
    initial_level: int = 1,
    space: bool = False,
) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    mid = bits ^ 1 if space else bits

    # Toggle flags before each half-bit chip: boundary toggles at even
    # positions (skipped for the very first chip), mid-bit toggles at odd.
    toggles = np.empty(bits.size * 2, dtype=np.uint8)
    toggles[0::2] = 1
    toggles[1::2] = mid
    if toggles.size:
        toggles[0] = 0
    levels = (initial_level + np.cumsum(toggles)) % 2
    return levels.astype(np.uint8)


def decode(chips: np.ndarray, space: bool = False) -> np.ndarray:
    chips = np.asarray(chips, dtype=np.uint8)
    if chips.size % 2:
        raise ValueError("biphase chip stream must have an even length")
    pairs = chips.reshape(-1, 2)

    missing = np.flatnonzero(pairs[1:, 0] == pairs[:-1, 1])
    if missing.size:
        raise ValueError(
            f"missing bit-boundary transition before bit {int(missing[0]) + 1}"
        )

    mid = (pairs[:, 0] != pairs[:, 1]).astype(np.uint8)
    return mid ^ 1 if space else mid
