"""Manchester code (EM4100 and friends).

Default convention (G.E. Thomas, as used by EM4100): '1' -> high-low
``(1, 0)``, '0' -> low-high ``(0, 1)``. ``invert=True`` gives the IEEE 802.3
convention.
"""

from __future__ import annotations

import numpy as np


def encode(bits: np.ndarray, invert: bool = False) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    first = bits ^ 1 if invert else bits
    chips = np.empty(bits.size * 2, dtype=np.uint8)
    chips[0::2] = first
    chips[1::2] = first ^ 1
    return chips


def decode(chips: np.ndarray, invert: bool = False) -> np.ndarray:
    chips = np.asarray(chips, dtype=np.uint8)
    if chips.size % 2:
        raise ValueError("Manchester chip stream must have an even length")
    pairs = chips.reshape(-1, 2)
    invalid = np.flatnonzero(pairs[:, 0] == pairs[:, 1])
    if invalid.size:
        raise ValueError(f"invalid Manchester pair at bit {int(invalid[0])}")
    bits = pairs[:, 0].copy()
    return bits ^ 1 if invert else bits
