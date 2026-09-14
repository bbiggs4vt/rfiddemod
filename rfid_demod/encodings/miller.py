"""Miller (delay) code and Miller-modulated subcarrier (EPC Gen2 M=2/4/8).

Baseband Miller: '1' -> mid-bit transition; a bit-boundary transition occurs
only between two consecutive '0's. Gen2 tags transmit "Miller-modulated
subcarrier": the baseband multiplied by a square wave of M subcarrier
cycles per bit.

Note: ISO 14443A reader->tag uses *modified* Miller (pause-position coding),
which lives with the HF decoder in build-order step 4.
"""

from __future__ import annotations

import numpy as np


def encode(bits: np.ndarray, initial_level: int = 1) -> np.ndarray:
    """Baseband Miller as 2 chips (half-bit levels) per bit."""
    bits = np.asarray(bits, dtype=np.uint8)
    chips = np.empty(bits.size * 2, dtype=np.uint8)
    level = initial_level & 1
    prev = None
    for i, b in enumerate(bits):
        if b == 0 and prev == 0:
            level ^= 1  # boundary transition between consecutive zeros
        chips[2 * i] = level
        if b == 1:
            level ^= 1  # mid-bit transition
        chips[2 * i + 1] = level
        prev = b
    return chips


def decode(chips: np.ndarray) -> np.ndarray:
    chips = np.asarray(chips, dtype=np.uint8)
    if chips.size % 2:
        raise ValueError("Miller chip stream must have an even length")
    pairs = chips.reshape(-1, 2)
    bits = (pairs[:, 0] != pairs[:, 1]).astype(np.uint8)

    boundary = pairs[1:, 0] != pairs[:-1, 1]
    expected = (bits[:-1] == 0) & (bits[1:] == 0)
    bad = np.flatnonzero(boundary != expected)
    if bad.size:
        raise ValueError(f"Miller boundary rule violated before bit {int(bad[0]) + 1}")
    return bits


def modulate_subcarrier(bits: np.ndarray, m: int, initial_level: int = 1) -> np.ndarray:
    """Miller-modulated subcarrier at half-cycle resolution.

    Returns 0/1 levels with ``2 * m`` half-cycles per bit (M subcarrier
    cycles per bit); upsample for a sampled waveform.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    chips = encode(bits, initial_level=initial_level)
    base = np.repeat(chips, m)                    # m half-cycles per half-bit
    sub = np.tile([1, 0], base.size // 2).astype(np.uint8)
    return base ^ sub
