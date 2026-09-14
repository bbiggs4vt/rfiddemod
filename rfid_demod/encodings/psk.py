"""PSK phase-symbol codecs (LF: T5577 PSK modes).

Works on per-bit phase symbols (0 = 0 deg, 1 = 180 deg); the carrier-domain
modulation/demodulation belongs to synth and the LF decoder.

PSK1: phase changes where the data changes (so decode needs the first bit
as a reference). PSK2: phase changes on every '1' (differential).
"""

from __future__ import annotations

import numpy as np


def encode(bits: np.ndarray, mode: int = 1, initial_phase: int = 0) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size == 0:
        return np.empty(0, dtype=np.uint8)
    if mode == 1:
        toggles = np.diff(bits, prepend=bits[:1]) != 0
    elif mode == 2:
        toggles = bits.astype(bool)
    else:
        raise ValueError("mode must be 1 (PSK1) or 2 (PSK2)")
    return ((initial_phase + np.cumsum(toggles)) % 2).astype(np.uint8)


def decode(
    phases: np.ndarray,
    mode: int = 1,
    initial_phase: int = 0,
    first_bit: int = 0,
) -> np.ndarray:
    phases = np.asarray(phases, dtype=np.uint8)
    if phases.size == 0:
        return np.empty(0, dtype=np.uint8)
    toggles = (np.diff(phases, prepend=np.uint8(initial_phase)) % 2).astype(np.uint8)
    if mode == 2:
        return toggles
    if mode != 1:
        raise ValueError("mode must be 1 (PSK1) or 2 (PSK2)")
    # PSK1: a toggle marks a data change; integrate from the reference bit.
    bits = np.empty_like(phases)
    bits[0] = first_bit & 1
    if phases.size > 1:
        bits[1:] = (first_bit + np.cumsum(toggles[1:])) % 2
    return bits
