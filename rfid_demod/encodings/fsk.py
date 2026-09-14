"""FSK level-waveform generation (LF: HID Prox FSK2a).

FSK2a per the brief: the two tones are RF/8 and RF/10 (carrier cycles per
FSK cycle) and each bit lasts 50 carrier cycles. Which tone maps to which
bit is pinned down with the HID decoder in build-order step 2; the
generator below is generic.
"""

from __future__ import annotations

import numpy as np

# HID Prox constants, in carrier cycles.
HID_CYCLES_PER_BIT = 50
HID_FSK_PERIODS = (8, 10)  # RF/8, RF/10


def modulate(
    bits: np.ndarray,
    samples_per_bit: int,
    period0: float,
    period1: float,
) -> np.ndarray:
    """Square-wave FSK: bit b uses an FSK period of ``period{b}`` samples.

    Phase is continuous across bit boundaries. Returns 0/1 levels.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size == 0:
        return np.empty(0, dtype=np.uint8)
    inst_freq = np.where(bits == 1, 1.0 / period1, 1.0 / period0)
    inc = np.repeat(inst_freq, samples_per_bit)
    # midpoint sampling keeps float accumulation off the comparator boundary
    phase = np.cumsum(inc) - 0.5 * inc
    return ((phase % 1.0) < 0.5).astype(np.uint8)
