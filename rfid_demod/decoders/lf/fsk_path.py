"""LF FSK path: envelope tone tracking -> FSK bits -> HID Prox frames.

FSK2a per the brief: tones at RF/8 and RF/10 (carrier cycles per FSK
cycle), 50 carrier cycles per bit. Tone decisions come from the spacing of
rising envelope crossings; bit timing is the fixed 50-cycle grid aligned
to the observed tone transitions (carrier-derived, no clock recovery).

Convention pending validation against a real capture: bit 1 -> RF/10.
Both polarities are scanned when hunting the preamble, so a flipped
mapping still decodes.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from rfid_demod.common import envelope as env_mod
from rfid_demod.io import Frame
from rfid_demod.parsers import hid_wiegand

FSK2A_PERIODS = (8, 10)   # carrier cycles per FSK cycle: bit 0, bit 1
CYCLES_PER_BIT = 50


def _rising_edges(binary: np.ndarray) -> np.ndarray:
    d = np.diff(binary.astype(np.int8))
    return np.flatnonzero(d == 1) + 1


def fsk_bits(
    env: np.ndarray,
    sample_rate: float,
    carrier_freq: float,
    periods: Tuple[int, int] = FSK2A_PERIODS,
    cycles_per_bit: int = CYCLES_PER_BIT,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Demodulate envelope FSK to bits on the 50-cycle grid.

    Returns ``(bits, valid, grid_offset_samples, samples_per_bit)``;
    empty arrays when the capture doesn't look like FSK at these tones.
    """
    p0 = periods[0] * sample_rate / carrier_freq
    p1 = periods[1] * sample_rate / carrier_freq
    spb = cycles_per_bit * sample_rate / carrier_freq

    smoothed = env_mod.moving_average(env, max(1, int(p0 / 8)))
    baseline = env_mod.moving_average(smoothed, max(1, int(4 * p1)))
    binary = (smoothed - baseline > 0).astype(np.uint8)

    rising = _rising_edges(binary)
    if rising.size < 3 * cycles_per_bit // periods[1]:
        return np.empty(0, np.uint8), np.empty(0, bool), 0.0, spb

    intervals = np.diff(rising)
    near = (np.abs(intervals - p0) <= 0.25 * p0) | (np.abs(intervals - p1) <= 0.25 * p1)
    if near.mean() < 0.5:
        return np.empty(0, np.uint8), np.empty(0, bool), 0.0, spb

    tone = (np.abs(intervals - p1) < np.abs(intervals - p0)).astype(np.int8)

    # Per-sample tone labels (-1 = unknown, outside the crossing span).
    labels = np.full(env.size, -1, dtype=np.int8)
    labels[rising[0]:rising[-1]] = np.repeat(tone, intervals)

    # Align the 50-cycle bit grid to the tone transitions (circular mean).
    trans = rising[np.flatnonzero(np.diff(tone)) + 1]
    if trans.size:
        ang = 2.0 * np.pi * (trans % spb) / spb
        offset = (np.angle(np.mean(np.exp(1j * ang))) / (2.0 * np.pi) * spb) % spb
    else:
        offset = 0.0

    nbits = int((env.size - offset) // spb)
    bits = np.zeros(nbits, dtype=np.uint8)
    valid = np.zeros(nbits, dtype=bool)
    for k in range(nbits):
        lo = int(round(offset + k * spb))
        hi = int(round(offset + (k + 1) * spb))
        seg = labels[lo:hi]
        known = seg >= 0
        n_known = int(known.sum())
        if n_known < (hi - lo) // 2:
            continue
        ones = int((seg == 1).sum())
        frac = ones / n_known
        if 0.4 < frac < 0.6:
            continue  # too ambiguous, likely straddling junk
        bits[k] = frac >= 0.5
        valid[k] = True
    return bits, valid, float(offset), spb


def decode_fsk(
    env: np.ndarray,
    sample_rate: float,
    carrier_freq: float,
) -> List[Frame]:
    """HID Prox over FSK2a. Returns one Frame per detected repeat."""
    bits, valid, offset, spb = fsk_bits(env, sample_rate, carrier_freq)
    if not bits.size:
        return []

    frames: List[Frame] = []
    for bit_offset, fields, inverted, value in hid_wiegand.find_frames(bits, valid):
        bit_length = fields["bit_length"]
        wiegand = value & ((1 << bit_length) - 1)
        frames.append(Frame(
            timestamp=(offset + bit_offset * spb) / sample_rate,
            band="lf",
            direction="T->R",
            bits=format(wiegand, f"0{bit_length}b"),
            fields={
                **fields,
                "protocol": "hid_prox",
                "encoding": "fsk2a",
                "inverted": inverted,
            },
            crc_ok=fields.get("parity_ok"),
        ))
    return frames
