"""ISO 15693: 1-of-4 PPM reader commands, single-subcarrier replies.

R->T (1-of-4): each 75.52 us symbol carries one 9.44 us pause; the pause
sits in the second half of quarter v (offset v*18.88 + 9.44), encoding
two bits, least-significant pair first. SOF = 9.44 unmodulated + 9.44
pause + 18.88 unmodulated; EOF = 9.44 unmodulated + 9.44 pause + 9.44
unmodulated. 1-of-256 is not implemented yet.

T->R (single subcarrier fc/32 = 423.75 kHz, high data rate 26.48 kbps):
bit = 37.76 us; '0' = 8 subcarrier pulses then off, '1' = off then 8
pulses. SOF = 56.64 us pulse burst + a logic 1; EOF = a logic 0 + burst.
CRC is the ISO 15693 CRC (X-25, same as CRC-B).

Timing layouts are encoded once here and mirrored by the synthesizers;
validation against a real capture is pending (as noted for HID/Miller).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from rfid_demod.common import edges as edges_mod
from rfid_demod.common import envelope as env_mod
from rfid_demod.common.subcarrier import mix_magnitude
from rfid_demod.io import Frame
from rfid_demod.parsers import iso15693_frames as v_frames

from ._util import align_half_split, window_sums

FC = 13.56e6
SUBCARRIER = FC / 32          # 423.75 kHz
T9_US = 9.44                  # base time unit, 128/fc
PAUSE_RANGE_US = (6.5, 12.5)


def _t9(sample_rate: float) -> float:
    return T9_US * 1e-6 * sample_rate


# ---------------------------------------------------------------- synth

def reader_frame_levels(data: bytes, sample_rate: float) -> np.ndarray:
    """One R->T frame (SOF + 1-of-4 symbols + EOF) as 0/1 levels."""
    t9 = int(round(_t9(sample_rate)))
    parts = [np.ones(t9), np.zeros(t9), np.ones(2 * t9)]        # SOF
    for byte in data:
        for pair in range(4):                                    # LSB pair first
            v = (byte >> (2 * pair)) & 0x3
            sym = np.ones(8 * t9)
            sym[(2 * v + 1) * t9:(2 * v + 2) * t9] = 0
            parts.append(sym)
    parts += [np.ones(t9), np.zeros(t9), np.ones(t9)]            # EOF
    return np.concatenate(parts).astype(np.uint8)


def tag_wave(data: bytes, sample_rate: float) -> np.ndarray:
    """T->R reply (high data rate, single subcarrier) as a 0/1 waveform."""
    half = int(round(2 * _t9(sample_rate)))   # 18.88 us = 8 subcarrier pulses
    gate: List[int] = [1, 1, 1]               # SOF burst: 56.64 us of pulses
    gate += [0, 1]                            # SOF logic 1
    for byte in data:
        for i in range(8):                    # bits LSB first
            gate += [0, 1] if (byte >> i) & 1 else [1, 0]
    gate += [1, 0]                            # EOF logic 0
    gate += [1, 1, 1]                         # EOF burst
    levels = np.repeat(np.array(gate, np.uint8), half)
    n = np.arange(levels.size)
    square = ((n * SUBCARRIER / sample_rate) % 1.0 < 0.5).astype(np.uint8)
    return (levels & square).astype(np.float64)


# --------------------------------------------------------------- decode

def _pauses(env: np.ndarray, sample_rate: float) -> np.ndarray:
    smoothed = env_mod.moving_average(env, max(1, int(1e-6 * sample_rate)))
    threshold = env_mod.midpoint_threshold(smoothed, lo_pct=0.5, hi_pct=99.5)
    binary = env_mod.to_binary(smoothed, threshold=threshold, hysteresis=0.15)
    starts, lengths = edges_mod.level_runs(binary, level=0)
    lo = PAUSE_RANGE_US[0] * 1e-6 * sample_rate
    hi = PAUSE_RANGE_US[1] * 1e-6 * sample_rate
    keep = (lengths >= lo) & (lengths <= hi)
    return starts[keep].astype(np.float64)


def reader_frames(env: np.ndarray, sample_rate: float) -> List[dict]:
    t9 = _t9(sample_rate)
    sym = 8 * t9
    pauses = _pauses(env, sample_rate)
    if pauses.size == 0:
        return []

    groups = np.split(pauses, np.flatnonzero(np.diff(pauses) > 2 * sym) + 1)
    frames: List[dict] = []
    for grp in groups:
        if grp.size < 3:          # SOF + at least one symbol + EOF
            continue
        origin = grp[0] - t9      # SOF pause sits at t9 into the frame
        data0 = origin + 4 * t9   # first symbol after the 37.76 us SOF

        values: List[int] = []
        valid = True
        for i, p in enumerate(grp[1:-1]):
            sym_start = data0 + i * sym
            v = (p - sym_start - t9) / (2 * t9)
            vi = int(round(v))
            if not 0 <= vi <= 3 or abs(v - vi) > 0.25:
                valid = False
                break
            values.append(vi)
        if not valid or len(values) % 4:
            continue
        # EOF: pause at t9 into the window right after the last symbol
        eof_expected = data0 + len(values) * sym + t9
        if abs(grp[-1] - eof_expected) > 0.5 * t9:
            continue

        data = bytearray()
        for k in range(0, len(values), 4):
            data.append(values[k] | values[k + 1] << 2
                        | values[k + 2] << 4 | values[k + 3] << 6)
        frames.append({
            "bytes": bytes(data),
            "start": int(origin),
            "end": int(eof_expected + 2 * t9),
        })
    return frames


def decode_tag(
    window: np.ndarray,
    sample_rate: float,
    expected: Optional[str],
) -> Optional[Tuple[dict, int]]:
    t9 = _t9(sample_rate)
    half = 2 * t9                     # 18.88 us half-bit
    sc = mix_magnitude(window, sample_rate, SUBCARRIER, avg_cycles=2.0)
    # p10 floor: replies can dominate the window
    ref, floor = (float(v) for v in env_mod.percentile_est(sc, [99, 10]))
    threshold = 0.5 * (ref + floor)

    # Coarse start = onset of the 56.64 us SOF burst on a heavily smoothed
    # envelope; the bit grid nominally starts 3 half-units later. Refine by
    # half-contrast alignment (the burst end itself is noise-fragile).
    sc_slow = env_mod.moving_average(sc, max(1, int(half / 2)))
    above = np.flatnonzero(sc_slow > threshold)
    if above.size < 2 * half:
        return None
    coarse = int(above[0] + 3.0 * half)
    grid = float(align_half_split(sc, coarse, half, nbits=8))

    # ON/OFF against a tracked signal level seeded from the SOF burst.
    csum = window_sums(sc)
    burst_lo = max(0, int(grid - 3.0 * half))
    burst_hi = max(burst_lo + 1, int(grid))
    sig = float(csum[burst_hi] - csum[burst_lo]) / (burst_hi - burst_lo)
    bits: List[int] = []
    k = 0
    while True:
        a0 = int(round(grid + 2 * k * half))
        a1 = int(round(grid + (2 * k + 1) * half))
        a2 = int(round(grid + (2 * k + 2) * half))
        if a2 > sc.size:
            break
        first = (csum[a1] - csum[a0]) / (a1 - a0)
        second = (csum[a2] - csum[a1]) / (a2 - a1)
        level = floor + 0.4 * (sig - floor)
        f_on, s_on = first > level, second > level
        if f_on and s_on:              # EOF burst reached
            if bits:
                bits.pop()             # preceding bit was the EOF logic 0
            break
        if not f_on and not s_on:
            break                      # signal lost
        bits.append(0 if f_on else 1)
        sig = 0.8 * sig + 0.2 * max(first, second)
        k += 1

    if len(bits) < 9 or bits[0] != 1:  # SOF ends with a logic 1
        return None
    data_bits = bits[1:len(bits) - (len(bits) - 1) % 8]
    data = bytearray()
    for i in range(0, len(data_bits), 8):
        data.append(sum(bit << j for j, bit in enumerate(data_bits[i:i + 8])))
    fields = v_frames.parse_tag(bytes(data), expected)
    return fields, burst_lo


def decode(x: np.ndarray, sample_rate: float) -> List[Frame]:
    env = np.abs(np.asarray(x))
    rframes = reader_frames(env, sample_rate)
    out: List[Frame] = []
    margin = max(4, int(2e-6 * sample_rate))

    for j, rf in enumerate(rframes):
        info = v_frames.parse_reader(rf["bytes"])
        out.append(Frame(
            timestamp=rf["start"] / sample_rate,
            band="hf",
            direction="R->T",
            bits="".join(f"{byte:08b}" for byte in rf["bytes"]),
            fields={**info, "protocol": "iso15693"},
            crc_ok=info.get("crc_ok"),
        ))

        gap_start = rf["end"] + margin
        gap_end = rframes[j + 1]["start"] - margin if j + 1 < len(rframes) else env.size
        if gap_end - gap_start < 16 * _t9(sample_rate):
            continue
        expected = "inventory" if info.get("command") == "Inventory" else None
        reply = decode_tag(x[gap_start:gap_end], sample_rate, expected)
        if reply is None:
            continue
        fields, offset = reply
        out.append(Frame(
            timestamp=(gap_start + offset) / sample_rate,
            band="hf",
            direction="T->R",
            bits="".join(f"{int(b):08b}" for b in bytes.fromhex(fields["bytes"])),
            fields={**fields, "protocol": "iso15693"},
            crc_ok=fields.get("crc_ok"),
        ))
    return out
