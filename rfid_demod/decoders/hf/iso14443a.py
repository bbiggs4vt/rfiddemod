"""ISO 14443A: modified-Miller reader commands, fc/16 subcarrier replies.

R->T: 100% ASK pauses (~2-3 us) on a 9.44 us (128/fc) bit grid. Pause at
the bit start = sequence Z, pause mid-bit = X, no pause = Y. Rules:
'1' -> X; '0' -> Z (after 0 or start of comm) or Y (after 1); SOC = Z;
EOC = '0' followed by one unmodulated bit. Decoding classifies pause
positions on the grid — carrier-derived timing, no clock recovery.

T->R: OOK subcarrier at fc/16 = 847.5 kHz, Manchester at 106 kbps:
logic 1 = subcarrier in the first half-bit, SOF = one '1'-shaped bit,
end = one bit without subcarrier.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from rfid_demod.common import edges as edges_mod
from rfid_demod.common import envelope as env_mod
from rfid_demod.common.subcarrier import mix_magnitude
from rfid_demod.encodings import manchester
from rfid_demod.io import Frame
from rfid_demod.parsers import iso14443a_frames as a_frames

from ._util import align_half_split

FC = 13.56e6
SUBCARRIER = FC / 16          # 847.5 kHz
PAUSE_RANGE_US = (1.0, 4.5)

_EXPECT = {
    "REQA": "atqa",
    "WUPA": "atqa",
    "Anticollision": "uid",
    "Select": "sak",
}


def _bit_samples(sample_rate: float) -> float:
    return 128.0 * sample_rate / FC


# ---------------------------------------------------------------- synth

def reader_frame_levels(bits: np.ndarray, sample_rate: float,
                        pause_us: float = 2.5) -> np.ndarray:
    """One R->T frame as 0/1 levels: SOC + bits + EOC + one idle bit."""
    bit = int(round(_bit_samples(sample_rate)))
    pw = int(round(pause_us * 1e-6 * sample_rate))
    bits = np.asarray(bits, dtype=np.uint8)
    total = (bits.size + 3) * bit
    levels = np.ones(total, dtype=np.uint8)

    def pause(pos: int) -> None:
        levels[pos:pos + pw] = 0

    pause(0)                                    # SOC = Z
    prev = 0
    seq = list(bits) + [0]                      # EOC contributes a coded '0'
    for i, b in enumerate(seq, start=1):
        if b:
            pause(i * bit + bit // 2)           # X
        elif prev == 0:
            pause(i * bit)                      # Z
        # else Y: no pause
        prev = b
    return levels


def tag_wave(bits: np.ndarray, sample_rate: float) -> np.ndarray:
    """T->R reply as a 0/1 waveform: SOF ('1') + data, OOK'd subcarrier."""
    half = int(round(_bit_samples(sample_rate) / 2))
    chips = manchester.encode(np.concatenate([[1], np.asarray(bits, np.uint8)]))
    gate = np.repeat(chips, half)
    n = np.arange(gate.size)
    square = ((n * SUBCARRIER / sample_rate) % 1.0 < 0.5).astype(np.uint8)
    return (gate & square).astype(np.float64)


# --------------------------------------------------------------- decode

def _pauses(env: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Centers and widths of deep short pauses (100% ASK dips)."""
    smoothed = env_mod.moving_average(env, max(1, int(0.5e-6 * sample_rate)))
    threshold = env_mod.midpoint_threshold(smoothed, lo_pct=0.5, hi_pct=99.5)
    binary = env_mod.to_binary(smoothed, threshold=threshold, hysteresis=0.15)
    starts, lengths = edges_mod.level_runs(binary, level=0)
    lo = PAUSE_RANGE_US[0] * 1e-6 * sample_rate
    hi = PAUSE_RANGE_US[1] * 1e-6 * sample_rate
    keep = (lengths >= lo) & (lengths <= hi)
    return (starts[keep] + lengths[keep] / 2.0), lengths[keep]


def reader_frames(env: np.ndarray, sample_rate: float) -> List[dict]:
    bit = _bit_samples(sample_rate)
    centers, _ = _pauses(env, sample_rate)
    if centers.size == 0:
        return []

    groups = np.split(centers, np.flatnonzero(np.diff(centers) > 2.5 * bit) + 1)
    frames: List[dict] = []
    for grp in groups:
        if grp.size < 2:
            continue
        offsets = grp - grp[0]
        seq = {}
        for off in offsets:
            pos = off / bit
            k = int(round(pos))
            if abs(pos - k) <= 0.25:
                seq.setdefault(k, "Z")
                continue
            k = int(round(pos - 0.5))
            if abs(pos - 0.5 - k) <= 0.25:
                seq.setdefault(k, "X")
        if seq.get(0) != "Z":
            continue  # SOC must be a Z
        last = max(seq)

        bits: List[int] = []
        prev = 0
        valid = True
        for k in range(1, last + 1):
            kind = seq.get(k)
            if kind == "X":
                bits.append(1)
                prev = 1
            elif kind == "Z":
                bits.append(0)
                prev = 0
            elif prev == 1:      # Y after '1' -> 0
                bits.append(0)
                prev = 0
            else:                # Y after '0' mid-frame: malformed
                valid = False
                break
        if not valid or not bits:
            continue
        if seq[last] == "Z":
            bits.pop()           # trailing Z was the EOC's coded '0'
        if not bits:
            continue
        frames.append({
            "bits": np.array(bits, dtype=np.uint8),
            "start": int(grp[0] - bit / 4),
            "end": int(grp[0] + (last + 2) * bit),
        })
    return frames


def decode_tag(
    window: np.ndarray,
    sample_rate: float,
    expected: Optional[str],
) -> Optional[Tuple[dict, int, np.ndarray]]:
    """Decode one reply window; returns (fields, start_offset, bits)."""
    half = _bit_samples(sample_rate) / 2.0
    sc = mix_magnitude(window, sample_rate, SUBCARRIER, avg_cycles=2.0)
    ref = float(np.percentile(sc, 99))
    # p10, not the median: a reply can occupy most of the window, which
    # would drag a median "floor" up into the signal level.
    floor = float(np.percentile(sc, 10))
    threshold = 0.5 * (ref + floor)

    # Coarse SOF position from a heavily smoothed envelope (single noise
    # dips chop threshold-crossing runs), then align the bit grid by
    # maximizing the Manchester half-contrast.
    sc_slow = env_mod.moving_average(sc, max(1, int(half / 2)))
    above = np.flatnonzero(sc_slow > threshold)
    if above.size < half:
        return None
    start = align_half_split(sc, int(above[0]), half)

    # Decide each half-bit from mean subcarrier amplitude. When the
    # preceding command fixes the reply length, count bits instead of
    # detecting the end-of-communication level (carrier-derived timing:
    # at low SNR the weakest ON halves overlap the noise-pair level, so
    # length counting + parity/CRC is far more robust than level sensing).
    expected_bits = {"atqa": 18, "uid": 45, "sak": 27}.get(expected)
    bits: List[int] = []
    sig = float(sc[start:start + int(half)].mean())   # SOF's ON half
    max_bits = int((sc.size - start) / (2 * half))
    for k in range(max_bits):
        if expected_bits is not None and len(bits) > expected_bits:
            break
        a0 = start + int(round(2 * k * half))
        a1 = start + int(round((2 * k + 1) * half))
        a2 = start + int(round((2 * k + 2) * half))
        first = float(sc[a0:a1].mean()) if a1 > a0 else 0.0
        second = float(sc[a1:a2].mean()) if a2 > a1 else 0.0
        current = max(first, second)
        end_factor = 0.15 if expected_bits is not None else 0.35
        if current < floor + end_factor * (sig - floor):
            break  # one unmodulated bit = end of communication
        bits.append(1 if first > second else 0)
        sig = 0.8 * sig + 0.2 * current

    if len(bits) < 2 or bits[0] != 1:   # SOF is a '1'-shaped bit
        return None
    data_bits = np.array(bits[1:], dtype=np.uint8)
    frame = a_frames.parse_frame_bits(data_bits)
    fields = a_frames.parse_tag(frame, expected)
    return fields, start, data_bits


def decode(x: np.ndarray, sample_rate: float) -> List[Frame]:
    env = np.abs(np.asarray(x))
    rframes = reader_frames(env, sample_rate)
    out: List[Frame] = []
    margin = max(4, int(2e-6 * sample_rate))

    for j, rf in enumerate(rframes):
        frame = a_frames.parse_frame_bits(rf["bits"])
        info = a_frames.parse_reader(frame)
        out.append(Frame(
            timestamp=rf["start"] / sample_rate,
            band="hf",
            direction="R->T",
            bits="".join(map(str, rf["bits"])),
            fields={**info, "protocol": "iso14443a"},
            crc_ok=info.get("crc_ok"),
        ))

        gap_start = rf["end"] + margin
        gap_end = rframes[j + 1]["start"] - margin if j + 1 < len(rframes) else env.size
        if gap_end - gap_start < 4 * _bit_samples(sample_rate):
            continue
        reply = decode_tag(x[gap_start:gap_end], sample_rate,
                           _EXPECT.get(info.get("command")))
        if reply is None:
            continue
        fields, offset, bits = reply
        if fields.get("type") == "raw" and "bytes" not in fields:
            continue  # not even frame-shaped; almost certainly noise
        out.append(Frame(
            timestamp=(gap_start + offset) / sample_rate,
            band="hf",
            direction="T->R",
            bits="".join(map(str, bits)),
            fields={**fields, "protocol": "iso14443a"},
            crc_ok=fields.get("crc_ok"),
        ))
    return out
