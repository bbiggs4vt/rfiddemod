"""ISO 14443B: 10% ASK NRZ reader commands, BPSK subcarrier replies.

Both directions use character framing at 106 kbps (etu = 128/fc):
SOF = 10-11 etu of logic 0 + 2-3 etu of logic 1, characters = start bit 0
+ 8 data bits LSB-first + stop bit 1 (EGT gaps allowed between), EOF =
10-11 etu of logic 0. R->T carries it as 10% ASK NRZ-L on the carrier;
T->R as BPSK on the 847.5 kHz subcarrier (TR1 lead-in of constant phase
defines logic 1). CRC-B on every frame.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from rfid_demod.common import edges as edges_mod
from rfid_demod.common import envelope as env_mod
from rfid_demod.common.subcarrier import mix_baseband
from rfid_demod.io import Frame
from rfid_demod.parsers import iso14443b_frames as b_frames

FC = 13.56e6
SUBCARRIER = FC / 16
# SOF/EOF low runs are 10-11 etu. A character can contain up to 9
# consecutive zeros (start bit + a 0x00 byte), so the cutoff must sit
# between 9 and 10 etu.
_MARKER_ETU = 9.5


def _etu(sample_rate: float) -> float:
    return 128.0 * sample_rate / FC


# ---------------------------------------------------------------- synth

def char_stream_levels(data: bytes, tr_lead_etus: int = 0) -> np.ndarray:
    """Frame as logic levels at 1 sample/etu: [lead 1s] SOF chars EOF."""
    bits: List[int] = [1] * tr_lead_etus
    bits += [0] * 10 + [1] * 2                       # SOF
    for byte in data:
        bits += [0] + [(byte >> i) & 1 for i in range(8)] + [1]
    bits += [0] * 10 + [1] * 2                       # EOF (+ back to idle)
    return np.array(bits, dtype=np.uint8)


def tag_wave(data: bytes, sample_rate: float,
             tr_lead_etus: int = 8) -> np.ndarray:
    """T->R reply: BPSK (+-1) on the subcarrier, gated; 0 elsewhere."""
    etu = int(round(_etu(sample_rate)))
    nrz = np.repeat(char_stream_levels(data, tr_lead_etus), etu).astype(np.float64)
    n = np.arange(nrz.size)
    carrier = np.exp(2j * np.pi * SUBCARRIER * n / sample_rate)
    return (2.0 * nrz - 1.0) * carrier


# --------------------------------------------------------------- decode

def _decode_chars(binary: np.ndarray, etu: float,
                  lo: int, hi: int) -> Tuple[bytes, bool]:
    """Character decode between SOF end and EOF start (falling-edge sync)."""
    seg = binary[lo:hi].astype(np.int8)
    falls = np.flatnonzero(np.diff(seg) == -1) + 1 + lo
    out = bytearray()
    ok = True
    pos = lo
    fi = 0
    while True:
        while fi < falls.size and falls[fi] < pos:
            fi += 1
        if fi >= falls.size:
            break
        f = int(falls[fi])
        if f + 10 * etu > hi + 0.5 * etu:
            break
        samples = [int(binary[min(int(round(f + (i + 0.5) * etu)), binary.size - 1)])
                   for i in range(10)]
        if samples[0] != 0 or samples[9] != 1:
            ok = False
            break
        out.append(sum(samples[1 + j] << j for j in range(8)))
        pos = int(f + 9.5 * etu)
    return bytes(out), ok


def _frames_from_binary(binary: np.ndarray, sample_rate: float) -> List[dict]:
    """Pair SOF/EOF markers and decode the characters between them."""
    etu = _etu(sample_rate)
    starts, lengths = edges_mod.level_runs(binary, level=0)
    keep = lengths >= _MARKER_ETU * etu
    markers = list(zip(starts[keep], lengths[keep]))

    frames: List[dict] = []
    for i in range(0, len(markers) - 1, 2):
        sof_start, sof_len = markers[i]
        eof_start, eof_len = markers[i + 1]
        data, ok = _decode_chars(binary, etu,
                                 int(sof_start + sof_len), int(eof_start))
        if not data or not ok:
            continue
        frames.append({
            "bytes": data,
            "start": int(sof_start),
            "end": int(eof_start + eof_len),
        })
    return frames


def reader_frames(env: np.ndarray, sample_rate: float) -> List[dict]:
    smoothed = env_mod.moving_average(env, max(1, int(_etu(sample_rate) / 8)))
    threshold = env_mod.midpoint_threshold(smoothed, lo_pct=0.5, hi_pct=99.5)
    # 10% ASK: the two levels sit within ~10% of each other, so the
    # hysteresis band must be much narrower than that.
    binary = env_mod.to_binary(smoothed, threshold=threshold, hysteresis=0.02)
    return _frames_from_binary(binary, sample_rate)


def decode_tag(window: np.ndarray, sample_rate: float) -> Optional[Tuple[dict, int]]:
    etu = _etu(sample_rate)
    bb = mix_baseband(window, sample_rate, SUBCARRIER, avg_cycles=1.0)
    mag = np.abs(bb)   # mix_baseband already blanks the invalid edges
    ref, low = (float(v) for v in env_mod.percentile_est(mag, [99, 5]))
    if ref < 4.0 * low:
        return None                       # no subcarrier burst in this gap
    active = np.flatnonzero(mag > 0.5 * ref)
    if active.size < 12 * etu:
        return None
    r0, r1 = int(active[0]), int(active[-1])

    region = bb[r0:r1]
    theta = 0.5 * np.angle(np.mean(region * region))
    y = np.real(region * np.exp(-1j * theta))
    binary = (y > 0).astype(np.uint8)
    # TR1 lead-in is logic 1; flip the BPSK polarity ambiguity if needed.
    if binary[:int(2 * etu)].mean() < 0.5:
        binary ^= 1

    frames = _frames_from_binary(binary, sample_rate)
    if not frames:
        return None
    frame = frames[0]
    fields = b_frames.parse_tag(frame["bytes"])
    return fields, r0 + frame["start"]


def decode(x: np.ndarray, sample_rate: float) -> List[Frame]:
    env = np.abs(np.asarray(x))
    rframes = reader_frames(env, sample_rate)
    out: List[Frame] = []
    margin = max(4, int(2e-6 * sample_rate))

    for j, rf in enumerate(rframes):
        info = b_frames.parse_reader(rf["bytes"])
        out.append(Frame(
            timestamp=rf["start"] / sample_rate,
            band="hf",
            direction="R->T",
            bits="".join(f"{byte:08b}" for byte in rf["bytes"]),
            fields={**info, "protocol": "iso14443b"},
            crc_ok=info.get("crc_ok"),
        ))

        gap_start = rf["end"] + margin
        gap_end = rframes[j + 1]["start"] - margin if j + 1 < len(rframes) else env.size
        if gap_end - gap_start < 16 * _etu(sample_rate):
            continue
        reply = decode_tag(x[gap_start:gap_end], sample_rate)
        if reply is None:
            continue
        fields, offset = reply
        out.append(Frame(
            timestamp=(gap_start + offset) / sample_rate,
            band="hf",
            direction="T->R",
            bits="".join(f"{int(b):08b}" for b in bytes.fromhex(fields["bytes"])),
            fields={**fields, "protocol": "iso14443b"},
            crc_ok=fields.get("crc_ok"),
        ))
    return out
