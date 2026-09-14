"""LF ASK path: envelope -> threshold -> clocked chips -> EM4100 frames.

Timing is carrier-derived per the brief: the bit clock is one of RF/16,
RF/32, RF/64, RF/128 (carrier cycles per bit), picked by scoring the edge
spacing histogram against each candidate — no free-running recovery.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from rfid_demod.common import edges as edges_mod
from rfid_demod.common import envelope as env_mod
from rfid_demod.io import Frame
from rfid_demod.parsers import em4100

CLOCK_RF = (16, 32, 64, 128)     # carrier cycles per bit


def detect_half_bit(
    spacings: np.ndarray,
    candidates: List[float],
    tol: float = 0.25,
    min_score: float = 0.5,
) -> Optional[float]:
    """Pick the half-bit period whose {1x, 2x} grid explains most spacings."""
    spacings = np.asarray(spacings, dtype=np.float64)
    if spacings.size < 8:
        return None
    best_score, best_hb = 0.0, None
    for hb in candidates:  # ascending: the smaller clock wins ties
        k = np.round(spacings / hb)
        ok = (k >= 1) & (k <= 2) & (np.abs(spacings - k * hb) <= tol * hb)
        score = float(ok.mean())
        if score > best_score:
            best_score, best_hb = score, hb
    return best_hb if best_score >= min_score else None


def runs_to_chips(
    binary: np.ndarray,
    half_bit: float,
    tol: float = 0.35,
    max_chips_per_run: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize level runs to half-bit chips.

    Returns ``(chips, valid, start_samples)``. Runs that are not close to
    1 or 2 half-bits (glitches, idle stretches) yield invalid chips; the
    frame scanner skips regions containing them.
    """
    positions, _ = edges_mod.find_edges(binary)
    bounds = np.concatenate(([0], positions, [binary.size]))
    chips: List[int] = []
    valid: List[bool] = []
    starts: List[int] = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        run = e - s
        n = int(round(run / half_bit))
        good = 1 <= n <= 2 and abs(run - n * half_bit) <= tol * half_bit
        n = min(max(n, 1), max_chips_per_run)
        level = int(binary[s])
        for i in range(n):
            chips.append(level)
            valid.append(good)
            starts.append(s + int(round(i * half_bit)))
    return (
        np.array(chips, dtype=np.uint8),
        np.array(valid, dtype=bool),
        np.array(starts, dtype=np.int64),
    )


def lenient_manchester(
    chips: np.ndarray,
    chip_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise Manchester decode with a validity mask instead of raising."""
    m = chips.size // 2
    a = chips[0:2 * m:2]
    b = chips[1:2 * m:2]
    ok = (a != b) & chip_valid[0:2 * m:2] & chip_valid[1:2 * m:2]
    return a.copy(), ok


def decode_ask(
    env: np.ndarray,
    sample_rate: float,
    carrier_freq: float,
) -> List[Frame]:
    """EM4100 over ASK/Manchester. Returns one Frame per detected repeat."""
    env = env_mod.moving_average(env, max(1, int(sample_rate / carrier_freq)))
    binary = env_mod.to_binary(env, hysteresis=0.1)
    positions, _ = edges_mod.find_edges(binary)
    if positions.size < 8:
        return []

    candidates = [rf / 2 * sample_rate / carrier_freq for rf in CLOCK_RF]
    half_bit = detect_half_bit(edges_mod.edge_spacings(positions), candidates)
    if half_bit is None:
        return []

    chips, chip_valid, starts = runs_to_chips(binary, half_bit)

    frames: List[Frame] = []
    for align in (0, 1):
        bits, mask = lenient_manchester(chips[align:], chip_valid[align:])
        for offset, fields, inverted in em4100.find_frames(bits, mask):
            chip_index = align + 2 * offset
            raw = bits[offset:offset + em4100.FRAME_BITS]
            if inverted:
                raw = raw ^ 1
            frames.append(Frame(
                timestamp=float(starts[chip_index]) / sample_rate,
                band="lf",
                direction="T->R",
                bits="".join(map(str, raw)),
                fields={
                    **fields,
                    "protocol": "em4100",
                    "encoding": "manchester",
                    "rf_clock": int(round(2 * half_bit * carrier_freq / sample_rate)),
                    "inverted": inverted,
                },
                crc_ok=True,   # row + column parity validated
            ))
    return frames
