"""Gen2 reader-side decode: PIE symbols from the envelope.

Strategy (gr-rfid style, carrier-derived timing): find delimiters (low
pulses of ~12.5 us preceded by CW), then measure the intervals between
successive rising edges — each PIE symbol ends with its PW low pulse, so
rising-edge spacing equals symbol duration. The first two intervals give
Tari and RTcal; a third longer than RTcal is TRcal (preamble -> Query
follows). Data symbols slice against the RTcal/2 pivot.
"""

from __future__ import annotations

from typing import List

import numpy as np

from rfid_demod.common import edges as edges_mod
from rfid_demod.common import envelope as env_mod

DELIM_US = 12.5
_DELIM_TOL = 0.3          # +-30% on the delimiter width
_TARI_RANGE_US = (6.25 * 0.7, 25.0 * 1.3)
_RTCAL_RANGE = (2.4, 3.3)  # x Tari (spec: 2.5-3)
_TRCAL_RANGE = (1.05, 3.3)  # x RTcal (spec: 1.1-3)


def reader_frames(env: np.ndarray, sample_rate: float) -> List[dict]:
    """Locate and slice all R->T frames in an envelope.

    Each dict: bits, tari, rtcal, trcal (None for frame-sync), start
    (delimiter falling edge, samples), end (rising edge closing the last
    symbol), preamble (bool).
    """
    smoothed = env_mod.moving_average(env, 3)
    # PIE lows (PW pulses) are a small fraction of samples, so the default
    # 10/90-percentile midpoint would sit inside the CW level; use extreme
    # percentiles to bracket the true low/high states.
    threshold = env_mod.midpoint_threshold(smoothed, lo_pct=0.5, hi_pct=99.5)
    binary = env_mod.to_binary(smoothed, threshold=threshold, hysteresis=0.15)
    positions, polarities = edges_mod.find_edges(binary)
    rising = positions[polarities > 0]
    falling = positions[polarities < 0]

    delim_lo = DELIM_US * 1e-6 * sample_rate * (1 - _DELIM_TOL)
    delim_hi = DELIM_US * 1e-6 * sample_rate * (1 + _DELIM_TOL)
    tari_lo = _TARI_RANGE_US[0] * 1e-6 * sample_rate
    tari_hi = _TARI_RANGE_US[1] * 1e-6 * sample_rate

    frames: List[dict] = []
    used_until = 0
    for f in falling:
        if f < used_until:
            continue
        r_idx = int(np.searchsorted(rising, f))
        if r_idx >= rising.size:
            break
        low_run = rising[r_idx] - f
        if not delim_lo <= low_run <= delim_hi:
            continue

        rs = rising[r_idx:]
        if rs.size < 4:
            continue
        intervals = np.diff(rs)

        tari = float(intervals[0])
        if not tari_lo <= tari <= tari_hi:
            continue
        rtcal = float(intervals[1])
        if not _RTCAL_RANGE[0] * tari <= rtcal <= _RTCAL_RANGE[1] * tari:
            continue

        k = 2
        trcal = None
        if k < intervals.size and \
                _TRCAL_RANGE[0] * rtcal < intervals[k] <= _TRCAL_RANGE[1] * rtcal:
            trcal = float(intervals[k])
            k += 1

        pivot = rtcal / 2.0
        bits: List[int] = []
        while k < intervals.size and intervals[k] <= _TRCAL_RANGE[0] * rtcal:
            if not 0.6 * tari <= intervals[k] <= 2.3 * tari:
                break  # not a legal data symbol; end of frame
            bits.append(1 if intervals[k] > pivot else 0)
            k += 1

        end = int(rs[min(k, intervals.size)])
        used_until = end
        if len(bits) < 4:
            continue  # shortest command (QueryRep) is 4 bits
        frames.append({
            "bits": np.array(bits, dtype=np.uint8),
            "tari": tari,
            "rtcal": rtcal,
            "trcal": trcal,
            "start": int(f),
            "end": end,
            "preamble": trcal is not None,
        })
    return frames
