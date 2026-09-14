"""Gen2 tag-side decode: backscatter in the CW gaps between commands.

Per response window: subtract the window mean (per-gap carrier
cancellation, gr-rfid gate style), estimate the backscatter channel phase
from ``angle(mean(r^2))/2`` (the residual is a two-point BPSK-like
constellation), project onto that phase, then correlate against the
preamble and sample half-symbol units on the fixed BLF grid.

Preambles (TRext=0 / TRext=1):
* FM0: [12 zero bits +] the 6-symbol ``1010v1`` chip pattern
  ``1,1,0,1,0,0,1,0,0,0,1,1`` (matches gr-rfid's TAG_PREAMBLE).
* Miller: [4 or 16 zero bits] + ``010111``, generated with the same
  Miller-modulated-subcarrier convention as the synthesizer. Real-capture
  validation of this pilot interpretation is pending (like the HID FSK
  polarity note).

Both FM0 and Miller data decode from transitions only, so the 180-degree
phase ambiguity of the projection is irrelevant.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from rfid_demod.common import correlate as corr_mod
from rfid_demod.common import envelope as env_mod
from rfid_demod.encodings import fm0, miller

FM0_PREAMBLE = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1], dtype=np.uint8)
MILLER_PREAMBLE_BITS = [0, 1, 0, 1, 1, 1]
MIN_SCORE = 0.5


def fm0_preamble_chips(trext: int = 0) -> np.ndarray:
    pilot = np.tile([1, 0], 12).astype(np.uint8) if trext else \
        np.empty(0, dtype=np.uint8)
    return np.concatenate([pilot, FM0_PREAMBLE])


def miller_pre_bits(trext: int = 0) -> np.ndarray:
    pilot = 16 if trext else 4
    return np.array([0] * pilot + MILLER_PREAMBLE_BITS, dtype=np.uint8)


def fm0_reply_chips(bits: np.ndarray, trext: int = 0) -> np.ndarray:
    """Preamble + data + dummy-1 as half-bit chips (synthesis / tests)."""
    data = np.concatenate([np.asarray(bits, np.uint8), [1]])
    # preamble ends high, so the first data chip starts low (boundary rule)
    return np.concatenate([fm0_preamble_chips(trext), fm0.encode(data, initial_level=0)])


def miller_reply_halfcycles(bits: np.ndarray, m: int, trext: int = 0) -> np.ndarray:
    """Preamble + data + dummy-1 at half-subcarrier-cycle resolution."""
    all_bits = np.concatenate([miller_pre_bits(trext),
                               np.asarray(bits, np.uint8), [1]])
    return miller.modulate_subcarrier(all_bits, m, initial_level=1)


def _project(window: np.ndarray) -> np.ndarray:
    r = window - window.mean()
    theta = 0.5 * np.angle(np.mean(r * r))
    return np.real(r * np.exp(-1j * theta))


def decode_reply(
    window: np.ndarray,
    sample_rate: float,
    blf: float,
    m: int = 1,
    trext: int = 0,
) -> Optional[Tuple[np.ndarray, float, int]]:
    """Decode one CW gap. Returns ``(bits, score, start_sample)`` or None.

    ``bits`` includes the dummy-1 tail and any trailing junk decoded from
    CW after the reply — the frame parser slices what it needs.
    """
    unit = sample_rate / (2.0 * blf)   # half FM0 bit == half subcarrier cycle
    spc = max(2, int(round(unit)))

    if m == 1:
        pre = fm0_preamble_chips(trext)
    else:
        pre = miller.modulate_subcarrier(miller_pre_bits(trext), m, initial_level=1)

    template = np.repeat(pre.astype(np.float64) * 2.0 - 1.0, spc)
    if window.size < template.size + 4 * spc:
        return None

    y = _project(np.asarray(window))
    # Matched-filter before slicing: integrate most of a half-unit so a
    # single sample decision isn't at the mercy of the carrier noise floor
    # (the preamble correlation already integrates; the data must too).
    y_mf = env_mod.moving_average(y, max(1, int(round(0.75 * unit))))
    c = corr_mod.normalized_correlation(y, template)
    if c.size == 0:
        return None
    peak = int(np.argmax(np.abs(c)))
    score = float(np.abs(c[peak]))
    if score < MIN_SCORE:
        return None

    # Sample half-units on the fixed grid (float step: no cumulative drift).
    n_units = int((y.size - peak) / unit) - pre.size
    if n_units < 2:
        return None
    idx = peak + np.round((pre.size + np.arange(n_units) + 0.5) * unit).astype(int)
    idx = idx[idx < y_mf.size]
    units = (y_mf[idx] > 0).astype(np.uint8)

    if m == 1:
        pairs = units[:units.size // 2 * 2].reshape(-1, 2)
        bits = (pairs[:, 0] == pairs[:, 1]).astype(np.uint8)  # FM0: no mid flip = 1
    else:
        # Undo the subcarrier (tile phase continues from the preamble start),
        # then look for the mid-bit transition.
        g = pre.size + np.arange(units.size)
        base = units ^ (g % 2 == 0)
        upb = 2 * m                                  # half-cycles per bit
        nbits = base.size // upb
        blocks = base[:nbits * upb].reshape(nbits, 2, m)
        halves = blocks.mean(axis=2) > 0.5
        bits = (halves[:, 0] != halves[:, 1]).astype(np.uint8)

    return bits, score, peak
