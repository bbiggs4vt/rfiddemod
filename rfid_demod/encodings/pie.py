"""PIE — pulse-interval encoding (EPC Gen2 reader->tag).

Every symbol ends in a low pulse of width PW; the information is in the
symbol duration: data-0 lasts 1 Tari, data-1 lasts 1.5-2 Tari
(Tari = 6.25-25 us). A preamble (delimiter + data-0 + RTcal + TRcal)
starts an inventory round; a frame-sync (same, without TRcal) precedes
every other command. RTcal = data-0 + data-1 lengths; TRcal is
1.1x-3x RTcal and sets the tag's backscatter link frequency.
"""

from __future__ import annotations

import numpy as np


def _symbol(total: int, pw: int) -> np.ndarray:
    sym = np.ones(total, dtype=np.uint8)
    sym[-pw:] = 0
    return sym


def preamble(
    samples_per_tari: int,
    delim_samples: int,
    trcal_taris: float,
    data1_ratio: float = 2.0,
    pw_ratio: float = 0.5,
) -> np.ndarray:
    """Preamble levels: delimiter low + data-0 + RTcal + TRcal symbols."""
    pw = max(1, round(pw_ratio * samples_per_tari))
    rtcal = round((1.0 + data1_ratio) * samples_per_tari)
    trcal = round(trcal_taris * samples_per_tari)
    return np.concatenate([
        np.zeros(delim_samples, dtype=np.uint8),
        _symbol(samples_per_tari, pw),
        _symbol(rtcal, pw),
        _symbol(trcal, pw),
    ])


def frame_sync(
    samples_per_tari: int,
    delim_samples: int,
    data1_ratio: float = 2.0,
    pw_ratio: float = 0.5,
) -> np.ndarray:
    """Frame-sync levels: delimiter low + data-0 + RTcal (no TRcal)."""
    pw = max(1, round(pw_ratio * samples_per_tari))
    rtcal = round((1.0 + data1_ratio) * samples_per_tari)
    return np.concatenate([
        np.zeros(delim_samples, dtype=np.uint8),
        _symbol(samples_per_tari, pw),
        _symbol(rtcal, pw),
    ])


def encode(
    bits: np.ndarray,
    samples_per_tari: int,
    data1_ratio: float = 2.0,
    pw_ratio: float = 0.5,
) -> np.ndarray:
    """Encode bits to a 0/1 level waveform at ``samples_per_tari`` resolution."""
    if not 1.5 <= data1_ratio <= 2.0:
        raise ValueError("data-1 length must be 1.5-2.0 Tari")
    pw = max(1, round(pw_ratio * samples_per_tari))
    len0 = int(samples_per_tari)
    len1 = round(data1_ratio * samples_per_tari)
    if pw >= len0:
        raise ValueError("PW must be shorter than a data-0 symbol")

    sym0 = np.ones(len0, dtype=np.uint8)
    sym0[-pw:] = 0
    sym1 = np.ones(len1, dtype=np.uint8)
    sym1[-pw:] = 0

    bits = np.asarray(bits, dtype=np.uint8)
    return np.concatenate([sym1 if b else sym0 for b in bits]) if bits.size else np.empty(0, np.uint8)


def symbol_samples(samples_per_tari: int, data1_ratio: float = 2.0):
    """(data-0, data-1) symbol lengths in samples — handy for decoders/tests."""
    return int(samples_per_tari), round(data1_ratio * samples_per_tari)
