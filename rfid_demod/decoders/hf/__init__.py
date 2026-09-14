"""HF decoder (13.56 MHz): ISO 14443A, ISO 14443B, ISO 15693.

All three sub-decoders run over the capture; each has structural gates
(pause widths, marker lengths, symbol-grid validation) that keep it from
firing on the other protocols' waveforms. Reader commands come from the
carrier envelope; tag replies from the fc/16 or fc/32 subcarrier in the
gaps between commands.

Not yet implemented: ISO 15693 1-of-256 R->T coding, the dual-subcarrier
T->R mode, and 14443 higher bit rates (212-848 kbps).
Reference: Proxmark3 armsrc/iso14443a.c, iso14443b.c, iso15693.c; libnfc.
"""

from __future__ import annotations

from typing import List

import numpy as np

from rfid_demod.io import Frame

from . import iso14443a, iso14443b, iso15693


def decode(samples: np.ndarray, sample_rate: float) -> List[Frame]:
    from rfid_demod.decoders import check_rate

    check_rate("hf", sample_rate)
    x = np.asarray(samples)
    frames: List[Frame] = []
    frames += iso14443a.decode(x, sample_rate)
    frames += iso14443b.decode(x, sample_rate)
    frames += iso15693.decode(x, sample_rate)
    frames.sort(key=lambda f: f.timestamp)
    return frames
