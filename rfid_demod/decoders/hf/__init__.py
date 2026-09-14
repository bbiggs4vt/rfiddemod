"""HF decoder (13.56 MHz): ISO 14443A/B, ISO 15693.

Plan (build-order step 4): 14443A first (modified Miller R->T, fc/16
subcarrier Manchester T->R, CRC-A), then 14443B (NRZ-L / BPSK subcarrier,
CRC-B), then 15693 (PPM R->T, single-subcarrier Manchester T->R first).
Reference: Proxmark3 armsrc/iso14443a.c, iso15693.c; libnfc.
"""

from __future__ import annotations

import numpy as np


def decode(samples: np.ndarray, sample_rate: float):
    raise NotImplementedError(
        "HF decoding (14443A -> 14443B -> 15693) lands in build-order step 4"
    )
