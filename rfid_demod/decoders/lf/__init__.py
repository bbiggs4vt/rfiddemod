"""LF decoder (125 / 134.2 kHz): EM4100, HID Prox (FSK2a), T5577.

Plan (build-order step 2): envelope -> low-pass at ~10x bitrate ->
threshold; bit period (RF/32 vs RF/64) auto-detected from the edge-spacing
histogram; Manchester/biphase/PSK1/FSK2a chip decode; EM4100 and HID
Wiegand parsers. Reference: Proxmark3 cmddata.c / cmdlf*.c.
"""

from __future__ import annotations

import numpy as np


def decode(samples: np.ndarray, sample_rate: float):
    raise NotImplementedError(
        "LF decoding (EM4100 Manchester -> HID FSK) lands in build-order step 2"
    )
