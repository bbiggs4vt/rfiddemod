"""UHF decoder (EPC Gen2 / ISO 18000-6C).

Plan (build-order step 3): PIE reader decode (pulse widths vs Tari) ->
Query parsing to learn BLF = DR/TRcal and encoding (FM0 / Miller M) ->
FM0 tag decode -> Miller variants. Frames: RN16, then PC + EPC + CRC-16.
Reference: gr-rfid tag_decoder_impl.cc / gate_impl.cc.
"""

from __future__ import annotations

import numpy as np


def decode(samples: np.ndarray, sample_rate: float):
    raise NotImplementedError(
        "UHF Gen2 decoding (PIE -> Query -> FM0/Miller) lands in build-order step 3"
    )
