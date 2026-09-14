"""LF decoder (125 / 134.2 kHz): EM4100 (ASK/Manchester) and HID Prox (FSK2a).

Both paths run over the magnitude envelope of the baseband IQ — at LF the
reader carrier itself carries the tag's load modulation, so carrier
cancellation must stay off for this band (the CLI handles that default).

Still to come: biphase / PSK1 chip decode and raw T5577 block dumps
(tracked for a later pass of step 2+), per the brief's encoder list.
Reference: Proxmark3 client/src/cmddata.c, cmdlf*.c.
"""

from __future__ import annotations

from typing import List

import numpy as np

from rfid_demod.common import envelope as env_mod
from rfid_demod.io import Frame

from .ask_path import decode_ask
from .fsk_path import decode_fsk

DEFAULT_CARRIER = 125_000.0


def decode(
    samples: np.ndarray,
    sample_rate: float,
    carrier_freq: float = DEFAULT_CARRIER,
) -> List[Frame]:
    """Decode all LF frames found in a capture, sorted by timestamp."""
    env = env_mod.envelope(np.asarray(samples))
    frames = decode_ask(env, sample_rate, carrier_freq)
    frames += decode_fsk(env, sample_rate, carrier_freq)
    frames.sort(key=lambda f: f.timestamp)
    return frames
