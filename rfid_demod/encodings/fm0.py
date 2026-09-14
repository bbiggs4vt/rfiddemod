"""FM0 (bi-phase space) — EPC Gen2 tag-to-reader baseband.

Transition at every symbol boundary; '0' carries an extra mid-symbol
transition. Identical to biphase with ``space=True``.

The Gen2 preamble (``1010v1`` with its violation) and the dummy-1
terminator belong to the UHF decoder (build-order step 3), not the codec.
"""

from __future__ import annotations

import numpy as np

from . import biphase


def encode(bits: np.ndarray, initial_level: int = 1) -> np.ndarray:
    return biphase.encode(bits, initial_level=initial_level, space=True)


def decode(chips: np.ndarray) -> np.ndarray:
    return biphase.decode(chips, space=True)
