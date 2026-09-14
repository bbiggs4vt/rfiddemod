"""Common front end: retune -> carrier cancellation -> AGC -> decimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .agc import agc, normalize
from .carrier import dc_block
from .decimate import decimate_to
from .retune import retune

__all__ = ["FrontendConfig", "process", "retune", "dc_block", "agc", "normalize", "decimate_to"]


@dataclass
class FrontendConfig:
    sample_rate: float
    freq_offset: float = 0.0            # reader carrier offset from DC, Hz
    dc_block_cutoff_hz: Optional[float] = 1e3   # None disables carrier cancellation
    agc: bool = False                   # sliding AGC instead of peak normalization
    output_rate: Optional[float] = None  # decimate to this rate (None keeps input rate)


def process(x: np.ndarray, config: FrontendConfig) -> Tuple[np.ndarray, float]:
    """Run the common front end; returns ``(samples, output_sample_rate)``."""
    rate = float(config.sample_rate)

    if config.freq_offset:
        x = retune(x, config.freq_offset, rate)

    if config.dc_block_cutoff_hz is not None:
        x = dc_block(x, rate, config.dc_block_cutoff_hz)

    x = agc(x, rate) if config.agc else normalize(x)

    if config.output_rate is not None and config.output_rate < rate:
        x, rate = decimate_to(x, rate, config.output_rate)

    return x, rate
