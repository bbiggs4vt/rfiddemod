"""Decimation to the band's working rate.

Working rates from the brief: LF 1 Msps, HF 13.56/27.12 Msps, UHF 2 Msps.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Tuple

import numpy as np
from scipy.signal import resample_poly


def decimate_to(
    x: np.ndarray,
    in_rate: float,
    out_rate: float,
    max_denominator: int = 256,
) -> Tuple[np.ndarray, float]:
    """Polyphase-resample ``x`` from ``in_rate`` to (approximately) ``out_rate``.

    The rational approximation of the ratio is limited to ``max_denominator``;
    the achieved rate is returned alongside the samples.
    """
    if out_rate >= in_rate:
        return np.asarray(x), float(in_rate)
    ratio = Fraction(out_rate / in_rate).limit_denominator(max_denominator)
    if ratio.numerator == 0:
        raise ValueError(f"decimation ratio {out_rate}/{in_rate} too extreme")
    y = resample_poly(np.asarray(x), ratio.numerator, ratio.denominator)
    if np.iscomplexobj(y):
        y = y.astype(np.complex64)
    return y, float(in_rate) * ratio.numerator / ratio.denominator
