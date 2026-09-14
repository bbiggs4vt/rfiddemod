"""Synthetic IQ generation: bits -> chips -> IQ with controllable impairments.

Per the testing strategy in the brief, every decoder gets a round-trip test
(encode -> modulate -> demod -> compare) with controllable SNR and carrier
leakage, unblocking development before real captures exist.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def random_bits(n: int, seed: Optional[int] = None) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n, dtype=np.uint8)


def upsample(chips: np.ndarray, samples_per_chip: int) -> np.ndarray:
    """Hold each chip level for ``samples_per_chip`` samples."""
    return np.repeat(np.asarray(chips), samples_per_chip)


def _apply_offset(x: np.ndarray, freq_offset: float, sample_rate: float) -> np.ndarray:
    if not freq_offset:
        return x
    n = np.arange(len(x))
    return x * np.exp(2j * np.pi * freq_offset * n / sample_rate)


def ask_iq(
    levels: np.ndarray,
    mod_index: float = 1.0,
    freq_offset: float = 0.0,
    sample_rate: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Reader-style ASK/OOK: level 1 -> full carrier, level 0 -> 1 - mod_index.

    ``mod_index=1.0`` is OOK (14443A reader pulses); ~0.1 models the 10% ASK
    of 14443B/15693.
    """
    amp = (1.0 - mod_index) + mod_index * np.asarray(levels, dtype=np.float64)
    x = amp * np.exp(1j * phase)
    return _apply_offset(x, freq_offset, sample_rate).astype(np.complex64)


def backscatter_iq(
    levels: np.ndarray,
    carrier_amp: float = 1.0,
    mod_amp: float = 0.02,
    carrier_phase: float = 0.0,
    mod_phase: float = 0.0,
    freq_offset: float = 0.0,
    sample_rate: float = 1.0,
) -> np.ndarray:
    """Tag reply riding on a large reader carrier (30-60 dB above the tag).

    ``mod_phase`` sets the backscatter channel phase relative to the carrier
    — the reason carrier cancellation exists.
    """
    levels = np.asarray(levels, dtype=np.float64)
    x = carrier_amp * np.exp(1j * carrier_phase) + mod_amp * levels * np.exp(1j * mod_phase)
    return _apply_offset(x, freq_offset, sample_rate).astype(np.complex64)


def awgn(x: np.ndarray, snr_db: float, seed: Optional[int] = None) -> np.ndarray:
    """Add complex white Gaussian noise at ``snr_db`` relative to mean signal power."""
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    power = float(np.mean(np.abs(x) ** 2))
    noise_power = power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return (x + noise).astype(np.complex64)
