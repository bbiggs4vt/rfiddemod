"""Edge detection on binary waveforms.

Per the key design rule in the brief, RFID timing is carrier-derived: symbol
periods are fixed sample counts, and edge-spacing statistics are used to
identify the clock (e.g. RF/32 vs RF/64 at LF), not to track it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def find_edges(binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transitions of a 0/1 waveform.

    Returns ``(positions, polarities)``: ``positions[i]`` is the index of the
    first sample after the i-th transition; ``polarities[i]`` is +1 for a
    rising edge and -1 for a falling edge.
    """
    d = np.diff(np.asarray(binary, dtype=np.int8))
    idx = np.flatnonzero(d)
    return idx + 1, d[idx]


def level_runs(binary: np.ndarray, level: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Maximal runs of ``level`` in a 0/1 waveform: ``(starts, lengths)``."""
    b = np.asarray(binary, dtype=np.int8)
    positions, _ = find_edges(b)
    bounds = np.concatenate(([0], positions, [b.size]))
    starts, lengths = [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if b[s] == level:
            starts.append(s)
            lengths.append(e - s)
    return np.array(starts, dtype=np.int64), np.array(lengths, dtype=np.int64)


def edge_spacings(positions: np.ndarray) -> np.ndarray:
    """Sample counts between consecutive edges."""
    return np.diff(np.asarray(positions))


def spacing_histogram(spacings: np.ndarray, tolerance: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram of edge spacings, optionally quantized to ``tolerance`` bins.

    Used to auto-detect the symbol period (e.g. the RF/32 vs RF/64 decision
    for LF captures). Returns ``(values, counts)`` sorted by value.
    """
    spacings = np.asarray(spacings)
    if tolerance > 1:
        spacings = (spacings // tolerance) * tolerance
    values, counts = np.unique(spacings, return_counts=True)
    return values, counts
