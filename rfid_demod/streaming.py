"""Streaming decode: feed IQ in arbitrary chunks, get frames out.

The band decoders are batch functions over a capture, so streaming works
on a sliding buffer: whenever enough samples accumulate, the buffer is
decoded and every frame that *starts* in the committed region (everything
but the trailing ``overlap``) is emitted; the overlap tail is retained and
re-decoded with the next block, so frames and command/reply exchanges
that straddle a block boundary are seen whole. Duplicate emissions from
the re-decoded overlap are suppressed by (direction, bits, ~timestamp).

Choose ``overlap`` >= the longest frame (and, for UHF/HF, the longest
command->reply exchange) so any frame starting inside the committed
region is fully contained in the buffer. Band defaults cover the
protocols implemented so far.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Dict, List, Optional

import numpy as np

from .io import Frame

# Longest frame / exchange the overlap must cover, per band (seconds):
# lf: an EM4100 frame at RF/64 is ~33 ms, a HID frame ~38 ms;
# uhf/hf: a command->reply exchange (context for the tag decoder).
DEFAULT_OVERLAP_S = {"lf": 0.06, "uhf": 0.05, "hf": 0.01}

_DEDUPE_TOL_S = 0.5e-3


class StreamDecoder:
    """Incremental band decoding over a sliding, overlapped buffer."""

    def __init__(
        self,
        band: str,
        sample_rate: float,
        block_seconds: float = 0.25,
        overlap_seconds: Optional[float] = None,
    ):
        self.band = band
        self.sample_rate = float(sample_rate)
        self._decode = importlib.import_module(f"rfid_demod.decoders.{band}").decode
        self.overlap = int((overlap_seconds if overlap_seconds is not None
                            else DEFAULT_OVERLAP_S[band]) * self.sample_rate)
        self.block = max(int(block_seconds * self.sample_rate), 2 * self.overlap)
        self._buffer = np.empty(0, dtype=np.complex64)
        self._base = 0                       # absolute sample index of buffer[0]
        self._seen: Dict[tuple, List[int]] = {}
        self._tol = max(1, int(_DEDUPE_TOL_S * self.sample_rate))

    def feed(self, samples: np.ndarray) -> List[Frame]:
        """Add samples; returns frames that became final with this data."""
        self._buffer = np.concatenate(
            [self._buffer, np.asarray(samples, dtype=np.complex64)])
        out: List[Frame] = []
        while self._buffer.size >= self.block:
            out += self._process(final=False)
        return out

    def flush(self) -> List[Frame]:
        """Decode whatever remains (end of stream)."""
        if self._buffer.size == 0:
            return []
        frames = self._process(final=True)
        self._buffer = np.empty(0, dtype=np.complex64)
        return frames

    def _is_duplicate(self, key: tuple, abs_sample: int) -> bool:
        starts = self._seen.setdefault(key, [])
        for s in starts:
            if abs(s - abs_sample) <= self._tol:
                return True
        starts.append(abs_sample)
        return False

    def _prune_seen(self) -> None:
        for key in list(self._seen):
            kept = [s for s in self._seen[key] if s >= self._base - self.overlap]
            if kept:
                self._seen[key] = kept
            else:
                del self._seen[key]

    def _process(self, final: bool) -> List[Frame]:
        frames = self._decode(self._buffer, self.sample_rate)
        limit = self._buffer.size if final else self._buffer.size - self.overlap

        out: List[Frame] = []
        for frame in frames:
            local_start = int(round(frame.timestamp * self.sample_rate))
            if not final and local_start >= limit:
                continue  # may be incomplete; re-decoded with the next block
            abs_start = self._base + local_start
            key = (frame.direction, frame.bits)
            if self._is_duplicate(key, abs_start):
                continue
            out.append(dataclasses.replace(
                frame, timestamp=abs_start / self.sample_rate))

        if not final:
            consumed = self._buffer.size - self.overlap
            self._buffer = self._buffer[consumed:]
            self._base += consumed
            self._prune_seen()
        return out
