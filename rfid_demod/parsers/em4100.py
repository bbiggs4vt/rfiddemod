"""EM4100 frame structure.

64 bits = 9 header ones + 10 rows x (4 data + 1 even row parity)
+ 4 even column parities + stop bit (0). Payload: 40-bit ID whose first
byte is the version/customer ID.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

FRAME_BITS = 64
_HEADER = 9
_ROWS = 10


def encode(id40: int) -> np.ndarray:
    """40-bit ID -> 64-bit frame (also used to program T5577 clones later)."""
    if not 0 <= id40 < (1 << 40):
        raise ValueError("EM4100 ID must fit in 40 bits")
    nibbles = [(id40 >> (4 * i)) & 0xF for i in range(9, -1, -1)]

    bits = [1] * _HEADER
    for nib in nibbles:
        row = [(nib >> 3) & 1, (nib >> 2) & 1, (nib >> 1) & 1, nib & 1]
        bits.extend(row + [sum(row) & 1])
    for col in range(4):
        bits.append(sum((nib >> (3 - col)) & 1 for nib in nibbles) & 1)
    bits.append(0)
    return np.array(bits, dtype=np.uint8)


def parse_frame(bits: np.ndarray) -> Optional[dict]:
    """Validate one 64-bit window; returns parsed fields or None."""
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size != FRAME_BITS:
        return None
    if not bits[:_HEADER].all() or bits[63] != 0:
        return None

    rows = bits[_HEADER:_HEADER + 5 * _ROWS].reshape(_ROWS, 5)
    if (rows.sum(axis=1) % 2).any():
        return None
    data = rows[:, :4]
    col_parity = bits[59:63]
    if ((data.sum(axis=0) + col_parity) % 2).any():
        return None

    id40 = 0
    for nibble_bits in data:
        id40 = (id40 << 4) | int(nibble_bits[0]) << 3 | int(nibble_bits[1]) << 2 \
            | int(nibble_bits[2]) << 1 | int(nibble_bits[3])
    return {
        "id": f"{id40:010X}",
        "version": (id40 >> 32) & 0xFF,
        "data": id40 & 0xFFFFFFFF,
    }


def find_frames(
    bits: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> Iterator[Tuple[int, dict, bool]]:
    """Scan a demodulated bitstream (both polarities) for valid frames.

    Yields ``(bit_offset, fields, inverted)``. The stream repeats while the
    tag is powered, so multiple hits for the same ID are expected.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    for i in range(0, bits.size - FRAME_BITS + 1):
        if valid is not None and not valid[i:i + FRAME_BITS].all():
            continue
        window = bits[i:i + FRAME_BITS]
        for inv in (False, True):
            fields = parse_frame(window ^ 1 if inv else window)
            if fields is not None:
                yield i, fields, inv
                break
