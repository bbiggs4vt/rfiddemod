"""HID Prox frames and Wiegand payloads.

Air format (FSK2a, 50 carrier cycles per FSK bit): a 96-bit frame that
repeats while the tag is powered — an 8-bit preamble ``00011101`` followed
by 88 Manchester chips carrying a 44-bit value. Inside that value a
sentinel '1' sits directly above the Wiegand payload, so
``bit_length - 1`` gives the payload width (26-37 bits for the classic
formats). Reference: Proxmark3 lfdemod.c / cmdlfhid.c.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

from rfid_demod.encodings import manchester

PREAMBLE = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
FRAME_BITS = 96          # preamble + 88 chips
PAYLOAD_BITS = 44
_MIN_WIEGAND = 26
_MAX_WIEGAND = 37


def _popcount(value: int) -> int:
    return bin(value).count("1")


def pack_h10301(facility_code: int, card_number: int) -> int:
    """Standard 26-bit Wiegand (H10301): PE + 8-bit FC + 16-bit CN + PO."""
    data = ((facility_code & 0xFF) << 16) | (card_number & 0xFFFF)
    pe = _popcount((data >> 12) & 0xFFF) & 1        # even over the high 12
    po = (_popcount(data & 0xFFF) & 1) ^ 1          # odd over the low 12
    return (pe << 25) | (data << 1) | po


def wiegand_to_hid(wiegand: int, bit_length: int) -> int:
    """Add the sentinel bit above the payload (the HID 44-bit value)."""
    if wiegand >> bit_length:
        raise ValueError("payload wider than bit_length")
    return (1 << bit_length) | wiegand


def frame_fsk_bits(hid_value: int, width: int = PAYLOAD_BITS) -> np.ndarray:
    """The 96 FSK bits of one air frame (for synthesis / tests)."""
    if hid_value >> width:
        raise ValueError(f"HID value wider than {width} bits")
    bits = np.array([(hid_value >> i) & 1 for i in range(width - 1, -1, -1)],
                    dtype=np.uint8)
    return np.concatenate([PREAMBLE, manchester.encode(bits)])


def unpack(hid_value: int) -> Optional[dict]:
    """Split sentinel + Wiegand payload; parse known formats."""
    if hid_value <= 0:
        return None
    bit_length = hid_value.bit_length() - 1
    if not _MIN_WIEGAND <= bit_length <= _MAX_WIEGAND:
        return None
    wiegand = hid_value & ((1 << bit_length) - 1)
    fields = {
        "raw": f"{hid_value:011X}",
        "bit_length": bit_length,
        "wiegand": f"{wiegand:0{(bit_length + 3) // 4}X}",
    }
    if bit_length == 26:
        data = (wiegand >> 1) & 0xFFFFFF
        fields.update(
            format="H10301",
            facility_code=(data >> 16) & 0xFF,
            card_number=data & 0xFFFF,
            parity_ok=wiegand == pack_h10301((data >> 16) & 0xFF, data & 0xFFFF),
        )
    return fields


def find_frames(
    bits: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> Iterator[Tuple[int, dict, bool, int]]:
    """Scan demodulated FSK bits (both polarities) for HID frames.

    Yields ``(bit_offset, fields, inverted, hid_value)``.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size
    if n < FRAME_BITS:
        return
    if valid is not None:
        vsum = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
        valid_window = (vsum[FRAME_BITS:] - vsum[:-FRAME_BITS]) == FRAME_BITS
    else:
        valid_window = None

    for inv in (False, True):
        stream = bits ^ 1 if inv else bits
        # Prefilter: vectorized preamble match; only candidates get the
        # Manchester decode attempt.
        match = np.ones(n - FRAME_BITS + 1, dtype=bool)
        for j, p in enumerate(PREAMBLE):
            match &= stream[j:j + n - FRAME_BITS + 1] == p
        if valid_window is not None:
            match &= valid_window
        for i in np.flatnonzero(match):
            try:
                payload = manchester.decode(stream[i + 8:i + FRAME_BITS])
            except ValueError:
                continue
            value = 0
            for b in payload:
                value = (value << 1) | int(b)
            fields = unpack(value)
            if fields is not None:
                yield int(i), fields, inv, value
