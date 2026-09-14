"""CRC routines used across the RFID standards.

Named presets (RevEng catalogue names in parentheses; ``check`` is the CRC of
the ASCII string ``"123456789"``):

============  =====================================  ======
function      catalogue name                         check
============  =====================================  ======
crc16_ccitt   CRC-16/CCITT-FALSE                     0x29B1
crc16_gen2    CRC-16/GENIBUS (EPC Gen2 air value)    0xD64E
crc_a         CRC-16/ISO-IEC-14443-3-A               0xBF05
crc_b         CRC-16/ISO-IEC-14443-3-B (X-25)        0x906E
============  =====================================  ======

EPC Gen2 additionally uses a CRC-5 over the bits of the Query command
(:func:`crc5_gen2`).
"""

from __future__ import annotations

from typing import Iterable, List


def _reflect(value: int, width: int) -> int:
    out = 0
    for _ in range(width):
        out = (out << 1) | (value & 1)
        value >>= 1
    return out


def crc16(
    data: bytes,
    poly: int,
    init: int,
    refin: bool = False,
    refout: bool = False,
    xorout: int = 0x0000,
) -> int:
    """Generic bitwise CRC-16 (MSB-first shift register)."""
    crc = init & 0xFFFF
    for byte in data:
        if refin:
            byte = _reflect(byte, 8)
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    if refout:
        crc = _reflect(crc, 16)
    return crc ^ xorout


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection."""
    return crc16(data, 0x1021, 0xFFFF)


def crc16_gen2(data: bytes) -> int:
    """EPC Gen2 / ISO 18000-6C CRC-16.

    CCITT with ones-complemented output; the returned value is exactly what
    is transmitted on air (appended MSB-first).
    """
    return crc16(data, 0x1021, 0xFFFF, xorout=0xFFFF)


def crc_a(data: bytes) -> int:
    """ISO 14443-3 type A CRC (reflected 0x1021, init 0x6363)."""
    return crc16(data, 0x1021, 0xC6C6, refin=True, refout=True)


def crc_b(data: bytes) -> int:
    """ISO 14443-3 type B CRC (reflected 0x1021, init 0xFFFF, inverted out)."""
    return crc16(data, 0x1021, 0xFFFF, refin=True, refout=True, xorout=0xFFFF)


def append_crc_a(payload: bytes) -> bytes:
    """14443A frames carry the CRC LSB-first."""
    return payload + crc_a(payload).to_bytes(2, "little")


def check_crc_a(frame: bytes) -> bool:
    if len(frame) < 3:
        return False
    return crc_a(frame[:-2]) == int.from_bytes(frame[-2:], "little")


def append_crc_b(payload: bytes) -> bytes:
    return payload + crc_b(payload).to_bytes(2, "little")


def check_crc_b(frame: bytes) -> bool:
    if len(frame) < 3:
        return False
    return crc_b(frame[:-2]) == int.from_bytes(frame[-2:], "little")


def append_crc16_gen2(payload: bytes) -> bytes:
    """Gen2 transmits the CRC-16 MSB-first."""
    return payload + crc16_gen2(payload).to_bytes(2, "big")


def check_crc16_gen2(frame: bytes) -> bool:
    if len(frame) < 3:
        return False
    return crc16_gen2(frame[:-2]) == int.from_bytes(frame[-2:], "big")


_CRC5_POLY = 0b01001  # x^5 + x^3 + 1
_CRC5_INIT = 0b01001


def crc5_gen2(bits: Iterable[int]) -> int:
    """EPC Gen2 CRC-5 over a bit sequence (MSB / first-transmitted bit first)."""
    crc = _CRC5_INIT
    for bit in bits:
        feedback = ((crc >> 4) ^ (bit & 1)) & 1
        crc = (crc << 1) & 0x1F
        if feedback:
            crc ^= _CRC5_POLY
    return crc


def crc5_gen2_bits(bits: Iterable[int]) -> List[int]:
    """CRC-5 as the 5 bits to append, first-transmitted bit first."""
    crc = crc5_gen2(bits)
    return [(crc >> i) & 1 for i in range(4, -1, -1)]


def check_crc5_gen2(bits: Iterable[int]) -> bool:
    """True when `bits` (payload + appended CRC-5) has zero remainder."""
    return crc5_gen2(bits) == 0
