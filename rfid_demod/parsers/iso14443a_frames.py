"""ISO 14443-3 type A frame bit codecs and command/reply parsing.

Short frames: 7 data bits, no parity (REQA 0x26, WUPA 0x52).
Standard frames: bytes LSB-first, each followed by an odd parity bit.
CRC-A covers full frames where present (Select, SAK, HLTA, APDUs);
anticollision UID chunks carry only a BCC (XOR of the four UID bytes).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from rfid_demod.common import crc

REQA = 0x26
WUPA = 0x52


def _odd_parity(byte: int) -> int:
    return (bin(byte).count("1") & 1) ^ 1


def short_frame_bits(command: int) -> np.ndarray:
    """7 data bits, LSB first."""
    return np.array([(command >> i) & 1 for i in range(7)], dtype=np.uint8)


def standard_frame_bits(data: bytes) -> np.ndarray:
    """Each byte LSB first + odd parity bit."""
    bits: List[int] = []
    for byte in data:
        bits += [(byte >> i) & 1 for i in range(8)]
        bits.append(_odd_parity(byte))
    return np.array(bits, dtype=np.uint8)


def parse_frame_bits(bits: np.ndarray) -> dict:
    """Classify a decoded bit stream as short / standard / raw."""
    b = [int(x) for x in bits]
    n = len(b)
    if n == 7:
        return {"kind": "short", "value": sum(bit << i for i, bit in enumerate(b))}
    if n and n % 9 == 0:
        data = bytearray()
        parity_ok = True
        for k in range(n // 9):
            group = b[9 * k:9 * k + 9]
            byte = sum(bit << i for i, bit in enumerate(group[:8]))
            parity_ok &= group[8] == _odd_parity(byte)
            data.append(byte)
        return {"kind": "standard", "bytes": bytes(data), "parity_ok": bool(parity_ok)}
    return {"kind": "raw", "length": n}


def _bcc_ok(chunk: bytes) -> bool:
    return (chunk[0] ^ chunk[1] ^ chunk[2] ^ chunk[3]) == chunk[4]


_CASCADE = {0x93: 1, 0x95: 2, 0x97: 3}


def parse_reader(frame: dict) -> dict:
    """Frame dict from parse_frame_bits -> reader command fields."""
    if frame["kind"] == "short":
        value = frame["value"]
        name = {REQA: "REQA", WUPA: "WUPA"}.get(value, "short")
        return {"command": name, "value": f"{value:02X}"}
    if frame["kind"] != "standard":
        return {"command": "unknown"}

    data = frame["bytes"]
    info: dict = {"bytes": data.hex().upper(), "parity_ok": frame["parity_ok"]}
    if len(data) >= 2 and data[0] in _CASCADE:
        level = _CASCADE[data[0]]
        if data[1] == 0x20 and len(data) == 2:
            info.update(command="Anticollision", cascade_level=level)
            return info
        if data[1] == 0x70 and len(data) == 9:
            info.update(
                command="Select", cascade_level=level,
                uid=data[2:6].hex().upper(),
                bcc_ok=_bcc_ok(data[2:7]),
                crc_ok=crc.check_crc_a(data),
            )
            return info
    if len(data) == 4 and data[0] == 0x50 and data[1] == 0x00:
        info.update(command="HLTA", crc_ok=crc.check_crc_a(data))
        return info
    if len(data) >= 2 and data[0] == 0xE0:
        info.update(command="RATS", crc_ok=crc.check_crc_a(data) if len(data) >= 4 else None)
        return info
    info["command"] = "unknown"
    if len(data) >= 3:
        info["crc_ok"] = crc.check_crc_a(data)
    return info


def parse_tag(frame: dict, expected: Optional[str] = None) -> dict:
    """Frame dict from parse_frame_bits -> tag reply fields."""
    if frame["kind"] != "standard":
        return {"type": "raw"}

    data = frame["bytes"]
    info: dict = {"bytes": data.hex().upper(), "parity_ok": frame["parity_ok"]}
    if expected == "atqa" and len(data) == 2:
        info.update(type="ATQA", atqa=f"{data[1]:02X}{data[0]:02X}")  # LSByte first
        return info
    if expected == "uid" and len(data) == 5:
        info.update(type="UID", uid=data[0:4].hex().upper(), bcc_ok=_bcc_ok(data))
        return info
    if expected == "sak" and len(data) == 3:
        info.update(type="SAK", sak=f"{data[0]:02X}", crc_ok=crc.check_crc_a(data))
        return info
    info["type"] = "raw"
    if len(data) >= 3:
        info["crc_ok"] = crc.check_crc_a(data)
    return info
