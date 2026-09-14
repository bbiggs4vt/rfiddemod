"""EPC Gen2 / ISO 18000-6C frame building and parsing (bit level).

Reader commands (R->T) and tag replies (T->R). Builders exist for the
commands the synthetic tests exercise; the parser also handles Select
(EBV pointer) and falls back to a raw record for unknown prefixes.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from rfid_demod.common import crc

# Query M field -> cycles per symbol (FM0 is M=1).
M_CODES = {0: 1, 1: 2, 2: 4, 3: 8}
# Query DR field -> divide ratio; BLF = DR / TRcal.
DR_RATIOS = {0: 8.0, 1: 64.0 / 3.0}


def _bits(value: int, width: int) -> List[int]:
    return [(value >> i) & 1 for i in range(width - 1, -1, -1)]


def _val(bits: Sequence[int]) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def _hex(bits: Sequence[int]) -> str:
    return f"{_val(bits):0{(len(bits) + 3) // 4}X}"


# ---------------------------------------------------------------- builders

def build_query(dr=0, m=0, trext=0, sel=0, session=0, target=0, q=0) -> np.ndarray:
    body = ([1, 0, 0, 0] + _bits(dr, 1) + _bits(m, 2) + _bits(trext, 1)
            + _bits(sel, 2) + _bits(session, 2) + _bits(target, 1) + _bits(q, 4))
    return np.array(body + crc.crc5_gen2_bits(body), dtype=np.uint8)


def build_query_rep(session=0) -> np.ndarray:
    return np.array([0, 0] + _bits(session, 2), dtype=np.uint8)


def build_query_adjust(session=0, updn=0) -> np.ndarray:
    return np.array([1, 0, 0, 1] + _bits(session, 2) + _bits(updn, 3), dtype=np.uint8)


def build_ack(rn16: int) -> np.ndarray:
    return np.array([0, 1] + _bits(rn16, 16), dtype=np.uint8)


def build_nak() -> np.ndarray:
    return np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)


def build_req_rn(rn16: int) -> np.ndarray:
    body = [1, 1, 0, 0, 0, 0, 0, 1] + _bits(rn16, 16)
    return np.array(body + _bits(crc.crc16_gen2_bits(body), 16), dtype=np.uint8)


def build_select(target=0, action=0, membank=1, pointer=0x20,
                 mask_bits: Sequence[int] = (), truncate=0) -> np.ndarray:
    mask = list(mask_bits)
    body = ([1, 0, 1, 0] + _bits(target, 3) + _bits(action, 3) + _bits(membank, 2)
            + _ebv(pointer) + _bits(len(mask), 8) + mask + _bits(truncate, 1))
    return np.array(body + _bits(crc.crc16_gen2_bits(body), 16), dtype=np.uint8)


def build_tag_epc_reply(epc_hex: str, pc: Optional[int] = None) -> np.ndarray:
    """PC + EPC + CRC-16 bits for a tag's ACK reply (synthesis / tests)."""
    epc_bits = []
    for ch in epc_hex:
        epc_bits += _bits(int(ch, 16), 4)
    if len(epc_bits) % 16:
        raise ValueError("EPC must be a whole number of 16-bit words")
    if pc is None:
        pc = (len(epc_bits) // 16) << 11
    payload = _bits(pc, 16) + epc_bits
    return np.array(payload + _bits(crc.crc16_gen2_bits(payload), 16), dtype=np.uint8)


# ------------------------------------------------------------------ EBV

def _ebv(value: int) -> List[int]:
    """Extensible bit vector: 7-bit groups, MSB of each byte = extension."""
    groups = [value & 0x7F]
    value >>= 7
    while value:
        groups.append(value & 0x7F)
        value >>= 7
    out: List[int] = []
    for i, g in enumerate(reversed(groups)):
        ext = 1 if i < len(groups) - 1 else 0
        out += [ext] + _bits(g, 7)
    return out


def _parse_ebv(bits: Sequence[int], idx: int):
    """Returns (value, next_index) or (None, idx) on truncation."""
    value = 0
    while True:
        if idx + 8 > len(bits):
            return None, idx
        ext = bits[idx]
        value = (value << 7) | _val(bits[idx + 1:idx + 8])
        idx += 8
        if not ext:
            return value, idx


# --------------------------------------------------------------- parsers

def parse_reader(bits: np.ndarray) -> dict:
    """Parse one R->T frame's bits into command + fields."""
    b = [int(x) for x in bits]
    n = len(b)

    if n == 4 and b[:2] == [0, 0]:
        return {"command": "QueryRep", "session": _val(b[2:4])}

    if n == 18 and b[:2] == [0, 1]:
        return {"command": "ACK", "rn16": _hex(b[2:18])}

    if n == 22 and b[:4] == [1, 0, 0, 0]:
        return {
            "command": "Query",
            "dr": _val(b[4:5]),
            "dr_ratio": DR_RATIOS[_val(b[4:5])],
            "m": M_CODES[_val(b[5:7])],
            "trext": b[7],
            "sel": _val(b[8:10]),
            "session": _val(b[10:12]),
            "target": b[12],
            "q": _val(b[13:17]),
            "crc_ok": crc.check_crc5_gen2(b),
        }

    if n == 9 and b[:4] == [1, 0, 0, 1]:
        return {"command": "QueryAdjust", "session": _val(b[4:6]), "updn": _val(b[6:9])}

    if n == 8 and b == [1, 1, 0, 0, 0, 0, 0, 0]:
        return {"command": "NAK"}

    if n == 40 and b[:8] == [1, 1, 0, 0, 0, 0, 0, 1]:
        return {
            "command": "Req_RN",
            "rn16": _hex(b[8:24]),
            "crc_ok": crc.crc16_gen2_bits(b[:24]) == _val(b[24:40]),
        }

    if n >= 44 and b[:4] == [1, 0, 1, 0]:
        info = {"command": "Select"}
        pointer, idx = _parse_ebv(b, 12)
        if pointer is not None and idx + 8 + 1 + 16 <= n:
            mask_len = _val(b[idx:idx + 8])
            end = idx + 8 + mask_len + 1 + 16
            if end == n:
                info.update(
                    target=_val(b[4:7]), action=_val(b[7:10]), membank=_val(b[10:12]),
                    pointer=pointer, mask_length=mask_len,
                    mask=_hex(b[idx + 8:idx + 8 + mask_len]) if mask_len else "",
                    truncate=b[idx + 8 + mask_len],
                    crc_ok=crc.crc16_gen2_bits(b[:n - 16]) == _val(b[n - 16:]),
                )
                return info
        info["error"] = "truncated or malformed Select"
        return info

    return {"command": "unknown", "length": n}


def parse_tag(bits: np.ndarray, expected: Optional[str] = None) -> Optional[dict]:
    """Parse a T->R bit stream (dummy-1 tail and trailing junk tolerated).

    ``expected`` comes from the preceding reader command: "rn16" after
    Query/QueryRep/QueryAdjust, "epc" after ACK, "handle" after Req_RN.
    Returns fields including ``bits_used`` (frame length actually consumed).
    """
    b = [int(x) for x in bits]
    n = len(b)

    if expected == "epc" and n >= 33:
        nwords = _val(b[0:5])
        total = 16 + 16 * nwords + 16
        if n >= total:
            payload, crc_bits = b[:total - 16], b[total - 16:total]
            return {
                "type": "EPC",
                "pc": _hex(b[:16]),
                "epc": _hex(b[16:total - 16]),
                "crc_ok": crc.crc16_gen2_bits(payload) == _val(crc_bits),
                "bits_used": total,
            }

    if expected == "handle" and n >= 32:
        return {
            "type": "Handle",
            "handle": _hex(b[:16]),
            "crc_ok": crc.crc16_gen2_bits(b[:16]) == _val(b[16:32]),
            "bits_used": 32,
        }

    if n >= 16:
        return {"type": "RN16", "rn16": _hex(b[:16]), "bits_used": 16}

    return None
