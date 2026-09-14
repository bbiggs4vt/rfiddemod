"""ISO 14443-3 type B frame parsing (byte level).

Character framing (start bit 0 + 8 data bits LSB-first + stop bit 1,
between an SOF and EOF of 10-11 etu of logic 0) is handled by the HF
decoder; this module interprets the resulting bytes. All B frames carry
CRC-B over everything before the CRC.
"""

from __future__ import annotations

from rfid_demod.common import crc


def parse_reader(data: bytes) -> dict:
    info: dict = {"bytes": data.hex().upper()}
    if len(data) >= 3:
        info["crc_ok"] = crc.check_crc_b(data)
    if len(data) == 5 and data[0] == 0x05:
        wup = bool(data[2] & 0x08)
        info.update(
            command="WUPB" if wup else "REQB",
            afi=data[1],
            slots=1 << (data[2] & 0x07),
        )
        return info
    if len(data) >= 11 and data[0] == 0x1D:
        info.update(command="ATTRIB", pupi=data[1:5].hex().upper())
        return info
    if len(data) == 7 and data[0] == 0x50:
        info.update(command="HLTB", pupi=data[1:5].hex().upper())
        return info
    info["command"] = "unknown"
    return info


def parse_tag(data: bytes, expected=None) -> dict:
    info: dict = {"bytes": data.hex().upper()}
    if len(data) >= 3:
        info["crc_ok"] = crc.check_crc_b(data)
    if len(data) == 14 and data[0] == 0x50:
        info.update(
            type="ATQB",
            pupi=data[1:5].hex().upper(),
            app_data=data[5:9].hex().upper(),
            protocol_info=data[9:12].hex().upper(),
        )
        return info
    info["type"] = "raw"
    return info
