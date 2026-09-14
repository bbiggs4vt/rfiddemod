"""ISO 15693-3 frame parsing (byte level).

All frames end in the ISO 15693 CRC (identical to CRC-B / X-25, appended
LSByte first). Reader frames: flags + command [+ parameters]. Tag frames:
flags [+ payload]; an Inventory response is flags + DSFID + 8-byte UID
(transmitted LSByte first, displayed E0...).
"""

from __future__ import annotations

from typing import Optional

from rfid_demod.common import crc

COMMANDS = {
    0x01: "Inventory",
    0x02: "StayQuiet",
    0x20: "ReadSingleBlock",
    0x21: "WriteSingleBlock",
    0x22: "LockBlock",
    0x23: "ReadMultipleBlocks",
    0x25: "Select",
    0x26: "ResetToReady",
    0x27: "WriteAFI",
    0x2B: "GetSystemInfo",
}


def parse_reader(data: bytes) -> dict:
    info: dict = {"bytes": data.hex().upper()}
    if len(data) < 4:
        info["command"] = "unknown"
        return info
    info["crc_ok"] = crc.check_crc_b(data)
    cmd = data[1]
    info.update(
        command=COMMANDS.get(cmd, f"0x{cmd:02X}"),
        flags=f"{data[0]:02X}",
    )
    if len(data) > 4:
        info["data"] = data[2:-2].hex().upper()
    return info


def parse_tag(data: bytes, expected: Optional[str] = None) -> dict:
    info: dict = {"bytes": data.hex().upper()}
    if len(data) < 3:
        info["type"] = "raw"
        return info
    info["crc_ok"] = crc.check_crc_b(data)
    info["flags"] = f"{data[0]:02X}"
    if expected == "inventory" and len(data) == 12:
        info.update(
            type="InventoryResponse",
            dsfid=f"{data[1]:02X}",
            uid=data[2:10][::-1].hex().upper(),   # transmitted LSByte first
        )
        return info
    info["type"] = "response"
    if len(data) > 3:
        info["data"] = data[1:-2].hex().upper()
    return info
