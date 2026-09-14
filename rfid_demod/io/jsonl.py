"""Decoded-frame records and the JSON-lines sink."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Optional, Union


@dataclass
class Frame:
    """One decoded RFID frame.

    timestamp  seconds from the start of the capture
    band       "lf" / "hf" / "uhf"
    direction  "R->T" (reader command) or "T->R" (tag reply)
    bits       raw bits as a '0'/'1' string, first-transmitted bit first
    fields     parsed fields (UID / EPC / command name, ...)
    crc_ok     CRC status; None when the frame type carries no CRC
    """

    timestamp: float
    band: str
    direction: str
    bits: str
    fields: dict = field(default_factory=dict)
    crc_ok: Optional[bool] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class JsonlWriter:
    """Write frames as JSON lines to a file or stdout (path ``-``)."""

    def __init__(self, path: Union[str, Path] = "-"):
        self._own = str(path) != "-"
        self._fp: IO[str] = open(path, "w") if self._own else sys.stdout
        self.count = 0

    def write(self, frame: Frame) -> None:
        self._fp.write(frame.to_json() + "\n")
        self.count += 1

    def close(self) -> None:
        if self._own:
            self._fp.close()
        else:
            self._fp.flush()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
