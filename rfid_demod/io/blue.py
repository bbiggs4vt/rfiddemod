"""MIDAS BLUE (X-Midas) file reader.

BLUE files start with a fixed 512-byte header control block (HCB):
byte-order markers (``IEEE`` big / ``EEEI`` little endian) for header and
data, data offset/size, the file type (1000 = 1-D data), a two-character
data format (mode 'S'/'C' scalar/complex + element type), and a
type-specific adjunct at offset 256 — for type 1000: ``xstart``,
``xdelta`` (sample period, so sample_rate = 1/xdelta), ``xunits``.
An optional extended header carries keyword records (tag/type/value),
where capture tooling often stores the RF center frequency.

Layout follows the public X-Midas / REDHAWK ``bluefile`` definition.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

HCB_SIZE = 512

# Element type char -> (numpy base dtype, full-scale divisor for ints)
_ELEMENT = {
    "B": ("i1", 128.0),
    "I": ("i2", 32768.0),
    "L": ("i4", 2147483648.0),
    "F": ("f4", None),
    "D": ("f8", None),
}

# Extended-header keyword type char -> struct format
_KEY_TYPES = {"D": "d", "F": "f", "L": "i", "X": "q", "I": "h", "B": "b"}

_FREQ_TAGS = ("RF", "RFFREQ", "RF_FREQ", "FREQ", "CENTER_FREQ", "CENTERFREQ",
              "COLRF", "SNAP_RF", "VRF", "SBT", "FNOM")


def _endian(rep: bytes) -> str:
    if rep == b"EEEI":
        return "<"
    if rep == b"IEEE":
        return ">"
    raise ValueError(f"unknown BLUE byte-order marker {rep!r}")


def _parse_ext_keywords(raw: bytes, order: str) -> Dict[str, object]:
    """Best-effort extended-header keyword parse; malformed data is skipped.

    Record: int32 lkey (total bytes), int16 lext (header+tag+pad bytes),
    int8 ltag, char type, then (lkey-lext) value bytes, then the tag.
    """
    keywords: Dict[str, object] = {}
    pos = 0
    while pos + 8 <= len(raw):
        try:
            lkey, lext, ltag, ktype = struct.unpack_from(order + "ihbc", raw, pos)
            if lkey < 8 or pos + lkey > len(raw):
                break
            data_len = lkey - lext
            value_bytes = raw[pos + 8:pos + 8 + data_len]
            tag = raw[pos + 8 + data_len:pos + 8 + data_len + ltag].decode(
                "ascii", "replace")
            ktype = ktype.decode("ascii", "replace")
            if ktype == "A":
                keywords[tag] = value_bytes.decode("ascii", "replace").rstrip("\x00")
            elif ktype in _KEY_TYPES:
                fmt = _KEY_TYPES[ktype]
                n = data_len // struct.calcsize(fmt)
                vals = struct.unpack_from(order + fmt * n, value_bytes)
                keywords[tag] = vals[0] if n == 1 else list(vals)
            pos += lkey
        except (struct.error, UnicodeError):
            break
    return keywords


def read_header(path: Union[str, Path]) -> Tuple[dict, Dict[str, object]]:
    """Parse the HCB (+ extended keywords). Returns (header, keywords)."""
    path = Path(path)
    with open(path, "rb") as fh:
        hcb = fh.read(HCB_SIZE)
        if len(hcb) < HCB_SIZE:
            raise ValueError(f"{path}: shorter than a BLUE header ({len(hcb)} bytes)")
        if hcb[0:4] not in (b"BLUE", b"blue"):
            raise ValueError(f"{path}: not a BLUE file (magic {hcb[0:4]!r})")

        horder = _endian(hcb[4:8])
        dorder = _endian(hcb[8:12])
        detached, = struct.unpack_from(horder + "i", hcb, 12)
        ext_start, ext_size = struct.unpack_from(horder + "ii", hcb, 24)
        data_start, data_size = struct.unpack_from(horder + "dd", hcb, 32)
        ftype, = struct.unpack_from(horder + "i", hcb, 48)
        fmt = hcb[52:54].decode("ascii", "replace")
        xstart, xdelta = struct.unpack_from(horder + "dd", hcb, 256)
        xunits, = struct.unpack_from(horder + "i", hcb, 272)

        keywords: Dict[str, object] = {}
        if ext_start > 0 and ext_size > 0:
            fh.seek(ext_start * 512)
            keywords = _parse_ext_keywords(fh.read(ext_size), horder)

    header = {
        "data_order": dorder,
        "detached": detached,
        "data_start": data_start,
        "data_size": data_size,
        "type": ftype,
        "format": fmt,
        "xstart": xstart,
        "xdelta": xdelta,
        "xunits": xunits,
    }
    return header, keywords


def _center_freq(keywords: Dict[str, object]) -> Optional[float]:
    for tag in _FREQ_TAGS:
        value = keywords.get(tag)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def load(path: Union[str, Path]):
    """Load a BLUE capture as (samples c64, sample_rate, center_freq, keywords)."""
    path = Path(path)
    header, keywords = read_header(path)

    if header["detached"]:
        raise ValueError(
            f"{path}: detached BLUE file (data lives in a separate file); "
            "point the loader at a self-contained file")
    if header["type"] // 1000 != 1:
        raise ValueError(
            f"{path}: BLUE type {header['type']} is not 1-D data "
            "(only type 1000 files are supported)")

    fmt = header["format"].upper()
    mode, element = fmt[0], fmt[1]
    if element not in _ELEMENT:
        raise ValueError(f"{path}: unsupported BLUE element type {fmt!r}")
    if mode != "C":
        raise ValueError(
            f"{path}: scalar BLUE format {fmt!r} — the demodulator needs "
            "complex baseband IQ (mode 'C')")

    base, divisor = _ELEMENT[element]
    dtype = np.dtype(header["data_order"] + base)
    count = int(header["data_size"] // dtype.itemsize)
    raw = np.fromfile(path, dtype=dtype, count=count,
                      offset=int(header["data_start"]))
    if raw.size % 2:
        raw = raw[:-1]
    if divisor is not None:
        raw = raw.astype(np.float32) / divisor
    samples = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

    xdelta = header["xdelta"]
    sample_rate = 1.0 / xdelta if xdelta and xdelta > 0 else None
    return samples, sample_rate, _center_freq(keywords), keywords
