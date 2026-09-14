"""IQ file readers/writers.

Supported formats:

* ``cf32``  — raw interleaved complex float32 (GNU Radio ``.cfile``)
* ``ci16``  — raw interleaved int16 I/Q, full scale 32768
* ``wav``   — 2-channel WAV, channel 0 = I, channel 1 = Q
* ``sigmf`` — SigMF pair (``.sigmf-meta`` JSON + ``.sigmf-data``);
              datatypes ``cf32_le`` and ``ci16_le``

Everything is returned as complex64 in roughly [-1, 1].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

_CI16_SCALE = 32768.0

_SUFFIX_TO_FMT = {
    ".cf32": "cf32",
    ".fc32": "cf32",
    ".cfile": "cf32",
    ".ci16": "ci16",
    ".sc16": "ci16",
    ".iq16": "ci16",
    ".wav": "wav",
    ".sigmf-meta": "sigmf",
    ".sigmf-data": "sigmf",
}


@dataclass
class IQCapture:
    samples: np.ndarray                    # complex64
    sample_rate: Optional[float] = None    # Hz, when the container carries it
    center_freq: Optional[float] = None    # Hz, when the container carries it
    source: Optional[str] = None

    def __len__(self) -> int:
        return len(self.samples)


def _infer_format(path: Path) -> str:
    fmt = _SUFFIX_TO_FMT.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"cannot infer IQ format from suffix {path.suffix!r}; pass fmt= explicitly"
        )
    return fmt


def _ci16_to_complex64(raw: np.ndarray) -> np.ndarray:
    if raw.size % 2:
        raw = raw[:-1]
    x = raw.astype(np.float32) / _CI16_SCALE
    return (x[0::2] + 1j * x[1::2]).astype(np.complex64)


def _load_wav(path: Path) -> IQCapture:
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path}: need a 2-channel WAV (I, Q); got shape {data.shape}")
    data = data[:, :2]
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max))
        offset = (info.max + info.min + 1) / 2.0  # recenters unsigned formats
        data = (data.astype(np.float32) - offset) / scale
    else:
        data = data.astype(np.float32)
    samples = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
    return IQCapture(samples, sample_rate=float(rate), source=str(path))


def _load_sigmf(path: Path) -> IQCapture:
    stem = path.name
    for suffix in (".sigmf-meta", ".sigmf-data"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    meta_path = path.with_name(stem + ".sigmf-meta")
    data_path = path.with_name(stem + ".sigmf-data")

    meta = json.loads(meta_path.read_text())
    glob = meta.get("global", {})
    datatype = glob.get("core:datatype", "cf32_le")
    sample_rate = glob.get("core:sample_rate")
    captures = meta.get("captures", [])
    center_freq = captures[0].get("core:frequency") if captures else None

    if datatype == "cf32_le":
        samples = np.fromfile(data_path, dtype="<c8").astype(np.complex64)
    elif datatype == "ci16_le":
        samples = _ci16_to_complex64(np.fromfile(data_path, dtype="<i2"))
    else:
        raise ValueError(f"{meta_path}: unsupported SigMF datatype {datatype!r}")

    return IQCapture(
        samples,
        sample_rate=float(sample_rate) if sample_rate else None,
        center_freq=float(center_freq) if center_freq else None,
        source=str(data_path),
    )


def load(path: Union[str, Path], fmt: Optional[str] = None) -> IQCapture:
    """Load an IQ capture; format inferred from the suffix unless given."""
    path = Path(path)
    fmt = fmt or _infer_format(path)
    if fmt == "cf32":
        samples = np.fromfile(path, dtype=np.complex64)
        return IQCapture(samples, source=str(path))
    if fmt == "ci16":
        samples = _ci16_to_complex64(np.fromfile(path, dtype="<i2"))
        return IQCapture(samples, source=str(path))
    if fmt == "wav":
        return _load_wav(path)
    if fmt == "sigmf":
        return _load_sigmf(path)
    raise ValueError(f"unknown IQ format {fmt!r}")


def save(path: Union[str, Path], samples: np.ndarray, fmt: Optional[str] = None) -> None:
    """Write raw cf32/ci16 (mainly for test fixtures and synthetic captures)."""
    path = Path(path)
    fmt = fmt or _infer_format(path)
    samples = np.asarray(samples)
    if fmt == "cf32":
        samples.astype(np.complex64).tofile(path)
    elif fmt == "ci16":
        interleaved = np.empty(2 * len(samples), dtype="<i2")
        re = np.clip(np.round(samples.real * _CI16_SCALE), -32768, 32767)
        im = np.clip(np.round(samples.imag * _CI16_SCALE), -32768, 32767)
        interleaved[0::2] = re.astype("<i2")
        interleaved[1::2] = im.astype("<i2")
        interleaved.tofile(path)
    else:
        raise ValueError(f"save() supports cf32/ci16, not {fmt!r}")
