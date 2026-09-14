"""MIDAS BLUE reader tests (synthetic HCBs built to the X-Midas layout)."""

import json
import struct

import numpy as np
import pytest

from rfid_demod import cli
from rfid_demod.io import blue, iq

from test_lf_e2e import em4100_capture


def _keyword_record(tag: str, ktype: str, payload: bytes, order="<") -> bytes:
    body = 8 + len(payload) + len(tag)
    lkey = (body + 7) // 8 * 8
    lext = lkey - len(payload)
    rec = struct.pack(order + "ihbc", lkey, lext, len(tag), ktype.encode())
    rec += payload + tag.encode()
    return rec + b"\x00" * (lkey - body)


def write_blue(path, samples, sample_rate, fmt="CF", head_rep=b"EEEI",
               data_rep=b"EEEI", keywords=(), ftype=1000, detached=0):
    horder = "<" if head_rep == b"EEEI" else ">"
    dorder = "<" if data_rep == b"EEEI" else ">"

    ext = b"".join(_keyword_record(t, k, p, horder) for t, k, p in keywords)

    element = fmt[1]
    if element == "F":
        raw = np.empty(2 * len(samples), dtype=dorder + "f4")
        raw[0::2], raw[1::2] = samples.real, samples.imag
    elif element == "I":
        raw = np.empty(2 * len(samples), dtype=dorder + "i2")
        raw[0::2] = np.round(samples.real * 32768).clip(-32768, 32767)
        raw[1::2] = np.round(samples.imag * 32768).clip(-32768, 32767)
    else:
        raise ValueError(fmt)
    data = raw.tobytes()

    data_start = 512.0
    ext_start_blocks = 0
    ext_size = 0
    if ext:
        # extended header sits after the data, on a 512-byte boundary
        ext_offset = int(np.ceil((512 + len(data)) / 512.0)) * 512
        ext_start_blocks = ext_offset // 512
        ext_size = len(ext)

    hcb = bytearray(512)
    hcb[0:4] = b"BLUE"
    hcb[4:8] = head_rep
    hcb[8:12] = data_rep
    struct.pack_into(horder + "i", hcb, 12, detached)
    struct.pack_into(horder + "ii", hcb, 24, ext_start_blocks, ext_size)
    struct.pack_into(horder + "dd", hcb, 32, data_start, float(len(data)))
    struct.pack_into(horder + "i", hcb, 48, ftype)
    hcb[52:54] = fmt.encode()
    struct.pack_into(horder + "ddi", hcb, 256, 0.0, 1.0 / sample_rate, 1)

    with open(path, "wb") as fh:
        fh.write(hcb)
        fh.write(data)
        if ext:
            fh.seek(ext_start_blocks * 512)
            fh.write(ext)


def _tone(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-0.9, 0.9, n)
            + 1j * rng.uniform(-0.9, 0.9, n)).astype(np.complex64)


def test_cf_roundtrip_with_keywords(tmp_path):
    x = _tone()
    path = tmp_path / "capture.tmp"
    write_blue(path, x, 2e6, fmt="CF",
               keywords=[("RF", "D", struct.pack("<d", 915e6)),
                         ("COMMENT", "A", b"lab capture")])
    cap = iq.load(path)
    assert cap.sample_rate == 2e6
    assert cap.center_freq == 915e6
    assert cap.metadata["COMMENT"] == "lab capture"
    np.testing.assert_allclose(cap.samples, x, atol=1e-7)


def test_ci_roundtrip(tmp_path):
    x = _tone(seed=1)
    path = tmp_path / "capture.blue"
    write_blue(path, x, 1e6, fmt="CI")
    cap = iq.load(path)
    assert cap.sample_rate == 1e6
    assert cap.center_freq is None
    np.testing.assert_allclose(cap.samples, x, atol=1.0 / 32768)


def test_big_endian_header_and_data(tmp_path):
    x = _tone(seed=2)
    path = tmp_path / "capture.tmp"
    write_blue(path, x, 4e6, fmt="CF", head_rep=b"IEEE", data_rep=b"IEEE")
    cap = iq.load(path)
    assert cap.sample_rate == 4e6
    np.testing.assert_allclose(cap.samples, x, atol=1e-7)


def test_magic_sniff_without_suffix(tmp_path):
    x = _tone(seed=3)
    path = tmp_path / "capture_no_suffix"
    write_blue(path, x, 1e6, fmt="CF")
    cap = iq.load(path)   # no suffix: detected via the BLUE magic
    assert cap.sample_rate == 1e6
    np.testing.assert_allclose(cap.samples, x, atol=1e-7)


def test_rejections(tmp_path):
    x = _tone(100)
    p1 = tmp_path / "t2000.tmp"
    write_blue(p1, x, 1e6, fmt="CF", ftype=2000)
    with pytest.raises(ValueError, match="1-D"):
        iq.load(p1)

    p2 = tmp_path / "detached.tmp"
    write_blue(p2, x, 1e6, fmt="CF", detached=1)
    with pytest.raises(ValueError, match="detached"):
        iq.load(p2)

    p3 = tmp_path / "scalar.tmp"
    write_blue(p3, x, 1e6, fmt="SF")
    with pytest.raises(ValueError, match="complex"):
        iq.load(p3)

    p4 = tmp_path / "notblue.tmp"
    p4.write_bytes(b"JUNK" + b"\x00" * 600)
    with pytest.raises(ValueError, match="magic"):
        iq.load(p4)


def test_cli_decodes_blue_lf_capture(tmp_path):
    path = tmp_path / "em4100.tmp"
    write_blue(path, em4100_capture(0x1234567890), 1e6, fmt="CI",
               keywords=[("RF", "D", struct.pack("<d", 125e3))])
    out = tmp_path / "frames.jsonl"

    # no --rate: it comes from the BLUE header's xdelta
    rc = cli.main(["--band", "lf", "--in", str(path), "--out", str(out)])
    assert rc == 0
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records and all(r["fields"]["id"] == "1234567890" for r in records)


def test_low_rate_capture_refused(tmp_path):
    # a narrowband survey snip cannot contain HF signaling; the decoder
    # must refuse rather than emit garbage frames
    path = tmp_path / "narrow.tmp"
    write_blue(path, _tone(2000, seed=4), 20_000, fmt="CF")
    rc = cli.main(["--band", "hf", "--in", str(path), "--out", "/dev/null"])
    assert rc == 3
