"""Streaming decode tests: chunked feeding must match offline decoding."""

import json

import numpy as np
import pytest

from rfid_demod import cli
from rfid_demod.decoders import lf, uhf
from rfid_demod.io import iq
from rfid_demod.streaming import StreamDecoder

from test_lf_e2e import FS as LF_FS, em4100_capture
from test_uhf_e2e import EPC, FS as UHF_FS, build_inventory


def _feed_chunked(sd, x, chunk):
    frames = []
    for i in range(0, len(x), chunk):
        frames += sd.feed(x[i:i + chunk])
    frames += sd.flush()
    return frames


def test_lf_stream_matches_offline():
    x = em4100_capture(0x1234567890, repeats=10)
    offline = lf.decode(x, LF_FS)

    sd = StreamDecoder("lf", LF_FS, block_seconds=0.08, overlap_seconds=0.04)
    streamed = _feed_chunked(sd, x, chunk=10_000)

    assert len(streamed) == len(offline)
    assert all(f.fields["id"] == "1234567890" for f in streamed)
    # absolute timestamps line up with the offline decode
    off_ts = sorted(f.timestamp for f in offline)
    st_ts = sorted(f.timestamp for f in streamed)
    for a, b in zip(off_ts, st_ts):
        assert abs(a - b) < 1e-3


def test_stream_no_duplicates_across_boundaries():
    # tiny chunks + small blocks force every frame through several
    # overlapping decodes; each must still be emitted exactly once
    x = em4100_capture(0xAB54A98CEB, repeats=6)
    sd = StreamDecoder("lf", LF_FS, block_seconds=0.08, overlap_seconds=0.04)
    streamed = _feed_chunked(sd, x, chunk=3_333)
    offline = lf.decode(x, LF_FS)
    assert len(streamed) == len(offline)
    ts = [f.timestamp for f in streamed]
    assert ts == sorted(ts)


def test_uhf_stream_exchange_context():
    # a full inventory round placed mid-stream: the Query->reply context
    # must survive the block boundaries (overlap covers the exchange)
    lead = np.full(300_000, 1.0 + 0j, dtype=np.complex64)  # 150 ms of CW
    x = np.concatenate([lead, build_inventory(m_code=0)])
    sd = StreamDecoder("uhf", UHF_FS, block_seconds=0.1)
    streamed = _feed_chunked(sd, x, chunk=50_000)

    commands = [f.fields.get("command") for f in streamed if f.direction == "R->T"]
    assert commands == ["Query", "ACK", "QueryRep"]
    epcs = [f.fields.get("epc") for f in streamed if f.direction == "T->R"]
    assert EPC in epcs
    # timestamps are absolute in the stream, not block-relative
    assert min(f.timestamp for f in streamed) > 0.14


def test_iter_blocks_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    x = (rng.uniform(-0.5, 0.5, 100_000)
         + 1j * rng.uniform(-0.5, 0.5, 100_000)).astype(np.complex64)
    for fmt, atol in (("cf32", 0.0), ("ci16", 1.0 / 32768)):
        path = tmp_path / f"cap.{fmt}"
        iq.save(path, x, fmt=fmt)
        blocks = list(iq.iter_blocks(path, fmt, block_samples=7_777))
        assert sum(b.size for b in blocks) == x.size
        np.testing.assert_allclose(np.concatenate(blocks), x, atol=atol)


def test_iter_blocks_rejects_container_formats():
    with pytest.raises(ValueError, match="cf32/ci16"):
        next(iq.iter_blocks("cap.wav", "wav"))


def test_cli_stream(tmp_path):
    x = em4100_capture(0x1234567890, repeats=6)
    path = tmp_path / "em.cf32"
    iq.save(path, x)
    out = tmp_path / "frames.jsonl"

    rc = cli.main(["--band", "lf", "--stream", "--rate", "1e6",
                   "--in", str(path), "--out", str(out)])
    assert rc == 0
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records and all(r["fields"]["id"] == "1234567890" for r in records)
    assert len(records) == len(lf.decode(x, LF_FS))


def test_cli_stream_requires_rate_and_fmt(tmp_path):
    path = tmp_path / "cap.wav"
    path.write_bytes(b"\x00" * 64)
    assert cli.main(["--band", "lf", "--stream", "--in", str(path)]) == 2
    assert cli.main(["--band", "lf", "--stream", "--rate", "1e6",
                     "--in", str(path)]) == 2  # wav not streamable
