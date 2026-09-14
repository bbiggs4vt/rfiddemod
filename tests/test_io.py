import json

import numpy as np

from rfid_demod.io import Frame, JsonlWriter, iq


def _tone(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-0.9, 0.9, n) + 1j * rng.uniform(-0.9, 0.9, n)).astype(np.complex64)


def test_cf32_roundtrip(tmp_path):
    x = _tone()
    path = tmp_path / "capture.cf32"
    iq.save(path, x)
    cap = iq.load(path)
    assert cap.samples.dtype == np.complex64
    np.testing.assert_array_equal(cap.samples, x)
    assert cap.sample_rate is None  # raw file carries no metadata


def test_ci16_roundtrip(tmp_path):
    x = _tone()
    path = tmp_path / "capture.ci16"
    iq.save(path, x)
    cap = iq.load(path)
    assert cap.samples.dtype == np.complex64
    np.testing.assert_allclose(cap.samples, x, atol=1.0 / 32768)


def test_format_inference_error(tmp_path):
    path = tmp_path / "capture.bin"
    path.write_bytes(b"\x00" * 8)
    try:
        iq.load(path)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "fmt" in str(exc)
    cap = iq.load(path, fmt="cf32")
    assert cap.samples.size == 1


def test_wav_roundtrip(tmp_path):
    from scipy.io import wavfile

    x = _tone(500)
    stereo = np.stack([x.real, x.imag], axis=1)
    ints = np.round(stereo * 32767).astype(np.int16)
    path = tmp_path / "capture.wav"
    wavfile.write(path, 250_000, ints)

    cap = iq.load(path)
    assert cap.sample_rate == 250_000
    np.testing.assert_allclose(cap.samples, x, atol=2.0 / 32768)


def test_sigmf_pair(tmp_path):
    x = _tone(256)
    (tmp_path / "cap.sigmf-data").write_bytes(x.astype("<c8").tobytes())
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": 2e6},
        "captures": [{"core:sample_start": 0, "core:frequency": 915e6}],
    }
    (tmp_path / "cap.sigmf-meta").write_text(json.dumps(meta))

    for entry in ("cap.sigmf-meta", "cap.sigmf-data"):
        cap = iq.load(tmp_path / entry)
        assert cap.sample_rate == 2e6
        assert cap.center_freq == 915e6
        np.testing.assert_array_equal(cap.samples, x)


def test_jsonl_writer(tmp_path):
    path = tmp_path / "frames.jsonl"
    frames = [
        Frame(timestamp=0.001, band="lf", direction="T->R", bits="1" * 9 + "0" * 5,
              fields={"id": "0x1234567890"}, crc_ok=None),
        Frame(timestamp=0.5, band="uhf", direction="T->R", bits="0110",
              fields={"epc": "e2000017"}, crc_ok=True),
    ]
    with JsonlWriter(path) as w:
        for f in frames:
            w.write(f)
        assert w.count == 2

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[1])
    assert rec["band"] == "uhf"
    assert rec["crc_ok"] is True
    assert rec["fields"]["epc"] == "e2000017"
