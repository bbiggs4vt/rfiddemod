"""End-to-end LF tests: synthetic IQ capture -> decoders.lf -> parsed frames."""

import json

import numpy as np
import pytest

from rfid_demod import cli, synth
from rfid_demod.decoders import lf
from rfid_demod.encodings import fsk, manchester
from rfid_demod.io import iq
from rfid_demod.parsers import em4100, hid_wiegand

FS = 1_000_000.0
FC = 125_000.0


def em4100_capture(id40, rf=32, repeats=3, snr_db=25.0, mod_index=0.6,
                   freq_offset=500.0, seed=0):
    bits = np.tile(em4100.encode(id40), repeats)
    half_bit = int(rf // 2 * FS / FC)
    levels = synth.upsample(manchester.encode(bits), half_bit)
    x = synth.ask_iq(levels, mod_index=mod_index,
                     freq_offset=freq_offset, sample_rate=FS)
    return synth.awgn(x, snr_db, seed=seed)


def hid_capture(facility, card, repeats=3, snr_db=25.0, seed=0):
    value = hid_wiegand.wiegand_to_hid(hid_wiegand.pack_h10301(facility, card), 26)
    bits = np.tile(hid_wiegand.frame_fsk_bits(value), repeats)
    spb = int(50 * FS / FC)  # 50 carrier cycles per bit
    levels = fsk.modulate(bits, spb, period0=8 * FS / FC, period1=10 * FS / FC)
    x = synth.ask_iq(levels, mod_index=0.5, sample_rate=FS)
    return synth.awgn(x, snr_db, seed=seed)


@pytest.mark.parametrize("rf", [32, 64])
def test_em4100_end_to_end(rf):
    x = em4100_capture(0x1234567890, rf=rf)
    frames = lf.decode(x, FS)
    em = [f for f in frames if f.fields.get("protocol") == "em4100"]
    assert len(em) >= 2  # repeats detected
    for f in em:
        assert f.fields["id"] == "1234567890"
        assert f.fields["rf_clock"] == rf
        assert f.crc_ok is True
        assert len(f.bits) == 64
    # timestamps advance by one frame period per repeat
    dt = em[1].timestamp - em[0].timestamp
    assert abs(dt - 64 * rf / FC) < 2 * rf / FC


def test_em4100_inverted_polarity():
    # modulate the complemented chip stream (envelope polarity flipped)
    bits = np.tile(em4100.encode(0xDEADBEEF42), 3)
    levels = 1 - synth.upsample(manchester.encode(bits), int(16 * FS / FC))
    x = synth.awgn(synth.ask_iq(levels, mod_index=0.6, sample_rate=FS), 25.0, seed=1)
    frames = lf.decode(x, FS)
    em = [f for f in frames if f.fields.get("protocol") == "em4100"]
    assert em and all(f.fields["id"] == "DEADBEEF42" for f in em)
    assert all(f.fields["inverted"] for f in em)


def test_em4100_not_found_in_noise():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)) * 0.1 + 1.0
    assert lf.decode(x.astype(np.complex64), FS) == []


def test_hid_end_to_end():
    x = hid_capture(118, 1603)
    frames = lf.decode(x, FS)
    hid = [f for f in frames if f.fields.get("protocol") == "hid_prox"]
    assert len(hid) >= 2
    for f in hid:
        assert f.fields["format"] == "H10301"
        assert f.fields["facility_code"] == 118
        assert f.fields["card_number"] == 1603
        assert f.crc_ok is True
        assert len(f.bits) == 26


def test_hid_low_snr():
    x = hid_capture(42, 31337, snr_db=15.0, seed=4)
    hid = [f for f in lf.decode(x, FS) if f.fields.get("protocol") == "hid_prox"]
    assert hid and hid[0].fields["card_number"] == 31337


def test_cli_end_to_end(tmp_path):
    path = tmp_path / "em.cf32"
    iq.save(path, em4100_capture(0xAB54A98CEB))
    out = tmp_path / "frames.jsonl"

    rc = cli.main(["--band", "lf", "--rate", "1e6",
                   "--in", str(path), "--out", str(out)])
    assert rc == 0

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records
    ids = {r["fields"]["id"] for r in records if r["fields"].get("protocol") == "em4100"}
    assert ids == {"AB54A98CEB"}
    for r in records:
        assert r["band"] == "lf" and r["direction"] == "T->R"
