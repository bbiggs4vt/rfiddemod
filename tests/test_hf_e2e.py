"""End-to-end HF tests: synthetic 14443A / 14443B / 15693 exchanges."""

import json

import numpy as np

from rfid_demod import cli, synth
from rfid_demod.common import crc
from rfid_demod.decoders import hf
from rfid_demod.decoders.hf import iso14443a, iso14443b, iso15693
from rfid_demod.io import iq
from rfid_demod.parsers import iso14443a_frames as a_frames

FS = 13.56e6
UID = bytes([0x12, 0x34, 0x56, 0x78])
BCC = UID[0] ^ UID[1] ^ UID[2] ^ UID[3]


class HFBuilder:
    """Reader levels + complex tag overlay with a running cursor."""

    def __init__(self, mod_index=1.0):
        self.mod_index = mod_index
        self.reader = []
        self.tag = []

    def _add(self, reader_levels, tag_wave=None):
        reader_levels = np.asarray(reader_levels, np.float64)
        self.reader.append(reader_levels)
        if tag_wave is None:
            tag_wave = np.zeros(reader_levels.size, dtype=np.complex128)
        self.tag.append(np.asarray(tag_wave, np.complex128))

    def cw(self, us):
        self._add(np.ones(int(us * 1e-6 * FS)))

    def command(self, levels):
        self._add(levels)

    def reply(self, wave):
        self._add(np.ones(len(wave)), np.asarray(wave, np.complex128))

    def build(self, tag_amp=0.05, tag_phase=0.9, snr_db=30.0, seed=0):
        reader = np.concatenate(self.reader)
        tag = np.concatenate(self.tag)
        x = synth.ask_iq(reader, mod_index=self.mod_index).astype(np.complex128)
        x += tag_amp * tag * np.exp(1j * tag_phase)
        return synth.awgn(x.astype(np.complex64), snr_db, seed=seed)


def build_14443a(snr_db=30.0, seed=0, tag_amp=0.05):
    """REQA -> ATQA -> Anticollision -> UID -> Select -> SAK."""
    b = HFBuilder(mod_index=1.0)
    b.cw(50)
    b.command(iso14443a.reader_frame_levels(
        a_frames.short_frame_bits(a_frames.REQA), FS))
    b.cw(90)
    b.reply(iso14443a.tag_wave(a_frames.standard_frame_bits(b"\x04\x00"), FS))
    b.cw(90)
    b.command(iso14443a.reader_frame_levels(
        a_frames.standard_frame_bits(bytes([0x93, 0x20])), FS))
    b.cw(90)
    b.reply(iso14443a.tag_wave(
        a_frames.standard_frame_bits(UID + bytes([BCC])), FS))
    b.cw(90)
    b.command(iso14443a.reader_frame_levels(a_frames.standard_frame_bits(
        crc.append_crc_a(bytes([0x93, 0x70]) + UID + bytes([BCC]))), FS))
    b.cw(90)
    b.reply(iso14443a.tag_wave(
        a_frames.standard_frame_bits(crc.append_crc_a(b"\x08")), FS))
    b.cw(150)
    return b.build(tag_amp=tag_amp, snr_db=snr_db, seed=seed)


def build_14443b(snr_db=30.0, seed=0):
    """REQB -> ATQB."""
    etu = int(round(128 * FS / 13.56e6))
    b = HFBuilder(mod_index=0.1)
    b.cw(80)
    reqb = crc.append_crc_b(bytes([0x05, 0x00, 0x00]))
    b.command(np.repeat(iso14443b.char_stream_levels(reqb), etu))
    b.cw(120)
    atqb = crc.append_crc_b(bytes([0x50, 0xAA, 0xBB, 0xCC, 0xDD,
                                   1, 2, 3, 4, 0x00, 0x81, 0x71]))
    b.reply(iso14443b.tag_wave(atqb, FS))
    b.cw(200)
    return b.build(snr_db=snr_db, seed=seed)


def build_15693(snr_db=30.0, seed=0):
    """Inventory -> InventoryResponse."""
    b = HFBuilder(mod_index=1.0)
    b.cw(120)
    b.command(iso15693.reader_frame_levels(
        crc.append_crc_b(bytes([0x26, 0x01, 0x00])), FS))
    b.cw(320)
    uid = bytes.fromhex("E004010203040506")
    b.reply(iso15693.tag_wave(
        crc.append_crc_b(bytes([0x00, 0x00]) + uid[::-1]), FS))
    b.cw(300)
    return b.build(snr_db=snr_db, seed=seed)


def _protocols(frames):
    return {f.fields["protocol"] for f in frames}


def test_14443a_select_sequence():
    frames = hf.decode(build_14443a(), FS)
    assert _protocols(frames) == {"iso14443a"}
    reader = [f for f in frames if f.direction == "R->T"]
    tag = [f for f in frames if f.direction == "T->R"]

    assert [f.fields["command"] for f in reader] == \
        ["REQA", "Anticollision", "Select"]
    sel = reader[2].fields
    assert sel["uid"] == "12345678" and sel["crc_ok"] and sel["bcc_ok"]

    assert [f.fields["type"] for f in tag] == ["ATQA", "UID", "SAK"]
    assert tag[0].fields["atqa"] == "0004"
    assert tag[1].fields["uid"] == "12345678" and tag[1].fields["bcc_ok"]
    assert tag[2].fields["sak"] == "08" and tag[2].crc_ok is True
    assert all(f.fields["parity_ok"] for f in tag)

    # replies interleave with their commands
    ts = [f.timestamp for f in frames]
    assert ts == sorted(ts)
    assert reader[0].timestamp < tag[0].timestamp < reader[1].timestamp


def test_14443a_low_snr():
    # 20 dB channel SNR with a stronger (close-coupled) tag: load
    # modulation at HF proximity range is well above the UHF-style level.
    frames = hf.decode(build_14443a(snr_db=20.0, seed=5, tag_amp=0.15), FS)
    tag = [f for f in frames if f.direction == "T->R"]
    assert any(f.fields.get("uid") == "12345678" for f in tag)


def test_14443b_reqb_atqb():
    frames = hf.decode(build_14443b(), FS)
    assert _protocols(frames) == {"iso14443b"}
    reader = [f for f in frames if f.direction == "R->T"]
    tag = [f for f in frames if f.direction == "T->R"]

    assert len(reader) == 1 and reader[0].fields["command"] == "REQB"
    assert reader[0].crc_ok is True
    assert len(tag) == 1
    assert tag[0].fields["type"] == "ATQB"
    assert tag[0].fields["pupi"] == "AABBCCDD"
    assert tag[0].crc_ok is True


def test_15693_inventory():
    frames = hf.decode(build_15693(), FS)
    assert _protocols(frames) == {"iso15693"}
    reader = [f for f in frames if f.direction == "R->T"]
    tag = [f for f in frames if f.direction == "T->R"]

    assert len(reader) == 1
    assert reader[0].fields["command"] == "Inventory"
    assert reader[0].crc_ok is True
    assert len(tag) == 1
    assert tag[0].fields["type"] == "InventoryResponse"
    assert tag[0].fields["uid"] == "E004010203040506"
    assert tag[0].crc_ok is True


def test_noise_only_no_frames():
    rng = np.random.default_rng(9)
    n = 400_000
    x = (1.0 + 0.02 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))
    assert hf.decode(x.astype(np.complex64), FS) == []


def test_cli_end_to_end(tmp_path):
    path = tmp_path / "hf.cf32"
    iq.save(path, build_14443a())
    out = tmp_path / "frames.jsonl"

    rc = cli.main(["--band", "hf", "--rate", str(FS),
                   "--in", str(path), "--out", str(out)])
    assert rc == 0

    records = [json.loads(line) for line in out.read_text().splitlines()]
    uids = [r["fields"].get("uid") for r in records if r["direction"] == "T->R"]
    assert "12345678" in uids
    assert all(r["band"] == "hf" for r in records)
