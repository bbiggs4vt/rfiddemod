"""End-to-end UHF Gen2 tests: synthetic inventory round -> decoders.uhf."""

import json

import numpy as np
import pytest

from rfid_demod import cli, synth
from rfid_demod.decoders import uhf
from rfid_demod.decoders.uhf import tag_path
from rfid_demod.encodings import pie
from rfid_demod.io import iq
from rfid_demod.parsers import gen2_frames as gen2

FS = 2_000_000.0
TARI_US = 20.0
SPT = int(TARI_US * 1e-6 * FS)           # 40 samples per Tari
DELIM = int(12.5e-6 * FS)
PW = 0.4                                  # PW = 0.4 Tari (spec: 0.265-0.525)
TRCAL_TARIS = 8.0                         # TRcal 160 us; RTcal 60 us -> ratio 2.67
BLF = 8.0 / (TRCAL_TARIS * TARI_US * 1e-6)   # 50 kHz at DR=8
SPU = int(round(FS / (2 * BLF)))          # samples per half-unit (20)

EPC = "E280117020001234ABCD5678"
RN16 = 0x2E5B


def _bits_of(value, width):
    return np.array([(value >> i) & 1 for i in range(width - 1, -1, -1)], np.uint8)


class ExchangeBuilder:
    """Builds reader levels + tag overlay with a running sample cursor."""

    def __init__(self):
        self.reader = []
        self.tag = []

    def _add(self, reader_levels, tag_levels=None):
        reader_levels = np.asarray(reader_levels, np.float64)
        self.reader.append(reader_levels)
        if tag_levels is None:
            tag_levels = np.zeros(reader_levels.size)
        self.tag.append(np.asarray(tag_levels, np.float64))

    def cw(self, n):
        self._add(np.ones(n))

    def command(self, bits, preamble=False):
        head = pie.preamble(SPT, DELIM, TRCAL_TARIS, pw_ratio=PW) if preamble \
            else pie.frame_sync(SPT, DELIM, pw_ratio=PW)
        self._add(head)
        self._add(pie.encode(bits, SPT, pw_ratio=PW))

    def reply(self, unit_levels, spu=SPU):
        levels = synth.upsample(unit_levels, spu)
        self._add(np.ones(levels.size), levels)

    def build(self, tag_amp=0.06, tag_phase=1.1, snr_db=30.0, seed=0):
        reader = np.concatenate(self.reader)
        tag = np.concatenate(self.tag)
        x = synth.ask_iq(reader, mod_index=0.9).astype(np.complex128)
        x += tag_amp * tag * np.exp(1j * tag_phase)
        return synth.awgn(x.astype(np.complex64), snr_db, seed=seed)


def build_inventory(m_code=0, trext=0, tag_amp=0.06, snr_db=30.0, seed=0,
                    tag_present=True):
    """Query -> RN16 -> ACK -> PC+EPC+CRC -> QueryRep (no reply)."""
    m = gen2.M_CODES[m_code]
    b = ExchangeBuilder()
    b.cw(400)
    b.command(gen2.build_query(dr=0, m=m_code, trext=trext, q=4), preamble=True)
    b.cw(200)
    if tag_present:
        rn16_bits = _bits_of(RN16, 16)
        epc_bits = gen2.build_tag_epc_reply(EPC)
        if m == 1:
            b.reply(tag_path.fm0_reply_chips(rn16_bits, trext))
        else:
            b.reply(tag_path.miller_reply_halfcycles(rn16_bits, m, trext))
        b.cw(200)
        b.command(gen2.build_ack(RN16))
        b.cw(200)
        if m == 1:
            b.reply(tag_path.fm0_reply_chips(epc_bits, trext))
        else:
            b.reply(tag_path.miller_reply_halfcycles(epc_bits, m, trext))
        b.cw(200)
    else:
        b.cw(600)
    b.command(gen2.build_query_rep(0))
    b.cw(600)
    return b.build(tag_amp=tag_amp, snr_db=snr_db, seed=seed)


def _split(frames):
    reader = [f for f in frames if f.direction == "R->T"]
    tag = [f for f in frames if f.direction == "T->R"]
    return reader, tag


@pytest.mark.parametrize("m_code,encoding", [(0, "fm0"), (1, "miller_m2"),
                                             (2, "miller_m4"), (3, "miller_m8")])
def test_full_inventory(m_code, encoding):
    x = build_inventory(m_code=m_code)
    reader, tag = _split(uhf.decode(x, FS))

    commands = [f.fields["command"] for f in reader]
    assert commands == ["Query", "ACK", "QueryRep"]

    query = reader[0].fields
    assert query["crc_ok"] is True
    assert query["m"] == gen2.M_CODES[m_code]
    assert abs(query["blf_hz"] - BLF) < 0.03 * BLF
    assert abs(query["tari_us"] - TARI_US) < 1.0

    ack = reader[1].fields
    assert ack["rn16"] == f"{RN16:04X}"

    assert len(tag) == 2
    rn16, epc = tag
    assert rn16.fields["type"] == "RN16"
    assert rn16.fields["rn16"] == f"{RN16:04X}"
    assert rn16.fields["encoding"] == encoding

    assert epc.fields["type"] == "EPC"
    assert epc.fields["epc"] == EPC
    assert epc.fields["pc"] == "3000"
    assert epc.crc_ok is True

    # tag replies must land inside the exchange, after their commands
    assert reader[0].timestamp < rn16.timestamp < reader[1].timestamp
    assert reader[1].timestamp < epc.timestamp < reader[2].timestamp


def test_trext_pilot_tone():
    x = build_inventory(m_code=0, trext=1)
    _, tag = _split(uhf.decode(x, FS))
    assert [f.fields["type"] for f in tag] == ["RN16", "EPC"]
    assert tag[1].fields["epc"] == EPC


def test_no_tag_no_false_replies():
    x = build_inventory(tag_present=False)
    reader, tag = _split(uhf.decode(x, FS))
    assert [f.fields["command"] for f in reader] == ["Query", "QueryRep"]
    assert tag == []


def test_weak_tag():
    x = build_inventory(tag_amp=0.02, snr_db=35.0, seed=3)  # ~34 dB below carrier
    _, tag = _split(uhf.decode(x, FS))
    assert any(f.fields.get("epc") == EPC and f.crc_ok for f in tag)


def test_cli_end_to_end(tmp_path):
    path = tmp_path / "gen2.cf32"
    iq.save(path, build_inventory())
    out = tmp_path / "frames.jsonl"

    rc = cli.main(["--band", "uhf", "--rate", "2e6",
                   "--in", str(path), "--out", str(out)])
    assert rc == 0

    records = [json.loads(line) for line in out.read_text().splitlines()]
    commands = [r["fields"].get("command") for r in records if r["direction"] == "R->T"]
    assert commands == ["Query", "ACK", "QueryRep"]
    epcs = [r["fields"].get("epc") for r in records if r["direction"] == "T->R"]
    assert EPC in epcs
