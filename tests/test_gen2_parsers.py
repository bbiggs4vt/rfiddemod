import numpy as np

from rfid_demod.common import crc
from rfid_demod.parsers import gen2_frames as gen2


def test_crc16_bits_matches_byte_crc():
    data = b"\x30\x00\x12\x34"
    bits = [(byte >> i) & 1 for byte in data for i in range(7, -1, -1)]
    assert crc.crc16_bits(bits) == crc.crc16(data, 0x1021, 0xFFFF)
    assert crc.crc16_gen2_bits(bits) == crc.crc16_gen2(data)


def test_query_roundtrip():
    bits = gen2.build_query(dr=1, m=2, trext=1, sel=2, session=3, target=1, q=7)
    assert bits.size == 22
    info = gen2.parse_reader(bits)
    assert info["command"] == "Query"
    assert info["dr"] == 1 and info["dr_ratio"] == 64.0 / 3.0
    assert info["m"] == 4          # M code 2 -> Miller M=4
    assert info["trext"] == 1
    assert info["sel"] == 2 and info["session"] == 3 and info["target"] == 1
    assert info["q"] == 7
    assert info["crc_ok"] is True


def test_query_crc_error():
    bits = gen2.build_query(q=5)
    bits[10] ^= 1
    info = gen2.parse_reader(bits)
    # header still says Query, but the CRC flags the corruption
    assert info["command"] == "Query" and info["crc_ok"] is False


def test_simple_commands():
    assert gen2.parse_reader(gen2.build_query_rep(2)) == {
        "command": "QueryRep", "session": 2}
    assert gen2.parse_reader(gen2.build_query_adjust(1, 6)) == {
        "command": "QueryAdjust", "session": 1, "updn": 6}
    assert gen2.parse_reader(gen2.build_nak()) == {"command": "NAK"}

    ack = gen2.parse_reader(gen2.build_ack(0xBEEF))
    assert ack == {"command": "ACK", "rn16": "BEEF"}

    req = gen2.parse_reader(gen2.build_req_rn(0x1234))
    assert req["command"] == "Req_RN" and req["rn16"] == "1234"
    assert req["crc_ok"] is True


def test_select_roundtrip():
    mask = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1]
    bits = gen2.build_select(target=4, action=2, membank=1,
                             pointer=0x120, mask_bits=mask, truncate=0)
    info = gen2.parse_reader(bits)
    assert info["command"] == "Select"
    assert info["target"] == 4 and info["action"] == 2 and info["membank"] == 1
    assert info["pointer"] == 0x120        # EBV-coded (two groups)
    assert info["mask_length"] == len(mask)
    assert info["mask"] == "B2F"
    assert info["crc_ok"] is True


def test_tag_epc_reply_roundtrip():
    epc = "E280117020001234ABCD5678"  # 96-bit EPC
    bits = gen2.build_tag_epc_reply(epc)
    assert bits.size == 16 + 96 + 16
    info = gen2.parse_tag(bits, expected="epc")
    assert info["type"] == "EPC"
    assert info["epc"] == epc
    assert info["pc"] == "3000"            # 6 words in the length field
    assert info["crc_ok"] is True
    assert info["bits_used"] == 128

    corrupted = bits.copy()
    corrupted[40] ^= 1
    assert gen2.parse_tag(corrupted, expected="epc")["crc_ok"] is False


def test_tag_reply_with_trailing_junk():
    rng = np.random.default_rng(0)
    bits = np.concatenate([gen2.build_tag_epc_reply("DEADBEEF00112233445566AA"),
                           [1], rng.integers(0, 2, 25, dtype=np.uint8)])
    info = gen2.parse_tag(bits, expected="epc")
    assert info["epc"] == "DEADBEEF00112233445566AA" and info["crc_ok"] is True


def test_tag_rn16_and_handle():
    rn = gen2.parse_tag(np.array([0, 1] * 8 + [1, 0, 1], np.uint8), expected="rn16")
    assert rn["type"] == "RN16" and rn["rn16"] == "5555" and rn["bits_used"] == 16

    handle_bits = [(0xCAFE >> i) & 1 for i in range(15, -1, -1)]
    handle_bits += [(crc.crc16_gen2_bits(handle_bits) >> i) & 1 for i in range(15, -1, -1)]
    info = gen2.parse_tag(np.array(handle_bits + [1, 0], np.uint8), expected="handle")
    assert info["type"] == "Handle" and info["handle"] == "CAFE"
    assert info["crc_ok"] is True

    assert gen2.parse_tag(np.array([1, 0, 1], np.uint8)) is None
