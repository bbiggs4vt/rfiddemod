import numpy as np

from rfid_demod.common import crc
from rfid_demod.parsers import (iso14443a_frames as a_frames,
                                iso14443b_frames as b_frames,
                                iso15693_frames as v_frames)


class TestTypeAFrames:
    def test_short_frame_roundtrip(self):
        bits = a_frames.short_frame_bits(a_frames.REQA)
        assert bits.size == 7
        frame = a_frames.parse_frame_bits(bits)
        assert frame == {"kind": "short", "value": 0x26}
        assert a_frames.parse_reader(frame)["command"] == "REQA"
        wupa = a_frames.parse_frame_bits(a_frames.short_frame_bits(a_frames.WUPA))
        assert a_frames.parse_reader(wupa)["command"] == "WUPA"

    def test_standard_frame_roundtrip(self):
        data = bytes([0x93, 0x70, 0x12, 0x34, 0x56, 0x78])
        bits = a_frames.standard_frame_bits(data)
        assert bits.size == 9 * len(data)
        frame = a_frames.parse_frame_bits(bits)
        assert frame["kind"] == "standard"
        assert frame["bytes"] == data
        assert frame["parity_ok"] is True

    def test_parity_error_detected(self):
        bits = a_frames.standard_frame_bits(b"\x04\x00")
        bits[3] ^= 1  # flip a data bit; parity no longer matches
        assert a_frames.parse_frame_bits(bits)["parity_ok"] is False

    def test_parse_select_and_replies(self):
        uid = bytes([0x12, 0x34, 0x56, 0x78])
        bcc = uid[0] ^ uid[1] ^ uid[2] ^ uid[3]
        select = crc.append_crc_a(bytes([0x93, 0x70]) + uid + bytes([bcc]))
        info = a_frames.parse_reader(a_frames.parse_frame_bits(
            a_frames.standard_frame_bits(select)))
        assert info["command"] == "Select"
        assert info["uid"] == "12345678"
        assert info["bcc_ok"] is True and info["crc_ok"] is True

        atqa = a_frames.parse_tag(a_frames.parse_frame_bits(
            a_frames.standard_frame_bits(b"\x04\x00")), expected="atqa")
        assert atqa["type"] == "ATQA" and atqa["atqa"] == "0004"

        uid_frame = a_frames.parse_tag(a_frames.parse_frame_bits(
            a_frames.standard_frame_bits(uid + bytes([bcc]))), expected="uid")
        assert uid_frame["type"] == "UID"
        assert uid_frame["uid"] == "12345678" and uid_frame["bcc_ok"] is True

        sak = a_frames.parse_tag(a_frames.parse_frame_bits(
            a_frames.standard_frame_bits(crc.append_crc_a(b"\x08"))), expected="sak")
        assert sak["type"] == "SAK" and sak["sak"] == "08" and sak["crc_ok"] is True


class TestTypeBFrames:
    def test_reqb_and_atqb(self):
        reqb = crc.append_crc_b(bytes([0x05, 0x00, 0x00]))
        info = b_frames.parse_reader(reqb)
        assert info["command"] == "REQB" and info["slots"] == 1
        assert info["crc_ok"] is True

        wupb = crc.append_crc_b(bytes([0x05, 0x00, 0x08]))
        assert b_frames.parse_reader(wupb)["command"] == "WUPB"

        atqb = crc.append_crc_b(bytes([0x50, 0xAA, 0xBB, 0xCC, 0xDD,
                                       1, 2, 3, 4, 0x00, 0x81, 0x71]))
        info = b_frames.parse_tag(atqb)
        assert info["type"] == "ATQB" and info["pupi"] == "AABBCCDD"
        assert info["crc_ok"] is True


class TestIso15693Frames:
    def test_inventory_roundtrip(self):
        req = crc.append_crc_b(bytes([0x26, 0x01, 0x00]))
        info = v_frames.parse_reader(req)
        assert info["command"] == "Inventory" and info["crc_ok"] is True

        uid = bytes.fromhex("E004010203040506")
        resp = crc.append_crc_b(bytes([0x00, 0x00]) + uid[::-1])
        info = v_frames.parse_tag(resp, expected="inventory")
        assert info["type"] == "InventoryResponse"
        assert info["uid"] == "E004010203040506"
        assert info["crc_ok"] is True

    def test_crc_error_flagged(self):
        req = bytearray(crc.append_crc_b(bytes([0x26, 0x01, 0x00])))
        req[1] ^= 0x10
        assert v_frames.parse_reader(bytes(req))["crc_ok"] is False
