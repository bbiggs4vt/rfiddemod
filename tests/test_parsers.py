import numpy as np
import pytest

from rfid_demod.parsers import em4100, hid_wiegand


class TestEM4100:
    def test_encode_parse_roundtrip(self):
        for id40 in (0x1234567890, 0x0000000000, 0xFFFFFFFFFF, 0xDEADBEEF42):
            frame = em4100.encode(id40)
            assert frame.size == 64
            fields = em4100.parse_frame(frame)
            assert fields is not None
            assert fields["id"] == f"{id40:010X}"
            assert fields["version"] == (id40 >> 32) & 0xFF
            assert fields["data"] == id40 & 0xFFFFFFFF

    def test_corruption_rejected(self):
        frame = em4100.encode(0x1234567890)
        for i in (0, 9, 30, 60, 63):
            bad = frame.copy()
            bad[i] ^= 1
            assert em4100.parse_frame(bad) is None

    def test_find_in_stream_with_junk(self):
        rng = np.random.default_rng(5)
        frame = em4100.encode(0xAB54A98CEB)
        stream = np.concatenate([
            rng.integers(0, 2, 37, dtype=np.uint8), frame, frame,
            rng.integers(0, 2, 11, dtype=np.uint8),
        ])
        hits = list(em4100.find_frames(stream))
        offsets = {h[0] for h in hits}
        assert {37, 37 + 64} <= offsets
        assert all(h[1]["id"] == "AB54A98CEB" for h in hits if h[0] in (37, 101))

    def test_find_inverted_stream(self):
        frame = em4100.encode(0x0123456789)
        hits = list(em4100.find_frames(frame ^ 1))
        assert hits and hits[0][1]["id"] == "0123456789"
        assert hits[0][2] is True  # flagged as inverted

    def test_validity_mask_blocks_scan(self):
        frame = em4100.encode(0x1234567890)
        valid = np.ones(64, bool)
        valid[20] = False
        assert list(em4100.find_frames(frame, valid)) == []


class TestHID:
    def test_h10301_pack_unpack(self):
        for fc, cn in ((118, 1603), (0, 0), (255, 65535), (42, 31337)):
            w = hid_wiegand.pack_h10301(fc, cn)
            value = hid_wiegand.wiegand_to_hid(w, 26)
            fields = hid_wiegand.unpack(value)
            assert fields["format"] == "H10301"
            assert fields["facility_code"] == fc
            assert fields["card_number"] == cn
            assert fields["parity_ok"] is True
            assert fields["bit_length"] == 26

    def test_parity_error_detected(self):
        w = hid_wiegand.pack_h10301(118, 1603) ^ (1 << 20)  # flip a data bit
        fields = hid_wiegand.unpack(hid_wiegand.wiegand_to_hid(w, 26))
        assert fields["parity_ok"] is False

    def test_unknown_lengths_rejected(self):
        assert hid_wiegand.unpack(0) is None
        assert hid_wiegand.unpack(1 << 10) is None   # 10-bit: not a HID format
        assert hid_wiegand.unpack(1 << 40) is None   # 40-bit: out of range
        assert hid_wiegand.unpack(hid_wiegand.wiegand_to_hid(0x1F2F3F4F0, 35)) is not None

    def test_frame_scan_roundtrip(self):
        value = hid_wiegand.wiegand_to_hid(hid_wiegand.pack_h10301(37, 9999), 26)
        frame = hid_wiegand.frame_fsk_bits(value)
        assert frame.size == 96
        stream = np.concatenate([frame, frame, frame])[13:-7]  # arbitrary crop
        hits = list(hid_wiegand.find_frames(stream))
        assert hits
        for _, fields, inverted, hid_value in hits:
            assert hid_value == value
            assert fields["facility_code"] == 37
            assert fields["card_number"] == 9999

    def test_frame_scan_inverted(self):
        value = hid_wiegand.wiegand_to_hid(hid_wiegand.pack_h10301(1, 2), 26)
        frame = hid_wiegand.frame_fsk_bits(value) ^ 1
        hits = list(hid_wiegand.find_frames(frame))
        assert hits and hits[0][2] is True and hits[0][3] == value

    def test_overwide_value_rejected(self):
        with pytest.raises(ValueError):
            hid_wiegand.wiegand_to_hid(1 << 27, 26)
        with pytest.raises(ValueError):
            hid_wiegand.frame_fsk_bits(1 << 44)
