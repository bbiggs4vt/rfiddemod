import numpy as np

from rfid_demod.common import crc

CHECK_DATA = b"123456789"


def test_check_values():
    # RevEng catalogue check values for "123456789".
    assert crc.crc16_ccitt(CHECK_DATA) == 0x29B1   # CRC-16/CCITT-FALSE
    assert crc.crc16_gen2(CHECK_DATA) == 0xD64E    # CRC-16/GENIBUS
    assert crc.crc_a(CHECK_DATA) == 0xBF05         # CRC-16/ISO-IEC-14443-3-A
    assert crc.crc_b(CHECK_DATA) == 0x906E         # CRC-16/X-25


def test_append_and_check_roundtrip():
    payload = bytes([0x50, 0x00])  # 14443A HALT
    assert crc.check_crc_a(crc.append_crc_a(payload))
    assert crc.check_crc_b(crc.append_crc_b(payload))
    assert crc.check_crc16_gen2(crc.append_crc16_gen2(payload))


def test_corruption_detected():
    frame = bytearray(crc.append_crc_a(b"\x93\x20"))
    frame[0] ^= 0x01
    assert not crc.check_crc_a(bytes(frame))

    frame = bytearray(crc.append_crc16_gen2(b"\x30\x00"))
    frame[-1] ^= 0x80
    assert not crc.check_crc16_gen2(bytes(frame))


def test_known_crc_a_vector():
    # ANTICOLLISION cascade level 1 select: CRC-A(93 70 ...) — self-consistent
    # via the residue property instead of a table: appending the CRC and
    # recomputing over payload+crc yields 0 for this non-inverted CRC.
    payload = b"\x93\x70\x12\x34\x56\x78\x08"
    framed = crc.append_crc_a(payload)
    assert crc.crc_a(framed) == 0


def test_crc5_gen2_zero_remainder():
    rng = np.random.default_rng(1234)
    for _ in range(20):
        bits = list(rng.integers(0, 2, 17))  # Query body length sans CRC
        full = bits + crc.crc5_gen2_bits(bits)
        assert crc.check_crc5_gen2(full)
        corrupted = full.copy()
        corrupted[3] ^= 1
        assert not crc.check_crc5_gen2(corrupted)
