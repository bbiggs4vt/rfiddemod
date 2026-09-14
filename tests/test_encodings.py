import numpy as np
import pytest

from rfid_demod.encodings import biphase, fm0, fsk, manchester, miller, pie, psk
from rfid_demod.synth import random_bits


BITS = [random_bits(64, seed=s) for s in range(3)] + [
    np.zeros(8, np.uint8),
    np.ones(8, np.uint8),
    np.array([1, 0, 1, 1, 0, 0, 1, 0], np.uint8),
]


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("invert", [False, True])
def test_manchester_roundtrip(bits, invert):
    chips = manchester.encode(bits, invert=invert)
    assert chips.size == 2 * bits.size
    np.testing.assert_array_equal(manchester.decode(chips, invert=invert), bits)


def test_manchester_invalid_pair_raises():
    with pytest.raises(ValueError, match="bit 1"):
        manchester.decode(np.array([1, 0, 1, 1], np.uint8))


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("initial", [0, 1])
@pytest.mark.parametrize("space", [False, True])
def test_biphase_roundtrip(bits, initial, space):
    chips = biphase.encode(bits, initial_level=initial, space=space)
    np.testing.assert_array_equal(biphase.decode(chips, space=space), bits)


def test_biphase_missing_boundary_raises():
    chips = biphase.encode(np.array([1, 1, 0, 1], np.uint8))
    chips[2] = chips[1]  # destroy the boundary transition before bit 1
    with pytest.raises(ValueError, match="boundary"):
        biphase.decode(chips)


@pytest.mark.parametrize("bits", BITS)
def test_fm0_roundtrip(bits):
    chips = fm0.encode(bits)
    np.testing.assert_array_equal(fm0.decode(chips), bits)
    # FM0 = biphase-space: '0' carries the mid-symbol transition
    pairs = chips.reshape(-1, 2)
    np.testing.assert_array_equal((pairs[:, 0] == pairs[:, 1]).astype(np.uint8), bits)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("initial", [0, 1])
def test_miller_roundtrip(bits, initial):
    chips = miller.encode(bits, initial_level=initial)
    np.testing.assert_array_equal(miller.decode(chips), bits)


@pytest.mark.parametrize("m", [2, 4, 8])
def test_miller_subcarrier_shape(m):
    bits = np.array([1, 0, 0, 1], np.uint8)
    wave = miller.modulate_subcarrier(bits, m)
    assert wave.size == bits.size * 2 * m  # 2*M half-cycles per bit
    # every half-bit contains a full-rate subcarrier: level alternates
    assert set(np.unique(wave)) <= {0, 1}
    spacings = np.diff(np.flatnonzero(np.diff(wave)))
    assert spacings.max() <= 2  # never flat longer than one half-cycle + boundary


def test_pie_symbol_lengths():
    spt = 20  # samples per Tari
    levels = pie.encode(np.array([0, 1], np.uint8), spt, data1_ratio=2.0, pw_ratio=0.5)
    len0, len1 = pie.symbol_samples(spt, 2.0)
    assert levels.size == len0 + len1
    pw = round(0.5 * spt)
    # data-0: high then a PW low tail
    np.testing.assert_array_equal(levels[: len0 - pw], 1)
    np.testing.assert_array_equal(levels[len0 - pw:len0], 0)
    # data-1 also ends in a PW low tail
    np.testing.assert_array_equal(levels[-pw:], 0)
    np.testing.assert_array_equal(levels[len0:-pw], 1)


def test_fsk_periods():
    spb = 40
    wave = fsk.modulate(np.array([0], np.uint8), spb, period0=8, period1=10)
    assert wave.size == spb
    spacings = np.diff(np.flatnonzero(np.diff(wave.astype(np.int8))))
    assert set(spacings) == {4}  # square wave toggles every period0/2 samples

    wave = fsk.modulate(np.array([1], np.uint8), spb, period0=8, period1=10)
    spacings = np.diff(np.flatnonzero(np.diff(wave.astype(np.int8))))
    assert set(spacings) == {5}


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("mode", [1, 2])
def test_psk_roundtrip(bits, mode):
    phases = psk.encode(bits, mode=mode)
    decoded = psk.decode(phases, mode=mode, first_bit=int(bits[0]))
    np.testing.assert_array_equal(decoded, bits)
