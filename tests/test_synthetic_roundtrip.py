"""Step-1 round-trip tests: bits -> modulate -> (frontend/common) -> bits.

These exercise the synthetic modulators together with the envelope,
threshold, correlator, and carrier-cancellation primitives — the same
building blocks the band decoders (steps 2-4) will be assembled from.
"""

import numpy as np

from rfid_demod import synth
from rfid_demod.common import correlate, edges, envelope
from rfid_demod.encodings import fm0, manchester
from rfid_demod.frontend import carrier


def test_lf_ask_manchester_roundtrip():
    """LF-style path: ASK envelope -> smooth -> threshold -> Manchester."""
    sps = 32  # samples per half-bit chip
    bits = synth.random_bits(64, seed=7)
    chips = manchester.encode(bits)
    levels = synth.upsample(chips, sps)
    x = synth.ask_iq(levels, mod_index=0.6, freq_offset=2_000.0, sample_rate=500_000.0)
    x = synth.awgn(x, snr_db=20.0, seed=7)

    env = envelope.envelope(x)  # magnitude is carrier-offset invariant
    env = envelope.moving_average(env, 9)
    binary = envelope.to_binary(env, hysteresis=0.15)

    centers = np.arange(chips.size) * sps + sps // 2
    recovered_chips = binary[centers]
    np.testing.assert_array_equal(manchester.decode(recovered_chips), bits)


def test_backscatter_carrier_cancellation_recovers_tag():
    """UHF-style path: tag 40 dB below the carrier, quadrature channel phase.

    Without cancellation the envelope barely moves; after the DC tracker the
    residual is dominated by the tag waveform (checked via correlation,
    phase-invariant).
    """
    fs = 2_000_000.0
    sps = 8
    bits = synth.random_bits(128, seed=3)
    chips = fm0.encode(bits)
    tag = synth.upsample(chips, sps).astype(np.float64)

    pad = np.zeros(4000)  # carrier-only gap before the reply (tracker settles)
    levels = np.concatenate([pad, tag])
    x = synth.backscatter_iq(
        levels, carrier_amp=1.0, mod_amp=0.01,
        carrier_phase=0.3, mod_phase=1.2,
    )
    x = synth.awgn(x, snr_db=60.0, seed=3)  # 60 dB below carrier ~ 20 dB below tag

    # Before cancellation: modulation is invisible in the envelope.
    data = slice(pad.size, None)
    env = np.abs(x[data])
    assert np.std(env) / np.mean(env) < 0.02

    y = carrier.dc_block(x, fs, cutoff_hz=1e3)[data]
    template = tag - tag.mean()
    score = np.abs(np.vdot(template, y)) / (np.linalg.norm(template) * np.linalg.norm(y))
    assert score > 0.9


def test_find_preamble_in_noise():
    sps = 16
    preamble_bits = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 0], np.uint8)
    template = synth.upsample(manchester.encode(preamble_bits), sps).astype(np.float64)

    rng = np.random.default_rng(11)
    signal = 0.05 * rng.standard_normal(4000) + 0.2
    pos = 777
    signal[pos:pos + template.size] += 0.5 * template

    found, score = correlate.find_preamble(signal, template, threshold=0.7)
    assert abs(found - pos) <= 1
    assert score > 0.9

    # pure noise: no preamble claimed
    found, _ = correlate.find_preamble(0.05 * rng.standard_normal(4000), template, 0.7)
    assert found == -1


def test_edges_and_spacing_histogram():
    # Alternating chips at 32 samples/chip -> all edge spacings are 32.
    chips = manchester.encode(np.ones(16, np.uint8))  # 1,0,1,0,...
    binary = synth.upsample(chips, 32)
    positions, polarities = edges.find_edges(binary)
    assert positions.size == chips.size - 1
    assert set(polarities) == {-1, 1}
    values, counts = edges.spacing_histogram(edges.edge_spacings(positions))
    assert list(values) == [32]
    assert counts[0] == positions.size - 1
