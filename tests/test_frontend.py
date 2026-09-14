import numpy as np

from rfid_demod import frontend
from rfid_demod.frontend import agc, dc_block, decimate_to, normalize, retune


FS = 1_000_000.0


def _tone(freq, n=20000, fs=FS, amp=1.0):
    t = np.arange(n) / fs
    return (amp * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


def test_retune_moves_tone_to_dc():
    x = _tone(50_000)
    y = retune(x, 50_000, FS)
    spectrum = np.abs(np.fft.fft(y))
    assert np.argmax(spectrum) == 0
    # a tone at DC is (nearly) constant
    assert np.std(np.abs(y)) < 1e-3


def test_dc_block_removes_carrier_keeps_sidebands():
    x = 5.0 * np.ones(50000, dtype=np.complex64) + _tone(50_000, 50000, amp=0.1)
    y = dc_block(x, FS, cutoff_hz=1e3)
    tail = y[10000:]  # skip tracker settling
    assert abs(np.mean(tail)) < 0.01           # carrier gone
    spectrum = np.abs(np.fft.fft(tail))
    peak_bin = np.argmax(spectrum)
    freq = peak_bin / tail.size * FS
    assert abs(freq - 50_000) < 100            # sideband survives
    amp = spectrum[peak_bin] / tail.size       # complex exponential: |X_k| = A*N
    assert 0.09 < amp < 0.11                   # ~unchanged amplitude


def test_decimate_to():
    x = _tone(10_000, 40000)
    y, rate = decimate_to(x, FS, 250_000.0)
    assert rate == 250_000.0
    assert y.size == 10000
    freq = np.argmax(np.abs(np.fft.fft(y))) / y.size * rate
    assert abs(freq - 10_000) < 50


def test_decimate_noop_when_rate_not_lower():
    x = _tone(10_000, 100)
    y, rate = decimate_to(x, FS, FS)
    assert rate == FS and y.size == x.size


def test_normalize_and_agc():
    x = 3.0 * _tone(1000, 5000)
    assert np.isclose(np.max(np.abs(normalize(x))), 1.0)
    y = agc(x, FS, tau=1e-4)
    # after settling, tracked RMS gain flattens the amplitude near 1
    assert np.allclose(np.abs(y[2000:]), 1.0, atol=0.05)


def test_process_pipeline():
    # carrier at +25 kHz with a weak sideband; full frontend brings it to
    # DC, cancels it, normalizes, and decimates.
    x = _tone(25_000, 40000, amp=2.0) + _tone(75_000, 40000, amp=0.02)
    cfg = frontend.FrontendConfig(
        sample_rate=FS, freq_offset=25_000, dc_block_cutoff_hz=1e3,
        output_rate=500_000.0,
    )
    y, rate = frontend.process(x, cfg)
    assert rate == 500_000.0
    assert y.size == 20000
    tail = y[5000:]
    # residual carrier well below the (normalized) sideband
    spectrum = np.abs(np.fft.fft(tail))
    freq = np.argmax(spectrum) / tail.size * rate
    assert abs(freq - 50_000) < 200
