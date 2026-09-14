"""Adaptive carrier canceller: drifting monostatic leakage, tag bursts."""

import numpy as np

from rfid_demod import frontend, synth
from rfid_demod.encodings import fm0
from rfid_demod.frontend.carrier import AdaptiveCanceller, adaptive_cancel, dc_block

FS = 2_000_000.0


def _drifting_capture(seed=0, tag_amp=0.01, snr_db=60.0):
    """Drifting monostatic leakage + 3 FM0 tag bursts.

    Drift rates are on the harsh end of physical (thermal/mechanical)
    leakage drift; the linear interpolation between quiet segments must
    track them across the ~2 ms modulated bursts.
    """
    rng = np.random.default_rng(seed)
    n = 200_000
    t = np.arange(n) / FS
    leak = (1.0 + 0.04 * np.sin(2 * np.pi * 8.0 * t)) \
        * np.exp(1j * (0.4 + 0.4 * np.sin(2 * np.pi * 12.0 * t)))

    levels = np.zeros(n)
    burst_starts = [30_000, 90_000, 150_000]
    for s in burst_starts:
        bits = rng.integers(0, 2, 200, dtype=np.uint8)
        chips = synth.upsample(fm0.encode(bits), 8)   # 3200 samples per burst
        levels[s:s + chips.size] = chips

    x = leak + tag_amp * levels * np.exp(1j * 1.3)
    x = synth.awgn(x.astype(np.complex64), snr_db, seed=seed)

    quiet = np.ones(n, dtype=bool)
    for s in burst_starts:
        quiet[s - 500:s + 3_700] = False
    return x, levels, quiet, burst_starts


def test_adaptive_cancel_drifting_leakage():
    x, levels, quiet, bursts = _drifting_capture()
    y = adaptive_cancel(x, FS, segment_us=100.0)

    # residual carrier in quiet regions: far below the 0.04 drift amplitude
    residual = float(np.sqrt(np.mean(np.abs(y[quiet]) ** 2)))
    assert residual < 0.01

    # tag bursts survive: after the per-window mean removal a decoder
    # would apply, the residual correlates with the transmitted chips
    s = bursts[0]
    seg = y[s:s + 3_200]
    seg = seg - seg.mean()
    template = levels[s:s + 3_200] - levels[s:s + 3_200].mean()
    score = np.abs(np.vdot(template, seg)) / (
        np.linalg.norm(template) * np.linalg.norm(seg))
    assert score > 0.9


def test_adaptive_beats_dc_tracker_on_drift():
    # A DC tracker slow enough not to bias through the tag modulation
    # (cutoff below the burst rate) cannot follow the 8-12 Hz leakage
    # drift; the quiet-segment interpolation can.
    x, _, quiet, _ = _drifting_capture(seed=1)
    y_adaptive = adaptive_cancel(x, FS, segment_us=100.0)
    y_dc = dc_block(x, FS, cutoff_hz=5.0)
    rms = lambda v: float(np.sqrt(np.mean(np.abs(v[quiet][20_000:]) ** 2)))
    assert rms(y_adaptive) < 0.5 * rms(y_dc)


def test_adaptive_cancel_constant_leakage():
    rng = np.random.default_rng(2)
    n = 50_000
    x = (0.8 * np.exp(1j * 0.7)) * np.ones(n, dtype=np.complex64)
    x = synth.awgn(x, 50.0, seed=2)
    y = adaptive_cancel(x, FS)
    assert float(np.mean(np.abs(y))) < 0.01


def test_adaptive_cancel_short_input_falls_back():
    x = np.full(100, 2.0 + 1.0j, dtype=np.complex64)
    y = adaptive_cancel(x, FS, segment_us=1000.0)   # < 4 segments
    assert np.allclose(y, 0, atol=1e-6)


def test_canceller_class_and_frontend_flag():
    x, _, quiet, _ = _drifting_capture(seed=3)
    canceller = AdaptiveCanceller(FS, segment_us=100.0)
    y = canceller(x)
    assert float(np.sqrt(np.mean(np.abs(y[quiet]) ** 2))) < 0.005

    cfg = frontend.FrontendConfig(sample_rate=FS, adaptive_cancel=True)
    z, rate = frontend.process(x, cfg)
    assert rate == FS and z.size == x.size
    # normalized output still shows the leak suppressed in quiet regions
    assert float(np.mean(np.abs(z[quiet]))) < 0.5 * float(np.mean(np.abs(z[~quiet])))
