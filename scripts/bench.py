#!/usr/bin/env python3
"""Decode-throughput benchmark: synthetic capture per band, wall time vs
capture duration. Run from the repo root: python scripts/bench.py"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import test_hf_e2e as hf_t          # noqa: E402
import test_lf_e2e as lf_t          # noqa: E402
import test_uhf_e2e as uhf_t        # noqa: E402
from rfid_demod.decoders import hf, lf, uhf  # noqa: E402


def bench(name, x, fs, decode, repeat=3):
    x = np.asarray(x)
    # warm-up + best-of-N
    best = min(
        (lambda t0: (decode(x, fs), time.perf_counter() - t0))(time.perf_counter())[1]
        for _ in range(repeat)
    )
    frames = decode(x, fs)
    dur = x.size / fs
    print(f"{name:22s} {dur * 1e3:8.1f} ms capture  {best * 1e3:8.1f} ms decode  "
          f"{dur / best:6.1f}x realtime  {len(frames):3d} frames")


def main():
    bench("lf em4100 (1 Msps)",
          lf_t.em4100_capture(0x1234567890, repeats=30), lf_t.FS, lf.decode)
    bench("lf hid (1 Msps)",
          lf_t.hid_capture(118, 1603, repeats=12), lf_t.FS, lf.decode)

    rounds = [uhf_t.build_inventory(m_code=0, seed=s) for s in range(4)]
    bench("uhf gen2 (2 Msps)", np.concatenate(rounds), uhf_t.FS, uhf.decode)

    seqs = [hf_t.build_14443a(seed=s) for s in range(3)]
    bench("hf 14443a (13.56 Msps)", np.concatenate(seqs), hf_t.FS, hf.decode)
    bench("hf 15693 (13.56 Msps)", hf_t.build_15693(), hf_t.FS, hf.decode)


if __name__ == "__main__":
    main()
