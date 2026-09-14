"""Command-line entry point.

    rfid-demod --band uhf --rate 2e6 --in capture.cf32 --out frames.jsonl

The IO -> frontend -> band-router path is complete; band decoders arrive
in build-order steps 2-4 and plug into the dispatch below unchanged.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Optional, Sequence

from . import frontend
from .decoders import WORKING_RATES
from .io import JsonlWriter, iq


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rfid-demod",
        description="Demodulate RFID frames from a complex baseband IQ capture.",
    )
    p.add_argument("--band", required=True, choices=("lf", "hf", "uhf"))
    p.add_argument("--in", dest="infile", required=True, metavar="PATH",
                   help="IQ capture (cf32/ci16/wav/sigmf)")
    p.add_argument("--out", default="-", metavar="PATH",
                   help="output JSONL file (default: stdout)")
    p.add_argument("--fmt", choices=("cf32", "ci16", "wav", "sigmf"),
                   help="input format (default: inferred from suffix)")
    p.add_argument("--rate", type=float,
                   help="input sample rate in Hz (required for raw captures "
                        "without embedded metadata)")
    p.add_argument("--freq-offset", type=float, default=0.0,
                   help="reader carrier offset from DC in Hz (retuned out)")
    p.add_argument("--dc-cutoff", type=float, default=None,
                   help="carrier-cancellation tracker cutoff in Hz "
                        "(default: 1000 for hf/uhf, off for lf — the LF "
                        "envelope rides on the carrier itself)")
    p.add_argument("--no-dc-block", action="store_true",
                   help="disable carrier cancellation")
    p.add_argument("--agc", action="store_true",
                   help="sliding AGC instead of peak normalization")
    p.add_argument("--work-rate", type=float,
                   help="decimate to this rate; default per band: "
                        + ", ".join(f"{b}={r:g}" for b, r in WORKING_RATES.items()))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    capture = iq.load(args.infile, fmt=args.fmt)
    rate = args.rate or capture.sample_rate
    if not rate:
        print("error: --rate is required for raw captures without metadata",
              file=sys.stderr)
        return 2

    if args.no_dc_block:
        dc_cutoff = None
    elif args.dc_cutoff is not None:
        dc_cutoff = args.dc_cutoff
    else:
        dc_cutoff = None if args.band == "lf" else 1e3

    config = frontend.FrontendConfig(
        sample_rate=rate,
        freq_offset=args.freq_offset,
        dc_block_cutoff_hz=dc_cutoff,
        agc=args.agc,
        output_rate=args.work_rate or WORKING_RATES[args.band],
    )
    samples, work_rate = frontend.process(capture.samples, config)

    decoder = importlib.import_module(f"rfid_demod.decoders.{args.band}")
    try:
        frames = decoder.decode(samples, work_rate)
        with JsonlWriter(args.out) as writer:
            for frame in frames:
                writer.write(frame)
            count = writer.count
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    print(f"wrote {count} frame(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
