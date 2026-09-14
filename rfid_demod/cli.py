"""Command-line entry point.

    rfid-demod --band uhf --rate 2e6 --in capture.cf32 --out frames.jsonl
    ... | rfid-demod --band lf --rate 1e6 --fmt cf32 --in - --stream

Offline mode loads the whole capture through the frontend; --stream (auto
for --in -) decodes incrementally with a sliding overlapped buffer and
emits frames as they become final.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Optional, Sequence

import numpy as np

from . import frontend
from .decoders import WORKING_RATES
from .io import JsonlWriter, iq
from .streaming import StreamDecoder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rfid-demod",
        description="Demodulate RFID frames from a complex baseband IQ capture.",
    )
    p.add_argument("--band", required=True, choices=("lf", "hf", "uhf"))
    p.add_argument("--in", dest="infile", required=True, metavar="PATH",
                   help="IQ capture (cf32/ci16/wav/sigmf), or '-' for stdin "
                        "(implies --stream)")
    p.add_argument("--out", default="-", metavar="PATH",
                   help="output JSONL file (default: stdout)")
    p.add_argument("--fmt", choices=("cf32", "ci16", "wav", "sigmf", "blue"),
                   help="input format (default: inferred from suffix, or "
                        "from the BLUE magic bytes)")
    p.add_argument("--rate", type=float,
                   help="input sample rate in Hz (required for raw captures "
                        "without embedded metadata)")
    p.add_argument("--freq-offset", type=float, default=0.0,
                   help="reader carrier offset from DC in Hz (retuned out)")
    p.add_argument("--dc-cutoff", type=float, default=None,
                   help="global carrier-cancellation tracker cutoff in Hz "
                        "(default: off — LF decodes the carrier envelope "
                        "itself and the UHF/HF decoders cancel the carrier "
                        "per reply window; set a cutoff to force the "
                        "frontend DC tracker on)")
    p.add_argument("--no-dc-block", action="store_true",
                   help="disable carrier cancellation")
    p.add_argument("--adaptive-cancel", action="store_true",
                   help="quiet-segment adaptive carrier canceller instead of "
                        "the DC tracker (offline mode; for monostatic "
                        "captures with drifting leakage)")
    p.add_argument("--agc", action="store_true",
                   help="sliding AGC instead of peak normalization")
    p.add_argument("--work-rate", type=float,
                   help="decimate to this rate; default per band: "
                        + ", ".join(f"{b}={r:g}" for b, r in WORKING_RATES.items()))
    p.add_argument("--stream", action="store_true",
                   help="incremental decode of a raw cf32/ci16 stream: "
                        "constant memory, frames emitted as they finalize. "
                        "Samples are taken at --rate (no decimation/AGC; "
                        "--freq-offset is applied phase-continuously)")
    p.add_argument("--block-seconds", type=float, default=0.25,
                   help="stream mode: decode block length (default 0.25 s)")
    return p


def _run_stream(args) -> int:
    if not args.rate:
        print("error: --rate is required in stream mode", file=sys.stderr)
        return 2
    fmt = args.fmt
    if fmt is None and args.infile != "-":
        try:
            fmt = iq._infer_format(iq.Path(args.infile))
        except ValueError:
            pass
    if fmt not in ("cf32", "ci16"):
        print("error: stream mode needs --fmt cf32 or ci16", file=sys.stderr)
        return 2
    if args.work_rate and args.work_rate != args.rate:
        print("error: decimation is not supported in stream mode; "
              "capture at the working rate", file=sys.stderr)
        return 2

    rate = args.rate
    sd = StreamDecoder(args.band, rate, block_seconds=args.block_seconds)
    base = 0
    count = 0
    with JsonlWriter(args.out) as writer:
        for block in iq.iter_blocks(args.infile, fmt):
            if args.freq_offset:
                n = base + np.arange(block.size)
                block = (block * np.exp(-2j * np.pi * args.freq_offset * n / rate)
                         ).astype(np.complex64)
            base += block.size
            for frame in sd.feed(block):
                writer.write(frame)
        for frame in sd.flush():
            writer.write(frame)
        count = writer.count
    print(f"wrote {count} frame(s)", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.stream or args.infile == "-":
        try:
            return _run_stream(args)
        except BrokenPipeError:
            # downstream consumer (head, grep -m, ...) closed the pipe
            import os
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            return 0

    capture = iq.load(args.infile, fmt=args.fmt)
    rate = args.rate or capture.sample_rate
    if not rate:
        print("error: --rate is required for raw captures without metadata",
              file=sys.stderr)
        return 2

    dc_cutoff = None if args.no_dc_block else args.dc_cutoff

    config = frontend.FrontendConfig(
        sample_rate=rate,
        freq_offset=args.freq_offset,
        dc_block_cutoff_hz=dc_cutoff,
        adaptive_cancel=args.adaptive_cancel,
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
