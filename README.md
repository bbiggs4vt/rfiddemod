# rfid-demod

Software demodulator for RFID IQ captures: takes a complex baseband stream
(reader carrier at or near DC) and emits decoded frames — reader commands
and tag replies — as JSON lines. Targets the three major RFID families:
LF (EM4100, HID Prox, T5577), UHF (EPC Gen2), and HF (ISO 14443A/B,
ISO 15693). See [rfid-demod-brief.md](rfid-demod-brief.md) for the full
design brief.

```
IQ source ─► Front end (common) ─► Band router ─► Band decoder ─► Frame parser ─► JSONL sink
```

## Status (build order)

- [x] **1.** `io/`, `frontend/`, `common/` (CRC, envelope, correlator) + synthetic modulators
- [x] **2.** LF: EM4100 Manchester → HID FSK; end-to-end CLI
      (biphase/PSK1 chip decode and raw T5577 dumps still pending)
- [x] **3.** UHF Gen2: PIE reader decode → Query parsing → FM0 → Miller M=2/4/8
- [x] **4.** HF: 14443A → 14443B → 15693 (1-of-256 and dual-subcarrier
      15693 modes and 14443 high bit rates still pending)
- [ ] **5.** Streaming input, adaptive carrier canceller, performance pass

## Install & test

```sh
pip install -e .[dev]
pytest
```

## Usage

```sh
rfid-demod --band lf --rate 1e6 --in capture.cf32 --out frames.jsonl
```

Input formats: raw `cf32` (GNU Radio), raw interleaved `ci16`, 2-channel
WAV (I/Q), and SigMF (`cf32_le` / `ci16_le`). Raw formats need `--rate`;
WAV and SigMF carry it.

Decoded end to end today:

- **LF** — EM4100 (ASK/Manchester, RF/16–RF/128 clock auto-detected from
  the edge-spacing histogram, both polarities) and HID Prox (FSK2a,
  H10301 26-bit Wiegand parsed to facility code / card number). One JSONL
  frame per repeat detected in the capture.
- **UHF** — EPC Gen2 both directions: reader PIE (delimiter hunt, Tari /
  RTcal / TRcal measured from the capture, Query / QueryRep /
  QueryAdjust / ACK / NAK / Req_RN / Select parsed with CRC-5/16 checks)
  and tag backscatter (per-gap carrier cancellation + channel-phase
  projection, preamble correlation, FM0 and Miller M=2/4/8, TRext pilot
  tolerated; RN16, PC+EPC+CRC-16, handle replies). The Query's DR/M/TRext
  fields plus the measured TRcal configure the tag decoder (BLF =
  DR/TRcal), per the brief.

- **HF** — ISO 14443A (modified-Miller pause decode → REQA / anticollision
  / Select with parity, BCC and CRC-A checks; fc/16 OOK-subcarrier
  Manchester replies: ATQA, UID, SAK), ISO 14443B (10% ASK NRZ character
  framing, BPSK-subcarrier ATQB, CRC-B), and ISO 15693 (1-of-4 PPM
  commands, single-subcarrier high-data-rate replies, Inventory → UID).
  All three sub-decoders run per capture; structural gates keep them from
  firing on each other's waveforms.

The frontend's *global* DC block defaults to off: LF decodes the carrier
envelope itself, and the UHF/HF decoders cancel the carrier per reply
window (or in the subcarrier domain); `--dc-cutoff` forces it on.

## Layout

```
rfid_demod/
  io/            IQ readers (cf32/ci16/WAV/SigMF), JSONL frame sink
  frontend/      retune, carrier cancel (DC tracker), AGC, decimation
  common/        envelope/threshold, edge detection, preamble correlator,
                 CRCs (CCITT, Gen2 CRC-16 + CRC-5, CRC-A, CRC-B)
  encodings/     manchester, biphase, fm0, miller, pie, fsk, psk
  synth.py       bits → IQ with controllable SNR / carrier leakage
  decoders/      lf/ uhf/ hf/  (stubs until steps 2–4)
  parsers/       em4100, gen2 frames, ...  (steps 2–4)
  cli.py
tests/           unit + synthetic round-trip tests; fixtures/ for captures
```

## Design notes

- All RFID timing is carrier-derived: decoders align on a preamble
  (`common.correlate`) and then count fixed symbol periods — no
  free-running clock recovery.
- Synthetic-first testing: `synth.py` + `encodings/` generate IQ with
  controllable SNR, carrier leakage, and channel phase, so every decoder
  has a round-trip test before real captures exist.
- Carrier cancellation is a slow DC tracker for now
  (`frontend.carrier.dc_block`); `AdaptiveCanceller` is the step-5 hook
  for monostatic (high-leakage) captures.
