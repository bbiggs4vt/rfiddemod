# RFID IQ Demodulator — Project Brief

## Goal

Build a software demodulator that takes a complex baseband IQ stream (reader carrier at or near DC) and emits decoded RFID frames — reader commands and tag replies — for the three major RFID families:

| Band | Carrier | Standards | Priority |
|------|---------|-----------|----------|
| LF | 125 / 134.2 kHz | EM4100, HID Prox (FSK), T5577 | 1 (simplest) |
| UHF | 860–960 MHz | ISO 18000-6C / EPC Gen2 | 2 (best reference code) |
| HF | 13.56 MHz | ISO 14443A/B, ISO 15693 | 3 (most variants) |

Input: IQ samples (complex64 or interleaved int16) from file or stream, plus metadata (band, sample rate, center frequency).
Output: decoded frames as structured records (JSON lines): timestamp, direction (R→T / T→R), raw bits, parsed fields (UID / EPC / command), CRC status.

## Non-goals (v1)

- Transmitting or acting as a reader
- Cryptographic layers (MIFARE Crypto1, DESFire, Gen2 access passwords) — we stop at bits/frames
- Real-time guarantees — file-based offline processing first, streaming later

## Architecture

```
IQ source ─► Front end (common) ─► Band router ─► Band decoder ─► Frame parser ─► JSONL sink
```

### Common front end (`frontend/`)
1. **Retune** — shift so reader carrier is at DC (`x * exp(-j2πf_off t)`).
2. **Carrier cancellation** — tag replies are 30–60 dB below the carrier. Start with a slow DC tracker (single-pole IIR, ~1 kHz cutoff) or high-pass; leave a hook for an adaptive canceller later.
3. **AGC / normalization**.
4. **Decimation** to the band's working rate (see table below).

### Working sample rates
| Band | Minimum | Recommended | Why |
|------|---------|-------------|-----|
| LF | 500 ksps | 1 Msps | data is a few kHz on a 125 kHz carrier |
| HF | 4 Msps | 13.56 or 27.12 Msps | subcarrier at ±847.5 kHz; all timings are integer carrier cycles (fc/128, fc/16) |
| UHF | 2 Msps | 2 Msps | covers BLF up to 640 kHz; matches gr-rfid reference |

Key design rule: **all RFID timing is derived from the reader carrier**. Use fixed symbol periods (in samples) rather than free-running clock recovery. Preamble correlation for alignment, then count.

## Band decoders

### LF (`decoders/lf/`)
- Envelope: `|x|` → low-pass at ~10× bitrate → threshold.
- Bit periods: RF/32 or RF/64 (carrier cycles per bit) — auto-detect from edge spacing histogram.
- Encoders to support: Manchester, Biphase, PSK1, FSK (FSK2a for HID: RF/8 and RF/10).
- Formats:
  - **EM4100**: 64 bits = 9 header ones + 10 rows × (4 data + 1 parity) + 4 column parity + stop bit. Output 40-bit ID.
  - **HID Prox**: FSK2a → 26–37-bit Wiegand payload with facility code / card number.
  - **T5577**: configurable; decode as raw blocks.
- Reference: Proxmark3 `client/src/cmddata.c`, `cmdlf*.c`.

### UHF EPC Gen2 (`decoders/uhf/`)
Reader → Tag:
- PIE (pulse-interval encoding), ASK or PR-ASK. Measure low-pulse widths against Tari (6.25–25 µs). Data-0 = 1 Tari, data-1 = 1.5–2 Tari.
- Detect preamble/frame-sync (delimiter 12.5 µs + Tari + RTcal + TRcal).
- **Decode `Query` first**: its DR, M, TRext fields tell you the tag's backscatter link frequency (BLF = DR / TRcal) and encoding (FM0, Miller M=2/4/8). Configure the T→R decoder from this.

Tag → Reader:
- Bandpass (or mix down) at BLF, envelope.
- Correlate against preamble (FM0: 6 symbols `1010v1` with violation; Miller: 4 or 16 subcarrier cycles + `010111`).
- Symbol timing from preamble → FM0 / Miller decode.
- Frames: RN16 (16 bits), then PC (16) + EPC (96 typ.) + CRC-16 (CCITT, poly 0x1021, init 0xFFFF).
- Reference: `gr-rfid` (`tag_decoder_impl.cc`, `gate_impl.cc`).

### HF (`decoders/hf/`)
ISO 14443A:
- R→T: 100% ASK, modified Miller, 106 kbps (fc/128 = 9.44 µs/bit). Envelope → detect pauses (~2–3 µs). Short frame (7 bits, REQA/WUPA) vs standard frame (bytes + odd parity).
- T→R: OOK subcarrier at fc/16 = 847.5 kHz, Manchester at 106 kbps. Bandpass at ±847.5 kHz → envelope → Manchester decode. SOF = one bit-period with subcarrier in first half. Parity per byte, CRC-A (poly 0x1021, init 0x6363).
- Parse: ATQA, UID (cascade levels), SAK, then raw APDUs.

ISO 14443B:
- R→T: 10% ASK NRZ-L, 106 kbps. T→R: BPSK on 847.5 kHz subcarrier, NRZ-L. CRC-B (init 0xFFFF, inverted output).

ISO 15693:
- R→T: 1-of-4 or 1-of-256 PPM, 100%/10% ASK. T→R: single subcarrier (fc/32 = 423.75 kHz, Manchester) or dual (423.75 + 484.28 kHz, FSK-like). Support single-subcarrier high-data-rate (26.48 kbps) first.

Reference: Proxmark3 `armsrc/iso14443a.c`, `iso15693.c`; libnfc.

## Suggested layout

```
rfid_demod/
  __init__.py
  io/            # IQ readers (raw cf32/ci16, SigMF, WAV), JSONL writer
  frontend/      # retune, carrier cancel, agc, decimate
  common/        # envelope, thresholds, edge detection, correlators, CRC (A/B/CCITT/16)
  encodings/     # manchester, miller, fm0, pie, fsk, psk, bpsk — pure bit-level codecs w/ tests
  decoders/
    lf/  uhf/  hf/
  parsers/       # em4100, hid_wiegand, gen2_frames, iso14443a_frames, ...
  cli.py         # rfid-demod --band uhf --rate 2e6 --in capture.cf32 --out frames.jsonl
tests/
  fixtures/      # short IQ captures per band + expected JSONL
  test_encodings.py  # synthetic round-trip tests: encode bits → modulate → demod → compare
```

Language: Python + numpy/scipy for v1 (fast to iterate, easy to plot). Structure so hot paths can be moved to a compiled block later.

## Testing strategy

1. **Synthetic first**: write a small modulator per encoding (bits → IQ) so every decoder has a round-trip test with controllable SNR and carrier leakage. This unblocks development before real captures exist.
2. **Real captures**: SigMF files per band; validate against Proxmark3 / URH output for the same capture.
3. Metrics: frame detection rate, bit error rate vs SNR, false-positive frames on noise.

## Build order

1. `io/`, `frontend/`, `common/` (CRC, envelope, correlator) + synthetic modulators.
2. LF: EM4100 Manchester → HID FSK. End-to-end CLI working.
3. UHF Gen2: PIE reader decode → Query parsing → FM0 tag decode → Miller variants.
4. HF 14443A → 14443B → 15693.
5. Streaming input, adaptive carrier canceller, performance pass.

## Open questions to resolve during scaffolding

- Antenna topology for real captures (monostatic → heavy carrier leakage, bistatic → less). Affects how aggressive carrier cancellation needs to be.
- Whether to require the user to specify band, or auto-detect from center frequency metadata.
- Plotting/debug hooks (matplotlib dumps of envelope + detected edges) — worth building early.

## References

- ISO/IEC 14443-2/-3, ISO/IEC 15693-2/-3, GS1 EPC Gen2 (UHF Class 1 Gen2 v2.x spec, freely available from GS1)
- gr-rfid — GNU Radio EPC Gen2 reader (Kargas, Mavromatis, Papadias)
- Proxmark3 firmware and client (LF/HF reference demodulators)
- Universal Radio Hacker — for inspecting captures and hand-decoding
- libnfc — ISO 14443 framing reference
