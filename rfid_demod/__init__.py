"""rfid_demod — software demodulator for RFID IQ captures.

Pipeline (see rfid-demod-brief.md):

    IQ source -> Front end (common) -> Band router -> Band decoder
              -> Frame parser -> JSONL sink

Subpackages
-----------
io        IQ readers (cf32 / ci16 / WAV / SigMF) and the JSONL frame sink.
frontend  Retune, carrier cancellation, AGC/normalization, decimation.
common    Envelope/threshold/edge primitives, preamble correlator, CRCs.
encodings Pure bit-level codecs (Manchester, biphase, FM0, Miller, PIE, FSK,
          PSK) shared by the synthetic modulators and the band decoders.
synth     Synthetic IQ generation (bits -> IQ) for round-trip tests.
decoders  Band decoders (lf / uhf / hf) — stubs until build-order steps 2-4.
parsers   Frame parsers (EM4100, Wiegand, Gen2, 14443A, ...) — steps 2-4.
"""

__version__ = "0.1.0"
