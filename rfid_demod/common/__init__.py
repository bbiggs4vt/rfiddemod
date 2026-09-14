"""Shared DSP primitives: envelope/threshold, edges, correlation, CRCs."""

from . import correlate, crc, edges, envelope

__all__ = ["correlate", "crc", "edges", "envelope"]
