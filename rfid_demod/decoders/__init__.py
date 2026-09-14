"""Band decoders. Each band module exposes ``decode(samples, sample_rate)``
yielding :class:`rfid_demod.io.Frame` records.

Status (build order): lf = step 2, uhf = step 3, hf = step 4.
"""

# Recommended working sample rates per band (Hz), from the brief.
WORKING_RATES = {
    "lf": 1e6,
    "uhf": 2e6,
    "hf": 13.56e6,
}
