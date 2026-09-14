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

# Below these rates the band's symbol timing cannot be represented at all
# (windows collapse to zero samples) and a decode would only produce
# garbage; decoders refuse to run instead. From the brief's minimums.
MINIMUM_RATES = {
    "lf": 250e3,
    "uhf": 1e6,
    "hf": 4e6,
}


def check_rate(band: str, sample_rate: float) -> None:
    minimum = MINIMUM_RATES[band]
    if sample_rate < minimum:
        raise ValueError(
            f"sample rate {sample_rate:g} Hz is too low for the {band} band "
            f"(minimum {minimum:g} Hz; recommended {WORKING_RATES[band]:g} Hz) "
            "— the capture bandwidth cannot contain this band's signaling")
