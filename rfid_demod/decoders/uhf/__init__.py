"""UHF decoder (EPC Gen2 / ISO 18000-6C).

Flow per the brief: decode the reader's PIE symbols first; the Query
command's DR/M/TRext fields plus the measured TRcal configure the
tag-side decoder (BLF = DR / TRcal, encoding FM0 or Miller M=2/4/8).
Tag replies are then decoded from the CW gaps between commands.

Carrier handling is per-gap (window mean removal in tag_path), so the
frontend's global DC block should stay off for this band.
Reference: gr-rfid tag_decoder_impl.cc / gate_impl.cc.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from rfid_demod.io import Frame
from rfid_demod.parsers import gen2_frames as gen2

from .pie_path import reader_frames
from .tag_path import decode_reply

_EXPECTED_REPLY = {
    "Query": "rn16",
    "QueryRep": "rn16",
    "QueryAdjust": "rn16",
    "ACK": "epc",
    "Req_RN": "handle",
}


def decode(samples: np.ndarray, sample_rate: float) -> List[Frame]:
    x = np.asarray(samples)
    env = np.abs(x)
    rframes = reader_frames(env, sample_rate)

    out: List[Frame] = []
    blf: Optional[float] = None
    m = 1
    trext = 0
    margin = max(4, int(2e-6 * sample_rate))

    for j, rf in enumerate(rframes):
        info = gen2.parse_reader(rf["bits"])
        command = info["command"]

        fields = dict(info)
        fields["tari_us"] = round(rf["tari"] / sample_rate * 1e6, 3)
        fields["rtcal_us"] = round(rf["rtcal"] / sample_rate * 1e6, 3)
        if command == "Query":
            m = info["m"]
            trext = info["trext"]
            if rf["trcal"]:
                fields["trcal_us"] = round(rf["trcal"] / sample_rate * 1e6, 3)
                blf = info["dr_ratio"] / (rf["trcal"] / sample_rate)
                fields["blf_hz"] = round(blf, 1)

        out.append(Frame(
            timestamp=rf["start"] / sample_rate,
            band="uhf",
            direction="R->T",
            bits="".join(map(str, rf["bits"])),
            fields=fields,
            crc_ok=info.get("crc_ok"),
        ))

        gap_start = rf["end"] + margin
        gap_end = rframes[j + 1]["start"] - margin if j + 1 < len(rframes) else env.size
        if blf is None or gap_end - gap_start <= 0:
            continue

        reply = decode_reply(x[gap_start:gap_end], sample_rate, blf, m, trext)
        if reply is None:
            continue
        bits, score, offset = reply
        tag_info = gen2.parse_tag(bits, _EXPECTED_REPLY.get(command))
        if tag_info is None:
            continue
        used = tag_info.pop("bits_used")
        tag_fields = dict(tag_info)
        tag_fields["encoding"] = "fm0" if m == 1 else f"miller_m{m}"
        tag_fields["blf_hz"] = round(blf, 1)
        tag_fields["preamble_score"] = round(score, 3)
        out.append(Frame(
            timestamp=(gap_start + offset) / sample_rate,
            band="uhf",
            direction="T->R",
            bits="".join(map(str, bits[:used])),
            fields=tag_fields,
            crc_ok=tag_info.get("crc_ok"),
        ))

    return out
