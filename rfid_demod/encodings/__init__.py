"""Pure bit-level line codes, independent of sample rate.

Conventions:

* "bits" are numpy uint8 arrays of 0/1, first-transmitted bit first.
* "chips" are 0/1 level arrays at the code's chip rate (2 chips/bit for the
  biphase family); upsample with :func:`rfid_demod.synth.upsample` to get a
  waveform.

Modules: manchester, biphase (mark/space), fm0, miller, pie, fsk, psk.
Encoders exist for all of them (needed by the synthetic modulators of
build-order step 1); sample-domain demodulators arrive with the band
decoders in steps 2-4.
"""

from . import biphase, fm0, fsk, manchester, miller, pie, psk

__all__ = ["biphase", "fm0", "fsk", "manchester", "miller", "pie", "psk"]
