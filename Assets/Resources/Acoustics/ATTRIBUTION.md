# MIT KEMAR HRTF measurements

Copyright 1994 MIT Media Laboratory. Measurements by **Bill Gardner and Keith Martin**, MIT Media Lab, May 1994.

Source and permission: https://sound.media.mit.edu/resources/KEMAR.html
Documentation: https://sound.media.mit.edu/resources/KEMAR/hrtfdoc.txt
FAQ: https://sound.media.mit.edu/resources/KEMAR/KEMAR-FAQ.txt

The authors permit research and commercial use without restrictions provided they are credited. Retain this attribution when distributing the bundled data or derived renders/datasets.

`Kemar.bytes` is derived from `compact.zip`, SHA-256 `0bcd69f8e8760cf8eacff4a89594ceca7d44fb94f28ecf0bf19970003bbfb3e0`. Rebuild with `python3 tools/import_kemar.py [compact.zip]`; the converter needs no Python packages. Unity playback needs no network or external audio SDK.

The compact measurements are symmetric small-pinna responses, speaker-equalized by the authors, 128 samples per ear at 44.1 kHz. Conversion scales signed PCM16 by 1/32768, mirrors the measured half-space with channels exchanged, and maps MIT azimuth 90 degrees to Unity +X (right), elevation to +Y, front to +Z. No per-ear normalization, peak alignment, minimum-phase conversion or additional ear delay is applied. The packaged HRIR timing and interaural delay are retained. The authors cropped bulk measurement travel/equipment delay; the residual timing reference is not a calibrated absolute eardrum arrival time. Geometric path delay is added separately once.

Binary format: little-endian `KMR1`, int32(sampleRate, directionCount, taps), then each row has float32(x,y,z), left[taps], right[taps]. Direction points from listener toward incoming sound's origin. Coverage is -40 to +90 degrees elevation. Nearest-direction selection clamps naturally outside measured coverage. This mannequin is a generic far-field headphone model, not individualized hearing, a near-field head solver, or headphone equalization.
