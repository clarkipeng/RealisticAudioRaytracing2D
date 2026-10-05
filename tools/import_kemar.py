#!/usr/bin/env python3
"""Reproduce the bundled KEMAR data using Python's standard library only."""
import hashlib
import io
import math
from pathlib import Path
import re
import struct
import sys
import urllib.request
import wave
import zipfile

URL = 'https://sound.media.mit.edu/resources/KEMAR/compact.zip'
SHA256 = '0bcd69f8e8760cf8eacff4a89594ceca7d44fb94f28ecf0bf19970003bbfb3e0'
root = Path(__file__).resolve().parents[1]
data = Path(sys.argv[1]).read_bytes() if len(sys.argv) > 1 else urllib.request.urlopen(URL).read()
assert hashlib.sha256(data).hexdigest() == SHA256, 'Unexpected source archive'
rows = {}
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for name in archive.namelist():
        match = re.search(r'H(-?\d+)e(\d+)a\.wav$', name)
        if not match:
            continue
        elevation, azimuth = map(int, match.groups())
        with wave.open(io.BytesIO(archive.read(name))) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (2, 2, 44100, 128)
            samples = struct.unpack('<256h', wav.readframes(128))
        left, right = [v / 32768 for v in samples[::2]], [v / 32768 for v in samples[1::2]]
        rows[elevation, azimuth] = left + right
        if azimuth not in (0, 180):
            rows[elevation, 360 - azimuth] = right + left
output = bytearray(struct.pack('<4siii', b'KMR1', 44100, len(rows), 128))
for (el, az), samples in sorted(rows.items()):
    e, a = math.radians(el), math.radians(az)
    direction = math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)
    output.extend(struct.pack('<259f', *direction, *samples))
path = root / 'Assets/Resources/Acoustics/Kemar.bytes'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(output)
print(f'{len(rows)} directions, 128 samples/ear, 44100 Hz; sha256={hashlib.sha256(output).hexdigest()}')
