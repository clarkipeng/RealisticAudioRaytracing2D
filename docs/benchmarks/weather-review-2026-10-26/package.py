"""Package matched weather stems with labeled reference stills (no new walking render).

Usage: python package.py <weather-review-dir> <previous-picture-capture> <delivery-dir>
Requires NumPy, SciPy, Pillow and ffmpeg. Runs without audio playback.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
from sequence_movie import TakeActivity, SoundMap, rotate

os.nice(19)
root, reference, delivery = map(Path, sys.argv[1:])
delivery.mkdir(parents=True, exist_ok=True)
rate, fps = 44100, 24

def audio(name, listener):
    directory = root / (name + '-audio')
    report = json.loads((directory / 'run.json').read_text())
    meta = json.loads((root / 'probes' / (name + '-take') / 'take.json').read_text())
    assert report['complete'] and report['sceneHash'] == meta['sceneHash'] and report['takeStateHash'] == meta['stateHash'] and report['takeEventHash'] == meta['eventHash']
    fs, pcm = wavfile.read(directory / f'listener-{listener}.wav')
    assert fs == rate and pcm.ndim == 1 and np.isfinite(pcm).all()
    start = report['dspLatencySamples'] + 2 * rate
    return pcm[start:start + 4 * rate].copy()

def joined(parts):
    faded = []
    for part in parts:
        assert len(part) == 4 * rate
        part = part.copy(); n = round(.015 * rate)
        fade = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
        part[:n] *= fade; part[-n:] *= fade[::-1]
        faded.append(part)
    result = np.concatenate(faded)
    assert np.isfinite(result).all() and np.max(abs(result)) < 1
    return result

wind_out, wind_in = audio('wind-dense', 0), audio('wind-dense', 1)
rain_out, rain_in = audio('rain-outside', 0), audio('rain-inside', 0)
mixed = joined([rain_out + wind_out, rain_in + wind_in])
for name, pcm in [('roof-weather-comparison.wav', mixed),
                  ('wind-before-after.wav', joined([audio('wind-baseline', 0), wind_out])),
                  ('wind-outside-inside.wav', joined([wind_out, wind_in]))]:
    wavfile.write(delivery / name, rate, pcm)

rain = TakeActivity(root / 'probes/rain-outside-take')
inside = TakeActivity(root / 'probes/rain-inside-take')
wind = TakeActivity(root / 'probes/wind-dense-take')
assert rain.scene['geometryHash'] == inside.scene['geometryHash'] == wind.scene['geometryHash']
geometry_owners = [i for i, obj in enumerate(rain.scene['objects']) if obj['primitiveCount']]
for other in [inside, wind]:
    np.testing.assert_array_equal(rain.poses(0)[geometry_owners, :7], other.poses(0)[geometry_owners, :7])
# Merge only visualization data from the separately authenticated source takes.
rain.scene['sources'].append(wind.scene['sources'][0])
rain.sources.append(1); rain.planes[1] = dict(wind.planes[0], source=1)
for field in ['births', 'times', 'voices', 'power', 'reference', 'max_gain']:
    getattr(rain, field)[1] = getattr(wind, field)[0]
map_overlay = SoundMap(reference, rain); map_overlay.route = 'comparison'
with (reference / 'view.csv').open() as stream:
    views = {int(row['frame']): row for row in csv.DictReader(stream)}
frames = [30, 540]
for index, take in enumerate([rain, inside]):
    pose = take.poses(0)[take.scene['listeners'][0]]; view = views[frames[index]]
    eye = np.array([float(view[k]) for k in ['x', 'y', 'z']])
    forward = np.array([float(view[k]) for k in ['forwardX', 'forwardY', 'forwardZ']])
    assert np.linalg.norm(pose[:3] - eye) < 1e-4
    assert np.linalg.norm(rotate(pose[3:7], np.array([0., 0., 1.])) - forward) < 1e-4
images = [Image.open(reference / 'frames' / f'frame_{f:04}.png').convert('RGB') for f in frames]
font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
font = ImageFont.truetype(font_path, 25); small = ImageFont.truetype(font_path, 18)
movie = delivery / 'roof-weather-comparison.mp4'
ffmpeg = shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg'
cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24',
       '-video_size', '1280x720', '-framerate', str(fps), '-i', 'pipe:0', '-i', str(delivery / 'roof-weather-comparison.wav'),
       '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '1',
       '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-t', '8', str(movie)]
with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
    try:
        for frame in range(8 * fps):
            section = frame // (4 * fps); seconds = 2 + (frame % (4 * fps)) / fps
            picture = map_overlay.draw(images[section].copy(), seconds, views[frames[section]]).convert('RGBA')
            caption = Image.new('RGBA', (850, 96), (0, 0, 0, 0)); draw = ImageDraw.Draw(caption)
            draw.rounded_rectangle((0, 0, 849, 95), 12, fill=(12, 21, 29, 242))
            draw.text((18, 12), ('OUTSIDE' if section == 0 else 'INSIDE — DOOR CLOSED') + '  /  RAIN + WIND', font=font, fill='white')
            draw.text((18, 51), 'Stationary audio comparison • Matched emissions • Same playback gain', font=small, fill=(191, 209, 219))
            picture.alpha_composite(caption, (20, 604))
            ImageDraw.Draw(picture).text((22, 20), 'Reference scene view', font=small, fill='white', stroke_width=1, stroke_fill='black')
            process.stdin.write(picture.convert('RGB').tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Encoding failed')
    finally:
        if process.poll() is None:
            process.terminate(); process.wait()
subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-ss', '5', '-i', str(movie), '-frames:v', '1', str(delivery / 'roof-weather-preview.png')], check=True)
manifest = dict(seconds=8, sampleRate=rate, masterGain=1, cropSeconds=[2, 6], sequence=['outside', 'closed interior'],
                edgeFadeMilliseconds=15, mixedPeak=float(abs(mixed).max()),
                picture='Two reference stills from the previous capture; receiver position/heading checked. This is not a new walking film.',
                audio='Sum of independently traced rain and wind sources. No per-segment normalization.', files={})
for name in ['roof-weather-comparison.mp4', 'roof-weather-comparison.wav', 'wind-before-after.wav', 'wind-outside-inside.wav', 'roof-weather-preview.png']:
    path = delivery / name
    manifest['files'][name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(root / 'delivery.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
