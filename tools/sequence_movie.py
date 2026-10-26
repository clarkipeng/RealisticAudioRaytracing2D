#!/usr/bin/env python3
"""Composite a captured first-person movie with source activity from its numeric take.

Requires numpy, Pillow and ffmpeg. The map shows emission, not listener loudness.
Fixed clip voices and masked fixed-site planes are supported; no audio is played.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


def vector(value):
    return np.array([value[k] for k in ('x', 'y', 'z')])


def rotate(q, v):
    return v + 2 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)


class TakeActivity:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / 'take.json').read_text())
        m = self.meta
        if m['version'] != 6 or not m['complete'] or m['scene']['dimension'] != 1:
            raise ValueError('Expected a complete spatial version-6 take.')
        for file, key in [('states.bin', 'stateHash'), ('events.bin', 'eventHash')]:
            if hashlib.sha256((self.directory / file).read_bytes()).hexdigest() != m[key]:
                raise ValueError(f'{file} does not match the take.')
        self.scene = m['scene']
        self.rate = m['sampleRate']
        dtype = np.dtype([('sample', '<i8'), ('state', '<f4', (len(self.scene['objects']), 20))])
        self.frames = np.fromfile(self.directory / 'states.bin', dtype=dtype)
        if len(self.frames) != m['frameCount']:
            raise ValueError('Incomplete state payload.')
        self.events = list(struct.iter_unpack('<qqiif?', (self.directory / 'events.bin').read_bytes()))
        if len(self.events) != m['eventCount']:
            raise ValueError('Incomplete event payload.')
        self.sources = list(dict.fromkeys(e[2] for e in self.events
                                         if e[3] == 0 and not self.scene['sources'][e[2]]['streamed']))
        self.times, self.voices, self.power, self.reference, self.max_gain = {}, {}, {}, {}, {}
        for source in self.sources:
            clip = self.scene['clips'][self.scene['sources'][source]['clipIndex']]
            payload = (self.directory / clip['file']).read_bytes()
            if hashlib.sha256(payload).hexdigest() != clip['sha256']:
                raise ValueError('Clip hash mismatch.')
            pcm = np.frombuffer(payload, dtype='<f4').astype(np.float64)
            if len(pcm) != clip['samples'] or not np.isfinite(pcm).all():
                raise ValueError('Invalid source PCM.')
            self.power[source] = np.concatenate(([0.0], np.cumsum(pcm * pcm)))
            block = max(1, self.rate // 50)
            energy = np.add.reduceat(pcm * pcm, np.arange(0, len(pcm), block)) / block
            self.reference[source] = max(1e-6, math.sqrt(float(np.quantile(energy, .95))))
            times, voices, voice = [], [], None
            for sample, offset, index, action, gain, loop in self.events:
                if index != source:
                    continue
                if action == 0:
                    voice = (sample, offset, gain, loop)
                elif action == 1:
                    voice = None
                elif voice is not None:
                    voice = (voice[0], voice[1], gain, voice[3])
                times.append(sample)
                voices.append(voice)
            self.times[source], self.voices[source] = times, voices
            self.max_gain[source] = max((v[2] for v in voices if v is not None), default=1)
        self.planes = {p['source']: p for p in self.scene.get('planes', [])}
        if any(p.get('generatorVersion', 0) != 0 for p in self.planes.values()):
            raise ValueError('Burst planes need a burst-aware activity view; this view supports fixed sites.')

    def poses(self, seconds):
        sample = seconds * self.rate
        right = min(len(self.frames) - 1, int(np.searchsorted(self.frames['sample'], sample)))
        left = max(0, right - 1)
        a, b = self.frames[left], self.frames[right]
        blend = np.clip((sample - a['sample']) / max(1, int(b['sample'] - a['sample'])), 0, 1)
        result = a['state'].astype(float).copy()
        result[:, :3] += (b['state'][:, :3] - result[:, :3]) * blend
        for i, q in enumerate(result[:, 3:7]):
            r = b['state'][i, 3:7].astype(float)
            dot = float(np.dot(q, r))
            if dot < 0:
                r, dot = -r, -dot
            angle = math.acos(min(1, dot))
            q = ((1 - blend) * q + blend * r if angle < 1e-5 else
                 (math.sin((1 - blend) * angle) * q + math.sin(blend * angle) * r) / math.sin(angle))
            result[i, 3:7] = q / np.linalg.norm(q)
        return result

    def activity(self, source, seconds):
        sample = math.floor(seconds * self.rate)
        index = bisect.bisect_right(self.times[source], sample) - 1
        voice = self.voices[source][index] if index >= 0 else None
        if voice is None:
            return False, 0.0
        start, offset, gain, loop = voice
        power = self.power[source]
        length = len(power) - 1
        end = offset + sample - start + 1
        if gain <= 0 or (not loop and end > length):
            return False, 0.0
        count = min(max(1, round(.06 * self.rate)), sample - start + 1)

        def integral(position):
            if loop:
                turns, phase = divmod(position, length)
                return turns * power[-1] + power[phase]
            return power[min(length, max(0, position))]

        rms = math.sqrt(max(0, integral(end) - integral(end - count)) / count)
        level = min(1, rms * gain / max(1e-12, self.reference[source] * self.max_gain[source]))
        return rms * gain > 1e-6, level


class SoundMap:
    COLORS = [(255, 195, 89), (255, 132, 116), (94, 219, 174), (104, 190, 255), (201, 164, 255)]

    def __init__(self, capture, take):
        self.take = take
        self.map = json.loads((capture / 'map.json').read_text())
        self.size = 330
        self.background = ImageEnhance.Brightness(Image.open(capture / 'map.png').convert('RGBA')).enhance(.86).resize((330, 330))
        font = next((p for p in ['/System/Library/Fonts/Supplemental/Arial.ttf',
                                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists()), None)
        self.font = ImageFont.truetype(font, 17) if font else ImageFont.load_default(size=17)
        self.small = ImageFont.truetype(font, 13) if font else ImageFont.load_default(size=13)
        self.title = ImageFont.truetype(font, 20) if font else ImageFont.load_default(size=20)
        self.mask_cache = {}

    def point(self, v):
        m = self.map
        return np.array([.5 + (v[0] - m['centerX']) / m['span'],
                         .5 - (v[2] - m['centerZ']) / m['span']]) * self.size

    def plane_mask(self, source, pose):
        key = (source, tuple(pose[:7]))
        if key in self.mask_cache:
            return self.mask_cache[key]
        p = self.take.planes[source]
        u, v = rotate(pose[3:7], vector(p['u'])), rotate(pose[3:7], vector(p['v']))
        origin = self.point(pose[:3] - u * .5 + v * .5)
        x = (self.point(pose[:3] + u) - self.point(pose[:3])) / p['width']
        y = (self.point(pose[:3] - v) - self.point(pose[:3])) / p['height']
        inverse = np.linalg.inv(np.column_stack((x, y)))
        affine = np.column_stack((inverse, -inverse @ origin)).flatten()
        pixels = np.clip(np.array(p['mask']).reshape(p['height'], p['width'])[::-1] * 255, 0, 255).astype('uint8')
        mask = Image.fromarray(pixels).transform((self.size, self.size), Image.Transform.AFFINE, affine,
                                               resample=Image.Resampling.NEAREST)
        # Cache only the latest pose per source, keeping memory bounded for moving planes.
        self.mask_cache = {k: v for k, v in self.mask_cache.items() if k[0] != source}
        self.mask_cache[key] = mask
        return mask

    def draw(self, frame, seconds, view):
        t = self.take
        poses = t.poses(seconds)
        eye = np.array([float(view[k]) for k in ('x', 'y', 'z')])
        heading = np.array([float(view['forwardX']), -float(view['forwardZ'])])
        heading /= max(1e-8, np.linalg.norm(heading))
        height = 438 + 34 * len(t.sources)
        panel = Image.new('RGBA', (378, height))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((0, 0, 377, height - 1), 16, fill=(12, 21, 29, 240), outline=(85, 109, 120, 200))
        draw.text((24, 17), 'SOUND MAP', font=self.title, fill='white')
        draw.text((292, 21), f'{int(seconds)//60}:{int(seconds)%60:02}', font=self.small, fill=(193, 210, 219))
        area = self.background.copy()
        activity = {source: t.activity(source, seconds) for source in t.sources}
        for i, source in enumerate(t.sources):
            if source not in t.planes:
                continue
            active, level = activity[source]
            color = self.COLORS[i % len(self.COLORS)]
            pose = poses[t.scene['sources'][source]['objectIndex']]
            mask = self.plane_mask(source, pose)
            tint = Image.new('RGBA', area.size, color)
            tint.putalpha(mask.point(lambda a: round(a * ((.16 + .28 * level) if active else .025))))
            area = Image.alpha_composite(area, tint)
        md = ImageDraw.Draw(area)
        for i, source in enumerate(t.sources):
            active, level = activity[source]
            if source in t.planes:
                continue
            pose = poses[t.scene['sources'][source]['objectIndex']]
            x, y = self.point(pose[:3])
            color = self.COLORS[i % len(self.COLORS)] if active else (133, 148, 155)
            if active:
                radius = 14 + 12 * ((seconds * 1.3) % 1) + 3 * level
                md.ellipse((x-radius, y-radius, x+radius, y+radius), outline=(*color, 170), width=2)
            md.ellipse((x-9, y-9, x+9, y+9), fill=(*color, 255), outline=(14, 23, 30), width=2)
            md.text((x, y), str(i + 1), anchor='mm', font=self.small, fill=(10, 17, 24))
        player = self.point(eye)
        side = np.array([-heading[1], heading[0]])
        cone = Image.new('RGBA', area.size)
        ImageDraw.Draw(cone).polygon([tuple(player), tuple(player+heading*38+side*26), tuple(player+heading*38-side*26)], fill=(245, 250, 255, 32))
        area = Image.alpha_composite(area, cone)
        md = ImageDraw.Draw(area)
        md.polygon([tuple(player+heading*9), tuple(player-heading*6+side*6), tuple(player-heading*6-side*6)], fill='white', outline=(20, 30, 38))
        md.text((312, 14), 'N', anchor='mm', font=self.small, fill='white')
        md.line((312, 38, 312, 23), fill='white', width=2)
        scale = 10 / self.map['span'] * self.size
        md.line((12, 312, 12+scale, 312), fill='white', width=2)
        md.text((12, 292), '10 m', font=self.small, fill='white')
        panel.alpha_composite(area, (24, 50))
        draw = ImageDraw.Draw(panel)
        for i, source in enumerate(t.sources):
            y = 396 + i * 34
            active, level = activity[source]
            color = self.COLORS[i % len(self.COLORS)] if active else (110, 129, 139)
            obj = t.scene['sources'][source]['objectIndex']
            label = t.scene['objects'][obj]['name'].capitalize()
            draw.ellipse((24, y, 42, y+18), fill=color)
            draw.text((33, y+9), str(i+1), anchor='mm', font=self.small, fill=(9, 18, 26))
            draw.text((51, y-1), label, font=self.font, fill='white' if active else (158, 175, 185))
            distance = 'AREA' if source in t.planes else f'{np.linalg.norm(poses[obj,:3]-eye):.0f} m'
            draw.text((354, y), distance, anchor='ra', font=self.small, fill=color)
            draw.rounded_rectangle((51, y+21, 354, y+24), 1, fill=(38, 54, 65))
            if active:
                draw.rectangle((51, y+21, 51+max(3, 303*level), y+24), fill=color)
        y = 398 + 34 * len(t.sources)
        draw.text((24, y), 'White arrow: you  /  Glow: emitting', font=self.small, fill=(191, 209, 219))
        draw.text((24, y+18), 'Sound arrives later; echoes may linger.', font=self.small, fill=(134, 158, 173))
        scale = frame.height / 1080
        panel = panel.resize((round(panel.width * scale), round(panel.height * scale)), Image.Resampling.LANCZOS)
        frame = frame.convert('RGBA')
        frame.alpha_composite(panel, (frame.width-panel.width-round(30*scale), round(30*scale)))
        return frame.convert('RGB')


def main():
    if hasattr(os, 'nice'):
        os.nice(19)  # The compositor and its encoder yield CPU priority to interactive apps.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--take', type=Path)
    parser.add_argument('--audio', type=Path, help='Rendered audio directory with listener-0.wav and run.json')
    parser.add_argument('--gain', type=float, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--hold-after', type=float, help='Hold the final captured image from this time; HUD continues updating')
    args = parser.parse_args()
    capture = args.capture
    take = TakeActivity(args.take or capture / 'take')
    overlay = SoundMap(capture, take)
    with (capture / 'view.csv').open() as file:
        views = {int(r['frame']): r for r in csv.DictReader(file)}
    frames = sorted((capture / 'frames').glob('frame_*.png'))
    if not frames:
        raise ValueError('No frames captured.')
    previews = capture / 'previews'
    previews.mkdir(exist_ok=True)
    if args.preview_only:
        for path in frames:
            index = int(path.stem.split('_')[1])
            with Image.open(path) as image:
                overlay.draw(image, float(views[index]['seconds']), views[index]).save(previews / path.name)
        return
    if not args.output or not args.audio or not math.isfinite(args.gain) or args.gain <= 0:
        raise ValueError('Encoding needs --output, --audio and a positive finite gain.')
    if [int(p.stem.split('_')[1]) for p in frames] != list(range(len(frames))):
        raise ValueError('Full movie requires contiguous frames from zero.')
    summary = json.loads((capture / 'summary.json').read_text())
    frame_count = round(summary['seconds'] * 30)
    hold = frame_count - 1 if args.hold_after is None else round(args.hold_after * 30)
    if not 0 <= hold < frame_count or len(frames) != hold + 1 or frame_count != summary['frames']:
        raise ValueError('The complete capture must finish before encoding.')
    if set(views) != set(range(frame_count)):
        raise ValueError('View poses must cover the whole movie, including its held ending.')
    for index in range(hold + 1, frame_count):
        if any(abs(float(views[index][k]) - float(views[hold][k])) > 1e-5
               for k in ('x', 'y', 'z', 'forwardX', 'forwardY', 'forwardZ')):
            raise ValueError('A held first-person image requires a stationary camera.')
    render = json.loads((args.audio / 'run.json').read_text())
    for rendered, captured in [('sceneHash', 'sceneHash'), ('takeStateHash', 'stateHash'), ('takeEventHash', 'eventHash')]:
        if render[rendered] != take.meta[captured]:
            raise ValueError('Audio does not match this take.')
    if not render['complete']:
        raise ValueError('Incomplete audio render.')
    duration = frame_count / 30
    with Image.open(frames[0]) as image:
        width, height = image.size
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required.')
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '30', '-i', 'pipe:0',
               '-i', str(args.audio / 'listener-0.wav'), '-map', '0:v:0', '-map', '1:a:0',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '1',
               '-c:a', 'aac', '-b:a', '192k', '-af',
               f"atrim=start_sample={render['dspLatencySamples']},asetpts=PTS-STARTPTS,volume={args.gain},apad=whole_dur={duration},atrim=duration={duration}",
               '-t', str(duration), '-movflags', '+faststart', str(args.output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            held_image = None
            for index in range(frame_count):
                path = frames[min(index, hold)]
                view = views[index]
                if abs(float(view['seconds']) - index/30) > 1e-7 or float(view['headDistance']) > 1e-5:
                    raise ValueError('View is not a timed first-person frame.')
                if index <= hold:
                    with Image.open(path) as image:
                        source_image = image.convert('RGB')
                    if index == hold:
                        held_image = source_image
                else:
                    source_image = held_image
                result = overlay.draw(source_image, index / 30, view)
                process.stdin.write(result.tobytes())
                if index in [0, 240, 585, 900, 1230, 1355, 1805]:
                    result.save(previews / f'frame_{index:04}.png')
                if index % 300 == 0:
                    print(f'Composited {index} / {frame_count} frames', flush=True)
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed.')
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
