"""Listening renders of the Church walk with the energy and hybrid methods, and their page.

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY docs/benchmarks/energy-audio-2026-11-09/listen.py render energy   # guarded Unity render
    nice -n 19 $PY docs/benchmarks/energy-audio-2026-11-09/listen.py render hybrid
    $PY docs/benchmarks/energy-audio-2026-11-09/listen.py page                       # .context/energy-audio/listen/index.html

The take is the Church speech walk (.context/church-motion/speech-full/take): the organ
plays synthetic speech while the listener stands, turns, walks to the organ and back,
opens the doors and walks out. Both renders use the same take, DSP and binaural output;
only the acoustic method differs. Unity runs go through tools/dlss_jobs.py and
tools/heavy_guard.py (isolated runner, nice 19, one job worker, machine-wide lock,
4 GiB footprint guard, 50% pacing).
"""
import base64
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_jobs as dj  # noqa: E402

# The speech walk re-captured with the published materials and interior proxies (church.py capture);
# the 15 October capture before it.
TAKE_BEFORE = ROOT / '.context/church-motion/speech-full/take'
TAKE = ROOT / '.context/energy-audio/church/speech-walk/take'
if not (TAKE / 'take.json').exists():
    TAKE = TAKE_BEFORE
BEFORE = ROOT / '.context/energy-audio/listen/before-2026-11-09'   # the 9 November renders (old materials, old edge sampling)
WORK = ROOT / '.context/energy-audio/listen'
LOGS = ROOT / '.context/energy-audio/logs'
# Shared by both renders: the Church walk's spectral DSP, binaural, 2 s horizon, air absorption.
COMMON = dict(responseStepSamples=1024, speedOfSound=343.0, referenceDistance=1.0, minimumDistance=.1, tailSamples=3 * 44100,
              transitionSamples=512, delayFilter=1, listenerOutput=1, sincRadius=16, maximumDelaySeconds=2.0, maximumSignalEvents=4096,
              maximumPendingObservations=16, maximumResponsesPerBatch=8, useBvh=True, transport=1, synthesis=1, audioBackend=0,
              seed=0, diffractionFactor=1.0, airAbsorption=True, fftSize=1024, spectralHop=256, maxBounces=64)
METHODS = {
    # Hybrid as the Church preset: first images (128 candidates), late energy field, no diffraction
    # (complete edge routes enumerate primitive pairs, out of reach for 12,761 triangles).
    'hybrid': dict(COMMON, method=0, raysPerBin=2, reflectionCandidates=128, completeRouteSamples=0, diffractionFactor=0.0,
                   lateRays=512, lateReceiverRadius=.5),
    # Energy method: exact discrete arrivals + the energy walk as the late field.
    # The diffuse field's receiver sphere is 2 m: in this 31,000 m^3 church a 0.5 m sphere is crossed
    # about 35 times per solve over 2 s by 2,048 rays, too few for a smooth 23 ms envelope; 2 m gives
    # 16x the crossings (time smear <= 12 ms, below the late field's 23 ms bins).
    # The late field continues its fitted decay to 16 s (LateTail: the model's reverberation is
    # 9-13 s, so a 2 s horizon cut it off while still audible), and is the moving average of
    # solves with new walk seeds over 1 s (smooths the walk's sampling noise over about 1.4 m
    # of walking).
    # Second-order reflections of the 96 planes most important to the source and listener each
    # solve (the Church has 6,869 planes, too many to pair all of them), 96 rendered arrival slots.
    # The walk renders 6 s after its end (the realism impulse responses keep the full 17.5 s).
    'energy': dict(COMMON, method=2, methodRays=2048, methodArrivalSlots=96, methodArrivalOrder=2, methodRouletteFloor=1e-6,
                   specularDetectionRadius=2.0, lateTailSeconds=16.0, lateAccumulationSeconds=1.0, tailSamples=6 * 44100,
                   methodNearPlanes=96),
}
# Diagnostic variants of the energy render (the coordinator's "echo doesn't sound natural" check).
# They need the uncommitted methodDiagnosticVariant knob (.context/energy-audio/diagnostic-variants.patch):
# 1 = diffuse-only (only the direct sound is a discrete arrival), 2 = arrivals-only (no diffuse field).
# The scattered-arrivals variant is the unchanged energy settings: arrivals already carry only the
# specular share (1-a)(1-t)(1-s) and the walk holds the scattered share (see render_inspection.py).
VARIANTS = {'energy-diffuse-only': dict(METHODS['energy'], methodDiagnosticVariant=1),
            'energy-arrivals-only': dict(METHODS['energy'], methodDiagnosticVariant=2),
            'energy-scattered-arrivals': dict(METHODS['energy'])}
METHODS.update(VARIANTS)


def launch(name, arguments, work_fraction=.5):
    logs = LOGS / time.strftime(f'{name}-%Y%m%d-%H%M%S')
    logs.mkdir(parents=True)
    dj.wait_for_mains()
    lock = dj.RUNNER / 'Temp/UnityLockfile'
    if lock.exists() and str(dj.RUNNER) not in os.popen('ps -A -o command=').read():
        lock.unlink()
    (logs / 'provenance.json').write_text(json.dumps(dict(files=dj.sync_runner(dj.RUNNER)), indent=1) + '\n')
    command = ['nice', '-n', '19', str(dj.UNITY), '-batchmode', '-force-metal', '-disable-audio', '-job-worker-count', '1',
               '-projectPath', str(dj.RUNNER)] + arguments + ['-logFile', str(logs / 'unity.log')]
    environment = dict(os.environ, ACOUSTIC_TEST_WORK_FRACTION=str(work_fraction), ACOUSTIC_SOLID_PYTHON=str(dj.SOLID_PYTHON))
    print(f'{logs.name}: {" ".join(arguments)}', flush=True)
    resources = dj.Worker(command, environment, lambda m: print(m, flush=True)).run()
    (logs / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    print(f'{logs.name}: exit {resources["exitCode"]}, {resources["wallSeconds"]:.0f} s, '
          f'peak footprint {resources["peakTreeFootprintBytes"] / dj.GIB:.2f} GiB', flush=True)
    return logs, resources


def render(method):
    output = WORK / method
    if (output / 'run.json').exists():
        raise SystemExit(f'{output} exists; remove it to render again.')
    job = dict(take=str(TAKE), output=str(output), shader='Assets/Shaders/Transport.compute', settings=METHODS[method], workFraction=.5)
    path = WORK / f'{method}.job.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=1) + '\n')
    logs, resources = launch(f'listen-{method}', ['-executeMethod', 'TakeBatch.Render', '-takeJob', str(path)])
    (output / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    if resources['exitCode'] != 0:
        raise SystemExit(f'Render failed; see {logs}/unity.log')


# ------------------------------------------------------------------ page

def read_wav(path):
    """Float WAV (format 3) written by TakeRender."""
    data = path.read_bytes()
    channels, rate, bits = int.from_bytes(data[22:24], 'little'), int.from_bytes(data[24:28], 'little'), int.from_bytes(data[34:36], 'little')
    at = 12
    while data[at:at + 4] != b'data':
        at += 8 + int.from_bytes(data[at + 4:at + 8], 'little')
    size = int.from_bytes(data[at + 4:at + 8], 'little')
    pcm = np.frombuffer(data[at + 8:at + 8 + size], dtype='<f4' if bits == 32 else '<i2').astype(np.float64)
    return pcm.reshape(-1, channels), rate


def write_wav16(path, pcm, rate):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(pcm.shape[1]); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes())


def take_scene():
    meta = json.loads((TAKE / 'take.json').read_text())
    scene = meta['scene']
    geometry = np.frombuffer((TAKE / 'geometry.bin').read_bytes(), dtype=[('vertices', '<f4', (3, 3)), ('owner', '<i4'), ('material', '<i4')])
    frames = np.fromfile(TAKE / 'states.bin', dtype=np.dtype([('sample', '<i8'), ('state', '<f4', (len(scene['objects']), 20))]))
    return meta, scene, geometry, frames


def rotate(q, v):
    q = np.asarray(q, float)
    return v + 2 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)


def scene_map(path):
    """Top-down cut of the Church at ear height: walls, source, listener route with times."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta, scene, geometry, frames = take_scene()
    rate = meta['sampleRate']
    listener, source = scene['listeners'][0], scene['sources'][0]['objectIndex']
    pose = frames['state'][0]
    route = frames['state'][:, listener, :3]
    height = float(np.median(route[:, 1]))
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    for owner in np.unique(geometry['owner']):
        tri = geometry['vertices'][geometry['owner'] == owner].astype(float)
        p = pose[owner]
        world = p[:3] + rotate(p[3:7], tri.reshape(-1, 3)).reshape(tri.shape)
        name = scene['objects'][owner]['name']
        if name in ('Meadow', 'CopperRoof', 'StoneRoof'):
            continue
        y = world[:, :, 1] - height
        segments = []
        for t, d in zip(world, y):
            points = []
            for i in range(3):
                a, b, da, db = t[i], t[(i + 1) % 3], d[i], d[(i + 1) % 3]
                if (da <= 0) != (db <= 0):
                    points.append(a + (b - a) * (da / (da - db)))
            if len(points) == 2:
                segments.append(points)
        if segments:
            s = np.array(segments)
            door = 'Door' in name
            ax.plot(s[:, :, 0].T, s[:, :, 2].T, color='#B22222' if door else '#555555', lw=1.4 if door else .8)
    ax.plot(route[:, 0], route[:, 2], color='#0072B2', lw=1.6, label='listener route')
    seconds = frames['sample'] / rate
    for t, label in [(0, '0-5 s: stand, turn head'), (7, '7 s: nearest the organ'), (10.5, '10-11.6 s: doors open'),
                     (13.6, '13.6 s: outside'), (18, '18 s: end')]:
        i = int(np.argmin(abs(seconds - t)))
        ax.plot(route[i, 0], route[i, 2], 'o', color='#0072B2', ms=4)
        ax.annotate(label, (route[i, 0], route[i, 2]), textcoords='offset points', xytext=(7, 4 if t != 13.6 else -12), fontsize=8, color='#0072B2')
    s = frames['state'][0, source, :3]
    ax.plot(s[0], s[2], '*', color='#D55E00', ms=16, label='source: organ, synthetic speech')
    ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('z (m)')
    ax.set_title(f'Church walk, cut at ear height ({height:.1f} m); doors in red', fontsize=10)
    ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def level_figure(path, tracks, rate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'hybrid': '#0072B2', 'energy': '#009E73', 'energy-before': '#999999'}
    fig, ax = plt.subplots(figsize=(10, 3.2))
    block = rate // 20
    for name, pcm in tracks.items():
        mono = (pcm ** 2).mean(1)
        n = len(mono) // block
        level = 10 * np.log10(np.maximum(mono[:n * block].reshape(n, block).mean(1), 1e-12))
        ax.plot(np.arange(n) * block / rate, level, color=colors.get(name, '#999999'), lw=.8 if name.endswith('before') else 1, label=name)
    for t, label in [(3, 'head turns'), (5, 'walk to organ'), (7, 'walk back'), (10, 'doors open'), (11.6, 'outside'), (18, 'speech ends')]:
        ax.axvline(t, color='#999999', lw=.6); ax.text(t + .1, ax.get_ylim()[1] - 6, label, fontsize=7, color='#666666')
    ax.set_xlabel('time (s)'); ax.set_ylabel('level, 50 ms (dB)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def page():
    WORK.mkdir(parents=True, exist_ok=True)
    scene_map(WORK / 'scene-map.png')
    tracks, reports = {}, {}
    rate = None
    for name in ('energy', 'hybrid', 'energy-before', 'hybrid-before', *VARIANTS):
        folder = BEFORE / name.replace('-before', '') if name.endswith('-before') else WORK / name
        if not (folder / 'listener-0.wav').exists():
            continue
        pcm, rate = read_wav(folder / 'listener-0.wav')
        report = json.loads((folder / 'run.json').read_text())
        latency = int(report.get('dspLatencySamples', 0))
        pcm = pcm[latency:]
        # The walk renders stop a few seconds after its end: fade the last second (no click).
        fade = np.minimum(1, (len(pcm) - np.arange(len(pcm))) / rate)
        tracks[name] = pcm * (.5 - .5 * np.cos(np.pi * fade))[:, None]
        reports[name] = report
    if not tracks:
        raise SystemExit('No renders yet.')
    peak = max(np.abs(tracks[n]).max() for n in ('energy', 'hybrid', 'energy-before', 'hybrid-before') if n in tracks)  # all tracks share this gain
    gain = .5 / peak
    for name, pcm in tracks.items():
        write_wav16(WORK / f'{name}.wav', pcm * gain, rate)
    level_figure(WORK / 'levels.png', {n: tracks[n] for n in ('energy-before', 'energy', 'hybrid') if n in tracks}, rate)
    rows = []
    for name, r in reports.items():
        if name in VARIANTS or name.endswith('-before'):
            continue
        rms = 20 * math.log10(math.sqrt((tracks[name] ** 2).mean()) * gain)
        audio = r.get('gpuAudio') or {}
        rows.append(f"<tr><td>{name}</td><td>{r.get('wallSeconds', 0):.0f} s</td><td>{audio.get('peakGroups', 0)}</td><td>{rms:.1f} dBFS</td></tr>")
    players = '\n'.join(f'<h3>{name}</h3><audio controls preload="auto" src="{name}.wav" style="width:100%"></audio>' for name in ('energy', 'hybrid') if name in tracks)
    players += '<h3>Before (9 November: old materials, old edge sampling)</h3>' + '\n'.join(
        f'<h4>{name}</h4><audio controls preload="auto" src="{name}.wav" style="width:100%"></audio>' for name in ('energy-before', 'hybrid-before') if name in tracks)
    variants = '\n'.join(f'<h4>{name}</h4><audio controls preload="auto" src="{name}.wav" style="width:100%"></audio>' for name in VARIANTS if name in tracks)
    if variants:
        players += ('<h3>Diagnostic variants of the energy render (same gain)</h3><p>diffuse-only: only the direct sound is discrete, everything else is the diffuse field; '
                    'arrivals-only: direct sound and discrete arrivals, no diffuse field; scattered-arrivals: arrivals at their specular share (already the case), '
                    'a fresh render of the energy settings. Findings: <code>.context/energy-audio/inspect/report.md</code>.</p>' + variants)
    html = f"""<!doctype html><meta charset="utf-8"><title>Energy vs hybrid: Church walk</title>
<style>body{{font:15px/1.5 -apple-system,sans-serif;max-width:980px;margin:2em auto;padding:0 1em;color:#222}}img{{max-width:100%}}
td,th{{padding:2px 10px;text-align:left}}.wrap{{display:flex;gap:24px;flex-wrap:wrap}}.wrap>div{{flex:1 1 420px}}</style>
<h1>Church walk: energy method vs hybrid</h1>
<p>The organ (orange star) plays synthetic speech. The listener (blue route) stands for 3 s, turns the head (3–5 s), walks towards the organ (5–7 s) and back (7–10 s), opens the doors (10 s) and walks outside (11.6–18 s). Headphones: the renders are binaural.</p>
<p>Both renders use the same take, DSP, horizon (2 s) and one shared playback gain ({20 * math.log10(gain):+.1f} dB), so their levels compare directly; the DSP latency is trimmed.
<b>energy</b>: exact discrete arrivals (direct sound, reflections, edge diffraction) heard as crisp delayed paths with Doppler, plus the energy walk as the smooth reverberant tail.
<b>hybrid</b>: the current default as the Church preset uses it (first-order images and the late energy field; no edge diffraction).</p>
<div class="wrap"><div><img src="scene-map.png" alt="Scene map"></div><div>{players}
<p><img src="levels.png" alt="Levels"></p>
<table><tr><th>method</th><th>render wall time*</th><th>most delayed paths at once</th><th>RMS</th></tr>{''.join(rows)}</table></div></div>
<p>* {len(next(iter(tracks.values()))) / rate:.0f} s of audio, offline on the M5 GPU with 50% work pacing (so about half is idle); transport and audio included.</p>
<p>What differs in this scene: the Church has 12,761 triangles, 4,864 mirror planes and 10,249 diffracting edges, so the energy method enumerates the direct sound, every first-order reflection and every single-edge diffraction path exactly (second-order and two-edge paths stay in its diffuse tail).
The hybrid samples 128 of the 12,761 triangles as first-order mirror candidates per solve; in this render it never held more than one delayed path at a time, i.e. almost always the direct sound alone, with its late field.</p>
<p>Renders: <code>.context/energy-audio/listen/&lt;method&gt;/</code> (float WAV, run.json with every setting). Reproduce with <code>docs/benchmarks/energy-audio-2026-11-09/listen.py</code>.</p>
"""
    (WORK / 'index.html').write_text(html)
    print(WORK / 'index.html')


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'render':
        render(sys.argv[2])
    elif command == 'page':
        page()
    else:
        raise SystemExit(__doc__)
