"""The Church speech walk with the Path method, beside the energy render and the static-IR baseline.

    PY=.context/fdtd-teacher/venv/bin/python
    B=docs/benchmarks/path-denoise-2026-11-12
    nice -n 19 $PY $B/church.py render path-realtime     # guarded Unity renders of the walk
    nice -n 19 $PY $B/church.py render path-clean
    nice -n 19 $PY $B/church.py irs                      # static impulse responses near/far for T30/EDT/C80 (realism.py)
    $PY $B/church.py metrics                             # T30 / EDT / C80 / level of the static IRs
    $PY $B/church.py levels                              # level of each walk render per segment
    $PY $B/church.py page                                # .context/energy-audio/listen/path.html and the live view

The walk is .context/energy-audio/church/speech-walk/take (the energy listening render's take). Both path
renders use the same DSP as the energy render (spectral synthesis, FFT 1,024, 1,024-sample updates,
binaural), a 6 s horizon (the Church rings 5-8 s; no tail extrapolation) and differ only in rays.
"""
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENERGY = ROOT / 'docs/benchmarks/energy-audio-2026-11-09'
sys.path.insert(0, str(ENERGY))
import listen  # noqa: E402

WORK = ROOT / '.context/energy-audio/listen'
LIVE = Path('/Users/clarkpeng/conductor/workspaces/realisticaudioraytracing2d/tyler/.context/live-view')
REALTIME_RAYS, CLEAN_RAYS = 512, 8192
COMMON = dict(listen.COMMON, maximumResponsesPerBatch=4, maximumDelaySeconds=6.0, tailSamples=6 * 44100, audioBackend=1, method=3,
              methodRouletteFloor=1e-6, methodEdgeProbability=.5, specularDetectionRadius=2.0)  # see README part 3
RENDERS = {'path-realtime': dict(COMMON, methodRays=REALTIME_RAYS, pathHistogramExport=str(WORK / 'path-realtime-histograms')),
           'path-clean': dict(COMMON, methodRays=CLEAN_RAYS),
           # the realtime render's histograms through the denoiser (tools/path_denoise.py church)
           'path-denoised': dict(COMMON, methodRays=REALTIME_RAYS, pathHistogramImport=str(WORK / 'path-denoised-histograms')),
           # one second of the walk at 256 rays: the first end-to-end run of the method's audio
           'path-smoke': dict(COMMON, methodRays=256, renderSeconds=1.0, tailSamples=44100)}


def render(name):
    output = WORK / name
    if (output / 'run.json').exists() and (output / 'listener-0.wav').stat().st_size > 0:
        raise SystemExit(f'{output} exists; remove it to render again.')
    job = dict(take=str(listen.TAKE), output=str(output), shader='Assets/Shaders/Transport.compute', settings=RENDERS[name], workFraction=.5)
    path = WORK / f'{name}.job.json'
    path.write_text(json.dumps(job, indent=1) + '\n')
    logs, resources = listen.launch(f'listen-{name}', ['-executeMethod', 'TakeBatch.Render', '-takeJob', str(path)])
    (output / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    if resources['exitCode'] != 0:
        raise SystemExit(f'Render failed; see {logs}/unity.log')


def irs():
    """Static impulse responses of the clean path settings at realism.py's two points (doors closed)."""
    import realism
    for point in ('near', 'far'):
        output = realism.OUT / 'renders' / f'{point}-path'
        if (output / 'run.json').exists():
            continue
        settings = dict(RENDERS['path-clean'], responseStepSamples=16384, tailSamples=int(17.5 * 44100))
        job = dict(take=str(realism.OUT / 'takes' / point), output=str(output), shader='Assets/Shaders/Transport.compute', settings=settings, workFraction=.5)
        path = realism.OUT / 'renders' / f'{point}-path.job.json'
        path.write_text(json.dumps(job, indent=1) + '\n')
        resources = realism.guarded(f'realism-{point}-path', ['-executeMethod', 'TakeBatch.Render', '-takeJob', str(path)])
        (output / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')


def metrics():
    """T30 / EDT / C80 per band (realism.summary) and the broadband level (dB, energy of the whole IR) of
    the path IRs, beside the energy method's and the baselines' IRs at the same points."""
    import realism
    out = {}
    for name, (title, distance, h) in realism.load_irs().items():
        m = realism.summary(h)
        m['level'] = float(10 * math.log10((h ** 2).sum() / h.shape[1]))
        out[name] = m
    return out


SEGMENTS = (('standing and turning', 0, 5), ('walking to the organ and back', 5, 13), ('doors open, outside', 13, 18))


def levels():
    """Level (dBFS, both ears, raw render gain) of each walk render per segment of the walk."""
    out = {}
    for name in ('energy', 'path-realtime', 'path-clean', 'path-denoised'):
        folder = WORK / name
        if not (folder / 'run.json').exists() or not json.loads((folder / 'run.json').read_text()).get('complete'):
            continue
        pcm, rate = listen.read_wav(folder / 'listener-0.wav')
        pcm = pcm[int(json.loads((folder / 'run.json').read_text()).get('dspLatencySamples', 0)):].astype(np.float64)
        out[name] = {title: float(10 * math.log10(max((pcm[int(a * rate):int(b * rate)] ** 2).mean(), 1e-30)))
                     for title, a, b in SEGMENTS}
    return out


def page():
    tracks, rate = {}, None
    names = [('path-realtime', f'path, {REALTIME_RAYS} rays (realtime budget, raw)'), ('path-clean', f'path, {CLEAN_RAYS} rays (clean)'),
             ('path-denoised', f'path, {REALTIME_RAYS} rays through the denoiser'), ('energy', 'energy method (11 November)')]
    for name, _ in names:
        folder = WORK / name
        if not (folder / 'listener-0.wav').exists() or not json.loads((folder / 'run.json').read_text()).get('complete'):
            continue   # missing or still rendering
        pcm, rate = listen.read_wav(folder / 'listener-0.wav')
        report = json.loads((folder / 'run.json').read_text())
        pcm = pcm[int(report.get('dspLatencySamples', 0)):]
        fade = np.minimum(1, (len(pcm) - np.arange(len(pcm))) / rate)
        tracks[name] = pcm * (.5 - .5 * np.cos(np.pi * fade))[:, None]
    base = ROOT / '.context/energy-audio/realism/clips/walk-baseline.wav'
    # One gain with the existing listening page: its energy.wav is the energy render at that page's gain.
    energy_page = listen.read_wav(WORK / 'energy.wav')[0] / 32768.0   # 16-bit page file (read_wav returns raw integers)
    energy_raw, _ = listen.read_wav(WORK / 'energy' / 'listener-0.wav')
    latency = int(json.loads((WORK / 'energy' / 'run.json').read_text()).get('dspLatencySamples', 0))
    n = min(len(energy_page), len(energy_raw) - latency)
    gain = math.sqrt((energy_page[:n] ** 2).mean() / (energy_raw[latency:latency + n] ** 2).mean())
    tracks.pop('energy', None)
    for name, pcm in tracks.items():
        listen.write_wav16(WORK / f'{name}.wav', pcm * gain, rate)
    players = []
    for name, title in names:
        if name in tracks or name == 'energy':
            players.append(f'<h3>{title}</h3><audio controls preload="auto" src="{name}.wav" style="width:100%"></audio>')
    if base.exists():
        shutil.copy(base, WORK / 'baseline.wav')
        players.append('<h3>static-IR baseline (York Minster response, direct sound at the true distance)</h3>'
                       '<audio controls preload="auto" src="baseline.wav" style="width:100%"></audio>')
    rows = ''.join(f"<tr><td>{n}</td><td>{json.loads((WORK / n / 'run.json').read_text()).get('wallSeconds', 0):.0f} s</td></tr>" for n in list(tracks) + ['energy'])
    html = f"""<!doctype html><meta charset="utf-8"><title>Church walk: path tracer</title>
<style>body{{font:15px/1.5 -apple-system,sans-serif;max-width:980px;margin:2em auto;padding:0 1em;color:#222}}td,th{{padding:2px 10px;text-align:left}}</style>
<h1>Church walk: the path tracer, the energy method and a static IR</h1>
<p>The organ plays synthetic speech; the listener stands, turns the head, walks to the organ and back, opens the doors and walks outside. Binaural: use headphones.</p>
<p>The renders share the playback gain of the earlier listening page ({20 * math.log10(gain):+.1f} dB) so their levels compare; the baseline is the York Minster response used as a static reverb, loudness-matched as on the realism page.
<b>path, {REALTIME_RAYS} rays</b> is the raw estimate at a realtime budget, a new seed every 23 ms: its noise is expected (the denoiser's job).
<b>path, {CLEAN_RAYS} rays</b> is the same code with {CLEAN_RAYS // REALTIME_RAYS} times the rays.</p>
{''.join(players)}
<table><tr><th>render</th><th>wall time</th></tr>{rows}</table>
<p>Results and method: <code>docs/benchmarks/path-denoise-2026-11-12/README.md</code>.</p>"""
    (WORK / 'path.html').write_text(html)
    print(WORK / 'path.html')
    # The live view plays the same files.
    for name in tracks:
        target = LIVE / f'{name}.wav'
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(WORK / f'{name}.wav')


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'render':
        render(sys.argv[2])
    elif command == 'irs':
        irs()
    elif command == 'metrics':
        print(json.dumps(metrics(), indent=1))
    elif command == 'levels':
        print(json.dumps(levels(), indent=1))
    elif command == 'page':
        page()
    else:
        raise SystemExit(__doc__)
