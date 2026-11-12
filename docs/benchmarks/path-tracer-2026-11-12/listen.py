"""Listening renders of the Church walk with the Path method: realtime rays (noisy) and clean rays.

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY docs/benchmarks/path-tracer-2026-11-12/listen.py render realtime   # guarded Unity render
    nice -n 19 $PY docs/benchmarks/path-tracer-2026-11-12/listen.py render clean
    $PY docs/benchmarks/path-tracer-2026-11-12/listen.py page                         # .context/path-tracer/listen/index.html

Both renders use the same take excerpt (the first EXCERPT seconds of the Church speech walk:
standing and turning the head, then the first steps), the same DSP and binaural output; only
the ray count differs. The realtime render is the raw estimator at a realtime budget (a new seed
every update, no averaging): it is expected to sound noisy, which is what the denoiser is for.
Unity runs go through tools/dlss_jobs.py and tools/heavy_guard.py.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'docs/benchmarks/energy-audio-2026-11-09'))
import listen as energy_listen  # noqa: E402  (take, launch, WAV helpers, scene map)

WORK = ROOT / '.context/path-tracer/listen'
EXCERPT, TAIL = 6.0, 3.0
COMMON = dict(energy_listen.COMMON, maximumResponsesPerBatch=4, maximumDelaySeconds=5.0, tailSamples=int(TAIL * 44100), renderSeconds=EXCERPT,
              audioBackend=1, method=3, methodRouletteFloor=1e-6, methodEdgeProbability=.5, specularDetectionRadius=.25)
# Clean: 8x the rays, updated every 4,096 samples (93 ms) instead of 1,024 to bound the render's cost.
RENDERS = {'realtime': dict(COMMON, methodRays=2048), 'clean': dict(COMMON, methodRays=16384, responseStepSamples=4096)}


def render(name):
    output = WORK / name
    if (output / 'run.json').exists():
        raise SystemExit(f'{output} exists; remove it to render again.')
    job = dict(take=str(energy_listen.TAKE), output=str(output), shader='Assets/Shaders/Transport.compute', settings=RENDERS[name], workFraction=.5)
    path = WORK / f'{name}.job.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=1) + '\n')
    energy_listen.LOGS = ROOT / '.context/path-tracer/logs'
    logs, resources = energy_listen.launch(f'path-listen-{name}', ['-executeMethod', 'TakeBatch.Render', '-takeJob', str(path)])
    (output / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    if resources['exitCode'] != 0:
        raise SystemExit(f'Render failed; see {logs}/unity.log')


def page():
    WORK.mkdir(parents=True, exist_ok=True)
    energy_listen.WORK = WORK
    energy_listen.scene_map(WORK / 'scene-map.png')
    tracks, reports, rate = {}, {}, None
    for name in RENDERS:
        folder = WORK / name
        if not (folder / 'listener-0.wav').exists():
            continue
        pcm, rate = energy_listen.read_wav(folder / 'listener-0.wav')
        report = json.loads((folder / 'run.json').read_text())
        pcm = pcm[int(report.get('dspLatencySamples', 0)):]
        fade = np.minimum(1, (len(pcm) - np.arange(len(pcm))) / rate)
        tracks[name] = pcm * (.5 - .5 * np.cos(np.pi * fade))[:, None]
        reports[name] = report
    if not tracks:
        raise SystemExit('No renders yet.')
    gain = .5 / max(np.abs(t).max() for t in tracks.values())
    for name, pcm in tracks.items():
        energy_listen.write_wav16(WORK / f'{name}.wav', pcm * gain, rate)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.2))
    block = rate // 20
    for name, pcm in tracks.items():
        mono = (pcm ** 2).mean(1); n = len(mono) // block
        ax.plot(np.arange(n) * block / rate, 10 * np.log10(np.maximum(mono[:n * block].reshape(n, block).mean(1), 1e-12)),
                color={'realtime': '#E69F00', 'clean': '#56B4E9'}[name], lw=1, label=name)
    ax.set_xlabel('time (s)'); ax.set_ylabel('level, 50 ms (dB)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(WORK / 'levels.png', dpi=120); plt.close(fig)
    rows = ''.join(f"<tr><td>{n}</td><td>{RENDERS[n]['methodRays']}</td><td>{r.get('wallSeconds', 0):.0f} s</td>"
                   f"<td>{r.get('solveSeconds', 0):.0f} s</td><td>{r.get('audioSeconds', 0):.0f} s</td></tr>" for n, r in reports.items())
    players = '\n'.join(f'<h3>{n} ({RENDERS[n]["methodRays"]} rays per update)</h3><audio controls preload="auto" src="{n}.wav" style="width:100%"></audio>'
                        for n in RENDERS if n in tracks)
    html = f"""<!doctype html><meta charset="utf-8"><title>Path tracer: Church walk</title>
<style>body{{font:15px/1.5 -apple-system,sans-serif;max-width:980px;margin:2em auto;padding:0 1em;color:#222}}img{{max-width:100%}}
td,th{{padding:2px 10px;text-align:left}}.wrap{{display:flex;gap:24px;flex-wrap:wrap}}.wrap>div{{flex:1 1 420px}}</style>
<h1>Church walk: the path tracer at realtime rays and at clean rays</h1>
<p>The organ plays synthetic speech; the listener stands and turns the head (3-5 s), then starts walking (from 5 s). The first {EXCERPT:.0f} s of the walk, plus {TAIL:.0f} s of tail. Headphones: binaural.</p>
<p>One estimator, one output: every update traces the given number of rays once (a new seed each time, no averaging, no tail extrapolation) into an energy histogram, and the audio is synthesised from it (the direct sound as an exact delayed path with Doppler; everything else as random-sign impulses with the histogram's energies).
<b>realtime</b> is the raw estimate at a realtime budget: its noise is audible and is what the denoiser is meant to remove. <b>clean</b> is the same code with 8 times the rays, updated every 93 ms instead of 23 ms (to bound its cost). Both share one playback gain ({20 * math.log10(gain):+.1f} dB).</p>
<div class="wrap"><div><img src="scene-map.png" alt="Scene map"></div><div>{players}<p><img src="levels.png" alt="Levels"></p>
<table><tr><th>render</th><th>rays</th><th>wall time</th><th>transport</th><th>audio synthesis</th></tr>{rows}</table></div></div>
<p>Renders: <code>.context/path-tracer/listen/&lt;name&gt;/</code>. Reproduce with <code>docs/benchmarks/path-tracer-2026-11-12/listen.py</code>.</p>
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
