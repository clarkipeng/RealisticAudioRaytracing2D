"""Late-energy benchmark: job lists, analysis, figures, report.json and the Church listening page.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY docs/benchmarks/late-energy-2026-11-04/late.py jobs          # writes .context/late-energy/jobs/*.json
    $PY docs/benchmarks/late-energy-2026-11-04/late.py report        # figures + report.json from the measurements
    $PY docs/benchmarks/late-energy-2026-11-04/late.py church        # Church before/after statistics and listening page

Pure NumPy/SciPy/Matplotlib; Unity runs go through unity.py.
"""
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parent
WORK = ROOT / '.context/late-energy'
C = 343.0
BANDS = [125 * 2 ** b for b in range(8)]

# ------------------------------------------------------------------ job lists

ROOM = dict(size=[8, 6, 4], source=[-2.1, .7, .5])
# Receivers at 1.5-5 m from the source, away from walls (> 1 m).
ROOM_LISTENERS = [-.6, .7, .5, .6, .2, .1, 1.9, -.9, -.6, 2.6, -1.4, -.9]
STONE = dict(frequencies=BANDS, absorptionCurve=[.02, .025, .03, .04, .05, .05, .06, .07],
             scatteringCurve=[.3] * 8)


def room_jobs():
    jobs = []
    for a in (.1, .2, .3):
        jobs.append(dict(name=f'diffuse-a{int(a * 100)}', absorption=a, scattering=1, maximumDelaySeconds=2.0))
    jobs.append(dict(name='partial-s20-a20', absorption=.2, scattering=.2, maximumDelaySeconds=2.0))
    jobs.append(dict(name='stone-air', absorption=0, scattering=1, airAbsorption=True, maximumDelaySeconds=6.0, **STONE))
    for job in jobs:
        job.update(kind='room', listeners=ROOM_LISTENERS, rays=[4096], seeds=[1, 2], maxBounces=8192,
                   sampleRate=8000, fftSize=64, responseStepSamples=64, receiverRadius=.5, **ROOM)
    return jobs


def lattice_jobs():
    base = dict(kind='room', size=[5, 4, 3], source=[-1.1, .4, .3], listener=[1.3, -.6, -.2], absorption=.05, scattering=0,
                maximumDelaySeconds=.3, rays=[65536], seeds=[1], maxBounces=8192, sampleRate=8000, fftSize=64,
                responseStepSamples=64, receiverRadius=.25)
    return [dict(base, name='lattice-first-images', completeRouteSamples=0),
            dict(base, name='lattice-complete-routes', completeRouteSamples=256)]


PILOTS = {'shoebox': '.context/residual-walk/after/scenes/shoebox-0900-walk',
          'doorway': '.context/residual-walk/after/scenes/doorway-0900-walk',
          'corridor': '.context/audio-dlss-data/scenes/corridor-0000',
          'furnished': '.context/audio-dlss-data/scenes/furnished-0000'}


def convergence_jobs():
    jobs = []
    for family, folder in PILOTS.items():
        jobs.append(dict(name=f'converge-{family}', kind='pilot', sceneDirectory=str(ROOT / folder), variant='tracer',
                         rays=[128, 512, 2048, 8192], seeds=[1, 2, 3, 4], points=8, maxBounces=32, maximumDelaySeconds=1.25,
                         sampleRate=8000, fftSize=64, responseStepSamples=64, receiverRadius=.5))
    # Bounce count: the shoebox at a 1024-interaction cap (rays reach the 1.25 s horizon).
    jobs.append(dict(jobs[0], name='converge-shoebox-deep', rays=[512, 2048], maxBounces=1024))
    return jobs


def write_jobs():
    folder = WORK / 'jobs'
    folder.mkdir(parents=True, exist_ok=True)
    for name, jobs in [('rooms', room_jobs()), ('lattice', lattice_jobs()), ('convergence', convergence_jobs())]:
        for job in jobs:
            job['output'] = str(WORK / 'measure' / job['name'])
        (folder / f'{name}.json').write_text(json.dumps(dict(jobs=jobs), indent=1) + '\n')
        print(folder / f'{name}.json', len(jobs))


# ------------------------------------------------------------------ measurement readers

def load(name):
    folder = WORK / 'measure' / name
    meta = json.loads((folder / 'late.json').read_text())
    runs = {}
    for run in meta['runs']:
        values = np.fromfile(folder / run['file'], '<f4').reshape(meta['points'], meta['directions'], meta['bands'], meta['bins'])
        raw = np.fromfile(folder / run['earlyFile'], '<f4').reshape(-1, 4)
        # Coherent sum per receiver and delay (complete routes split one route over samples).
        groups = {}
        for point, delay, re, im in raw:
            key = (int(point), round(float(delay) * 1e6))
            groups[key] = groups.get(key, 0) + complex(re, im)
        early = np.array([(k[0], k[1] * 1e-6, abs(v) ** 2) for k, v in groups.items()]).reshape(-1, 3)
        runs[(run['rays'], run['seed'])] = dict(late=values.astype(np.float64), early=early, seconds=np.array(run['solveSeconds']))
    return meta, runs


def decay_times(energy, dt):
    """EDT, T20, T30 from the Schroeder integral of an energy histogram (seconds)."""
    curve = np.cumsum(energy[::-1])[::-1]
    if curve[0] <= 0:
        return dict(EDT=np.nan, T20=np.nan, T30=np.nan)
    level = 10 * np.log10(np.maximum(curve / curve[0], 1e-30))
    t = (np.arange(len(energy)) + .5) * dt
    out = {}
    for key, lo, hi, scale in [('EDT', 0, -10, 6), ('T20', -5, -25, 3), ('T30', -5, -35, 2)]:
        mask = (level <= lo) & (level >= hi)
        if key == 'EDT':
            mask = level >= hi
        if mask.sum() < 3:
            out[key] = np.nan
            continue
        slope = np.polyfit(t[mask], level[mask], 1)[0]
        out[key] = -60 / slope
    return out


# ------------------------------------------------------------------ analysis

def eyring(size, absorption, air=0.0):
    v = size[0] * size[1] * size[2]
    s = 2 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0])
    return .161 * v / (-s * math.log(1 - absorption) + 4 * air * v), .161 * v / (s * absorption + 4 * air * v), v, s


def air_energy_coefficient(hz):
    losses = [0.0001, 0.0003, 0.0007, 0.0015, 0.0035, 0.0100, 0.0320, 0.1100]
    at = min(max(math.log2(max(125, hz) / 125), 0), 7)
    a = int(at)
    db = losses[a] + (losses[min(a + 1, 7)] - losses[a]) * (at - a)
    return 2 * db / 8.685889638   # energy nepers per metre (m in Sabine's 4mV)


def diffuse_level(size, absorption, air=0.0):
    """Reflected-energy level of a diffuse field, 16*pi/A (direct-sound units at 1 m), A by Eyring."""
    _, _, v, s = eyring(size, absorption)
    return 16 * math.pi / (-s * math.log(1 - absorption) + 4 * air * v)


def physics():
    """Decay times and levels in the measured rooms against Sabine/Eyring and the diffuse level."""
    rows = []
    for job in room_jobs():
        try:
            meta, runs = load(job['name'])
        except FileNotFoundError:
            continue
        dt = meta['binSeconds']
        late = np.mean([r['late'] for r in runs.values()], 0).sum(1)          # [points, bands, bins], directions summed
        early = list(runs.values())[0]['early']
        for band in range(8):
            a = job['absorption'] if 'absorptionCurve' not in job else job['absorptionCurve'][band]
            air = air_energy_coefficient(BANDS[band]) if job.get('airAbsorption') else 0.0
            t_eyring, t_sabine, v, s = eyring(job['size'], a, air)
            energy = late[:, band].mean(0).copy()
            if band == 0:
                # The exported images are frequency bin 1 (125 Hz = band 0); add them so EDT sees the whole reflected decay.
                for point, delay, value in early:
                    k = int(delay / dt)
                    if k < len(energy):
                        energy[k] += value / meta['points']
                # ISO 3382 decay curves include the direct sound (1/r^2, no absorption).
                for p in range(meta['points']):
                    r = float(np.linalg.norm(np.array(meta['listeners'][3 * p:3 * p + 3]) - np.array(job['source'])))
                    energy[int(r / C / dt)] += 1 / r ** 2 / meta['points']
            times = decay_times(energy, dt)
            if band != 0:
                times['EDT'] = float('nan')   # Late-only above 125 Hz: EDT needs the early images too.
            # Late + first images, per receiver, against the diffuse reflected level.
            levels, barron = [], []
            for p in range(meta['points']):
                images = early[early[:, 0] == p, 2].sum()
                reflected = late[p, band].sum() + images
                levels.append(10 * math.log10(reflected / diffuse_level(job['size'], a, air)))
                # Barron & Lee revised theory: the reflected energy falls with source distance r as exp(-13.82 r / (c T)).
                r = float(np.linalg.norm(np.array(meta['listeners'][3 * p:3 * p + 3]) - np.array(job['source'])))
                barron.append(10 * math.log10(reflected / (diffuse_level(job['size'], a, air) * math.exp(-13.82 * r / (C * t_eyring)))))
            rows.append(dict(job=job['name'], band=BANDS[band], absorption=a, scattering=job['scattering'] if 'scatteringCurve' not in job else job['scatteringCurve'][band],
                             eyring=t_eyring, sabine=t_sabine, **{k: float(v_) for k, v_ in times.items()},
                             levelVsDiffuseDb=levels, levelVsBarronDb=barron, solveSeconds=float(np.median([r['seconds'].mean() for r in runs.values()]))))
            if 'absorptionCurve' not in job and not job.get('airAbsorption'):
                break   # Frequency-flat rooms: every band is identical.
    return rows


def lattice():
    """Early images + late energy against the analytic shoebox image lattice (rigid, specular)."""
    out = []
    for job in lattice_jobs():
        try:
            meta, runs = load(job['name'])
        except FileNotFoundError:
            continue
        run = list(runs.values())[0]
        dt, bins = meta['binSeconds'], meta['bins']
        size, corner = np.array(job['size']), np.array(job['size']) / 2
        source, listener = np.array(job['source']) + corner, np.array(job['listener']) + corner
        lattice_energy = image_lattice(size, source, listener, job['absorption'], dt, bins)
        late = run['late'][0].sum(0)[0]                                        # band 0, directions summed
        early = run['early']
        images = np.zeros(bins)
        for _, delay, energy in early:
            k = int(delay / dt)
            if k < bins:
                images[k] += energy
        order = meta['earlyOrder']
        expected_late = image_lattice(size, source, listener, job['absorption'], dt, bins, order + 1)
        held = image_lattice(size, source, listener, job['absorption'], dt, bins, 1, order)
        out.append(dict(job=job['name'], earlyOrder=order, binSeconds=dt,
                        lateOverLattice=float(late.sum() / expected_late.sum()),
                        imagesOverLattice=float(images.sum() / held.sum()),
                        totalOverLattice=float((late.sum() + images.sum()) / image_lattice(size, source, listener, job['absorption'], dt, bins, 1).sum()),
                        curves=dict(late=late.tolist(), images=images.tolist(), latticeLate=expected_late.tolist(), latticeEarly=held.tolist())))
    return out


def image_lattice(size, source, listener, absorption, dt, bins, minimum=0, maximum=10 ** 9):
    energy = np.zeros(bins)
    reach = int(math.ceil(bins * dt * C / min(size))) + 1
    n = np.arange(-reach, reach + 1)
    for mask in range(8):
        coords, orders = [], []
        for a in range(3):
            mirrored = (mask >> a) & 1
            coords.append(2 * n * size[a] + (-source[a] if mirrored else source[a]))
            orders.append(np.abs(2 * n - 1) if mirrored else np.abs(2 * n))
        x, y, z = np.meshgrid(*coords, indexing='ij')
        ox, oy, oz = np.meshgrid(*orders, indexing='ij')
        distance = np.sqrt((x - listener[0]) ** 2 + (y - listener[1]) ** 2 + (z - listener[2]) ** 2)
        order = ox + oy + oz
        keep = (order >= minimum) & (order <= maximum)
        k = np.floor(distance / C / dt).astype(int)
        keep &= k < bins
        np.add.at(energy, k[keep], (1 - absorption) ** order[keep] / distance[keep] ** 2)
    return energy


def convergence():
    """Seed spread of the late histogram versus rays, per closed pilot scene and time window."""
    out = []
    windows = [(0, .1), (.1, .3), (.3, .6), (.6, 1.25)]
    for job in convergence_jobs():
        try:
            meta, runs = load(job['name'])
        except FileNotFoundError:
            continue
        dt = meta['binSeconds']
        for rays in job['rays']:
            values = np.array([runs[(rays, s)]['late'].sum(1) for s in job['seeds'] if (rays, s) in runs])  # [seeds, points, bands, bins]
            mean = values.mean(0)
            var = values.var(0, ddof=1)
            spread = math.sqrt(var.sum() / (mean ** 2).sum())
            per_window = []
            for a, b in windows:
                sl = slice(int(a / dt), int(b / dt))
                m, v = mean[..., sl], var[..., sl]
                per_window.append(math.sqrt(v.sum() / (m ** 2).sum()) if (m ** 2).sum() > 0 else float('nan'))
            out.append(dict(job=job['name'], rays=rays, seeds=len(values), spread=spread, windows=per_window,
                            solveSeconds=float(np.median(np.concatenate([runs[(rays, s)]['seconds'] for s in job['seeds'] if (rays, s) in runs]))),
                            maxBounces=job['maxBounces']))
    return out, windows


# ------------------------------------------------------------------ Church before/after

CHURCH = WORK / 'church'
# Derived from the weather walkthrough's 5-21 s: outside, through the door
# (opens at 4 s, closes at 8 s), then the closed interior.
CHURCH_WINDOWS = {'outside': (0.0, 4.0), 'doorway': (4.0, 8.0), 'interior': (9.0, 16.0), 'tail': (16.0, 17.4)}


def read_wav(path):
    import wave
    with open(path, 'rb') as f:
        data = f.read()
    # 32-bit float RIFF written by TakeRender.WriteWaveHeader (44-byte header).
    channels = int.from_bytes(data[22:24], 'little')
    rate = int.from_bytes(data[24:28], 'little')
    pcm = np.frombuffer(data[44:], '<f4').reshape(-1, channels)
    return pcm, rate


def frame_levels(x, rate, seconds):
    n = int(rate * seconds)
    count = len(x) // n
    frames = x[:count * n].reshape(count, n)
    return np.sqrt((frames ** 2).mean(1)), np.abs(frames).max(1)


def church_stats(path, latency):
    pcm, rate = read_wav(path)
    x = pcm[latency:, 0].astype(np.float64)
    rms5, peak5 = frame_levels(x, rate, .005)
    out = dict(rate=rate, seconds=len(x) / rate, peak=float(np.abs(x).max()), clipped=int((np.abs(x) > 1).sum()))
    windows = {}
    for name, (a, b) in CHURCH_WINDOWS.items():
        seg = x[int(a * rate):int(b * rate)]
        if len(seg) == 0:
            continue
        rms = math.sqrt(float((seg ** 2).mean()))
        q, _ = frame_levels(seg, rate, .25)
        lo, hi = int(a / .005), int(b / .005)
        # A spike: a 5 ms frame peaking 20 dB above the median 5 ms RMS of its +-0.25 s neighbourhood.
        spikes = 0
        for i in range(lo, min(hi, len(peak5))):
            ref = np.median(rms5[max(0, i - 50):i + 50])
            spikes += int(ref > 0 and peak5[i] > 10 * ref)
        windows[name] = dict(rmsDbfs=20 * math.log10(max(rms, 1e-12)), peak=float(np.abs(seg).max()),
                             crestDb=20 * math.log10(max(np.abs(seg).max(), 1e-12) / max(rms, 1e-12)),
                             quarterSecondSpreadDb=float(np.subtract(*np.percentile(20 * np.log10(np.maximum(q, 1e-12)), [90, 10]))),
                             spikes=spikes)
    out['windows'] = windows
    return out, x, rate


def church(listen=WORK / 'listen'):
    """Church before/after statistics, figure and a static listening page under .context/."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    renders = {}
    for name in ['before-live', 'after-live']:
        folder = CHURCH / name
        if not (folder / 'listener-0.wav').exists():
            continue
        run = json.loads((folder / 'run.json').read_text())
        stats, x, rate = church_stats(folder / 'listener-0.wav', run['dspLatencySamples'])
        stats.update(method=run['method'], solveSeconds=run['solveSeconds'], audioSeconds=run['audioSeconds'],
                     responses=run['responseCount'], wallSeconds=run['wallSeconds'], numericBytes=run['workingNumericBytes'])
        renders[name] = (stats, x, rate)
    if not renders:
        return {}
    listen.mkdir(parents=True, exist_ok=True)
    peak = max(s['peak'] for s, _, _ in renders.values())
    gain = .89 / peak                              # One shared gain keeps relative levels; -1 dBFS headroom.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for name, (stats, x, rate) in renders.items():
        import wave
        with wave.open(str(listen / f'{name}.wav'), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(x * gain, -1, 1) * 32767).astype('<i2').tobytes())
        rms, pk = frame_levels(x, rate, .05)
        t = (np.arange(len(rms)) + .5) * .05
        axes[0].plot(t, 20 * np.log10(np.maximum(rms, 1e-9)), label=name)
        axes[1].plot(t, 20 * np.log10(np.maximum(pk, 1e-9)), label=name)
    for a, b in CHURCH_WINDOWS.values():
        for ax in axes:
            ax.axvspan(a, b, color='0.9', zorder=0)
    axes[0].set_ylabel('50 ms RMS (dBFS)')
    axes[1].set_ylabel('50 ms peak (dBFS)')
    axes[1].set_xlabel('time (s)')
    axes[0].legend()
    axes[0].set_title('Church weather segment at the live preset: coherent walk (before) vs energy late field (after)')
    fig.tight_layout()
    fig.savefig(BENCH / 'fig4-church.png', dpi=120)
    rows = []
    for name, (stats, _, _) in renders.items():
        for window, values in stats['windows'].items():
            rows.append(f"<tr><td>{name}</td><td>{window}</td><td>{values['rmsDbfs']:.1f}</td><td>{values['peak']:.3f}</td>"
                        f"<td>{values['crestDb']:.1f}</td><td>{values['quarterSecondSpreadDb']:.1f}</td><td>{values['spikes']}</td></tr>")
    players = ''.join(f'<h2>{name}</h2><audio controls preload="none" src="{name}.wav"></audio>' for name in renders)
    (listen / 'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Church late field: before and after</title>
<style>body{{font:15px system-ui;max-width:52rem;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse}}td,th{{padding:.2rem .6rem;border-bottom:1px solid #ddd;text-align:right}}</style>
<h1>Church weather walkthrough, 16 s, live preset</h1>
<p>Outside 0-4 s, through the door 4-8 s (it closes at 8 s), closed interior 9-16 s, then the tail.
Both files share one gain ({20 * math.log10(gain):+.1f} dB) so their levels compare directly; DSP latency is trimmed. Mono, 44.1 kHz.</p>
<p><b>before</b>: coherent residual walk (solver v13). <b>after</b>: phase-exact direct sound and first images plus the energy late field (solver v14).</p>
{players}
<h2>Statistics</h2><table><tr><th>render</th><th>window</th><th>RMS dBFS</th><th>peak</th><th>crest dB</th><th>P90-P10 0.25 s dB</th><th>spikes</th></tr>{''.join(rows)}</table>
<p>A spike is a 5 ms frame whose peak is 20 dB above the median 5 ms RMS of its surrounding half second.</p>
<img src="fig4-church.png" style="max-width:100%">''')
    shutil.copy(BENCH / 'fig4-church.png', listen / 'fig4-church.png')
    return {name: stats for name, (stats, _, _) in renders.items()}


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = dict(physics=physics(), lattice=lattice())
    conv, windows = convergence()
    result['convergence'] = conv
    result['convergenceWindows'] = windows
    result['church'] = church()
    cost = WORK / 'cost.json'
    if cost.exists():
        result['cost'] = json.loads(cost.read_text())
    tests = WORK / 'tests.json'
    if tests.exists():
        result['regression'] = json.loads(tests.read_text())
    # Figure 1: convergence.
    if conv:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for job in sorted({r['job'] for r in conv}):
            rows = [r for r in conv if r['job'] == job]
            axes[0].loglog([r['rays'] for r in rows], [100 * r['spread'] for r in rows], 'o-', label=job.replace('converge-', ''))
        rays = np.array([128, 8192])
        axes[0].loglog(rays, 100 * .5 * np.sqrt(128 / rays), 'k--', label='1/sqrt(N)')
        axes[0].set_xlabel('late rays per band')
        axes[0].set_ylabel('seed spread (% of mean histogram)')
        axes[0].legend(fontsize=8)
        for job in sorted({r['job'] for r in conv}):
            rows = [r for r in conv if r['job'] == job and r['rays'] == 2048]
            if rows:
                axes[1].plot(range(len(windows)), [100 * v for v in rows[0]['windows']], 'o-', label=job.replace('converge-', ''))
        axes[1].set_xticks(range(len(windows)), [f'{a}-{b} s' for a, b in windows])
        axes[1].set_ylabel('seed spread at 2,048 rays (%)')
        axes[1].set_title('no growth with time (bounce count)')
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(BENCH / 'fig1-convergence.png', dpi=120)
    # Figure 2: decay curves against Eyring.
    if result['physics']:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for job in room_jobs():
            try:
                meta, runs = load(job['name'])
            except FileNotFoundError:
                continue
            energy = np.mean([r['late'] for r in runs.values()], 0).sum(1).mean(0)[0]
            curve = np.cumsum(energy[::-1])[::-1]
            t = (np.arange(len(energy)) + .5) * meta['binSeconds']
            line, = axes[0].plot(t, 10 * np.log10(np.maximum(curve / curve[0], 1e-12)), label=job['name'])
            if 'absorptionCurve' not in job:
                axes[0].plot(t, -60 * t / eyring(job['size'], job['absorption'])[0], '--', color=line.get_color(), lw=.8)
        axes[0].set_ylim(-60, 2)
        axes[0].set_xlim(0, 2)
        axes[0].set_xlabel('time (s)')
        axes[0].set_ylabel('Schroeder decay, 125 Hz (dB)')
        axes[0].set_title('late field (solid) vs Eyring (dashed)')
        axes[0].legend(fontsize=7)
        stone = [r for r in result['physics'] if r['job'] == 'stone-air']
        if stone:
            x = np.arange(len(stone))
            axes[1].bar(x - .2, [r['T30'] for r in stone], .4, label='late field T30')
            axes[1].bar(x + .2, [r['eyring'] for r in stone], .4, label='Eyring with air')
            axes[1].set_xticks(x, [str(r['band']) for r in stone])
            axes[1].set_xlabel('band (Hz)')
            axes[1].set_ylabel('reverberation time (s)')
            axes[1].set_title('stone-like room, s = 0.3, air absorption')
            axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(BENCH / 'fig2-decay.png', dpi=120)
    # Figure 3: early images + late vs the image lattice.
    if result['lattice']:
        fig, axes = plt.subplots(1, len(result['lattice']), figsize=(11, 4), squeeze=False)
        for ax, row in zip(axes[0], result['lattice']):
            c = row['curves']
            t = (np.arange(len(c['late'])) + .5) * row['binSeconds']
            ax.plot(t, np.cumsum(np.array(c['images']) + c['late']), label='images + late (renderer)')
            ax.plot(t, np.cumsum(np.array(c['latticeEarly']) + c['latticeLate']), '--', label='analytic image lattice')
            ax.plot(t, np.cumsum(c['late']), label='late field only')
            ax.plot(t, np.cumsum(c['latticeLate']), ':', label=f"lattice beyond order {row['earlyOrder']}")
            ax.set_xlabel('time (s)')
            ax.set_ylabel('cumulative reflected energy')
            ax.set_title(row['job'])
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(BENCH / 'fig3-lattice.png', dpi=120)
    for row in result['lattice']:
        row.pop('curves')
    (BENCH / 'report.json').write_text(json.dumps(result, indent=1, default=float) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'church'}, indent=1, default=float)[:6000])


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'jobs':
        write_jobs()
    elif command == 'report':
        report()
    elif command == 'church':
        print(json.dumps(church(), indent=1))
    else:
        raise SystemExit(__doc__)
