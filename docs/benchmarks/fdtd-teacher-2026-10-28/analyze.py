"""Score the FDTD teacher runs against analytic and BEM references; write report and figures.

    .context/fdtd-teacher/venv/bin/python docs/benchmarks/fdtd-teacher-2026-10-28/analyze.py \
      --evidence .context/fdtd-teacher/evidence --runs .context/fdtd-teacher/runs
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import fdtd_wave as fw  # noqa: E402
from compare_wave_field import shadow_mask  # noqa: E402

LEVELS = ['h50', 'h25', 'h17']
FIXTURES = dict(
    canopy=dict(hz=125.0, references={.5: 'reference-0.5-125.npz', .35: 'reference-0.35-125.npz',
                                      .25: 'reference-0.25-125.npz'}, fine=.25, probe=323),
    blocks=dict(hz=250.0, references={.14: 'reference-0.14-250.npz', .1: 'reference-0.1-250.npz'},
                fine=.1, probe=None),
)
# Projection: a 12 x 12 x 6.25 m (900 m^3) scene, rigid-free interior, 16-cell
# CPML on every side, 0.5 s and 1 s responses.
PROJECTION_VOLUME = (12.0, 12.0, 6.25)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(runs, name):
    path = runs / f'{name}.npz'
    if not path.exists():
        return None
    d = np.load(path)
    out = {k: d[k] for k in d.files if k != 'meta'}
    out['meta'] = json.loads(str(d['meta']))
    out['sha256'] = digest(path)
    return out


def rel(a, b, mask=None):
    return fw.relative_l2(a, b, mask)


def split_error(actual, target):
    """Amplitude-only and phase-only parts of a complex error."""
    ratio = actual / target
    return dict(amplitudeRelativeL2=float(np.linalg.norm(abs(actual) - abs(target)) / np.linalg.norm(target)),
                rmsPhaseRadians=float(np.sqrt(np.mean(np.angle(ratio) ** 2))),
                medianAmplitudeRatio=float(np.median(abs(ratio))))


def observed_order(f, h):
    """Order p from three levels with error ~ C h^p, and the finest-level error estimate."""
    d1, d2 = np.linalg.norm(f[0] - f[1]), np.linalg.norm(f[1] - f[2])
    g = lambda p: (h[0] ** p - h[1] ** p) / (h[1] ** p - h[2] ** p) - d1 / d2
    lo, hi = .1, 8.0
    if g(lo) * g(hi) > 0:
        return None, None
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if g(lo) * g(mid) > 0 else (lo, mid)
    p = (lo + hi) / 2
    finest = d2 * h[2] ** p / (h[1] ** p - h[2] ** p) / np.linalg.norm(f[2])
    return float(p), float(finest)


def richardson(f, h, p):
    r = (h[1] / h[2]) ** p
    return (r * f[2] - f[1]) / (r - 1)


def analyze(evidence, runs, output):
    budget = json.loads((HERE / 'budget.json').read_text())
    report = dict(budget=budget, solverSha256=digest(ROOT / 'tools/fdtd_wave.py'), fixtures={}, runs={})
    maps = {}
    for fixture, spec in FIXTURES.items():
        base = evidence / fixture / 'reference'
        scene = json.loads((base / 'scene.json').read_text())
        points = np.array(scene['points'], float).reshape(-1, 3)
        refs = {m: np.load(base / name)['pressure'] for m, name in spec['references'].items()}
        target = refs[spec['fine']]
        shadow = shadow_mask(scene, points)
        exact = fw.free_field_pressure(points, scene['source'], spec['hz'])
        coarse_ref = refs[sorted(refs)[1]]
        out = dict(hz=spec['hz'], points=len(points), shadowPoints=int(shadow.sum()),
                   referenceSha256=digest(base / spec['references'][spec['fine']]),
                   referenceRefinementChange=dict(whole=rel(coarse_ref, target), shadow=rel(coarse_ref, target, shadow)),
                   freeField={}, levels={}, refinement={}, truncation={}, absorbingLayer={})
        fields = {}
        for level in LEVELS:
            free = load(runs, f'{fixture}-free-{level}')
            if free is not None:
                report['runs'][f'{fixture}-free-{level}'] = free['meta'] | dict(npzSha256=free['sha256'])
                out['freeField'][level] = dict(spacing=free['meta']['spacing'],
                                               pointsPerWavelength=free['meta']['pointsPerWavelength'],
                                               complexRelativeL2=rel(free['H'], exact), **split_error(free['H'], exact))
            run = load(runs, f'{fixture}-{level}')
            if run is None:
                continue
            report['runs'][f'{fixture}-{level}'] = run['meta'] | dict(npzSha256=run['sha256'])
            H = run['H']
            fields[level] = H
            entry = dict(spacing=run['meta']['spacing'], pointsPerWavelength=run['meta']['pointsPerWavelength'],
                         duration=run['meta']['duration'], cells=run['meta']['cells'],
                         maxBoxSnap=run['meta']['maxBoxSnap'],
                         complexRelativeL2=rel(H, target), shadowComplexRelativeL2=rel(H, target, shadow),
                         litComplexRelativeL2=rel(H, target, ~shadow), **split_error(H, target))
            t, curve = run['truncationTimes'], run['truncationH']
            duration = run['meta']['duration']
            k = int(np.argmin(abs(t - .75 * duration)))
            entry['truncationChange'] = dict(fromSeconds=float(t[k]), toSeconds=duration,
                                             whole=rel(curve[k], H), shadow=rel(curve[k], H, shadow))
            if spec['probe'] is not None:
                i = spec['probe']
                entry['probe'] = dict(index=i, fdtd=[float(H[i].real), float(H[i].imag)],
                                      reference=[float(target[i].real), float(target[i].imag)],
                                      relativeError=float(abs(H[i] - target[i]) / abs(target[i])))
            out['levels'][level] = entry
            maps[(fixture, level)] = (run, scene, points, target, shadow)
        have = [lv for lv in LEVELS if lv in fields]
        for a, b in zip(have, have[1:]):
            out['refinement'][f'{a}->{b}'] = dict(whole=rel(fields[a], fields[b]), shadow=rel(fields[a], fields[b], shadow))
        if len(have) == 3:
            h = [out['levels'][lv]['spacing'] for lv in have]
            f = [fields[lv] for lv in have]
            p, finest = observed_order(f, h)
            out['refinement']['observedOrder'] = p
            out['refinement']['estimatedFinestDiscretizationError'] = finest
            fs = [x[shadow] for x in f]
            ps, finest_s = observed_order(fs, h)
            out['refinement']['observedOrderShadow'] = ps
            out['refinement']['estimatedFinestDiscretizationErrorShadow'] = finest_s
            if p:
                ext = richardson(f, h, p)
                out['richardsonVsReference'] = dict(order=p, whole=rel(ext, target), shadow=rel(ext, target, shadow))
                if spec['probe'] is not None:
                    i = spec['probe']
                    out['richardsonVsReference']['probe'] = float(abs(ext[i] - target[i]) / abs(target[i]))
        # Truncation study on the long coarse run.
        long = load(runs, f'{fixture}-long-h50')
        if long is not None:
            report['runs'][f'{fixture}-long-h50'] = long['meta'] | dict(npzSha256=long['sha256'])
            t, curve = long['truncationTimes'], long['truncationH']
            out['truncation'] = dict(referenceSeconds=float(t[-1]), times=t.tolist(),
                                     whole=[rel(c, curve[-1]) for c in curve],
                                     shadow=[rel(c, curve[-1], shadow) for c in curve])
            maps[(fixture, 'long')] = long
        # Absorbing-layer study: change against the thickest/farthest layer.
        for spacing_tag, ref_name in [('h50', 'pml32-m2'), ('h25', 'pml32-m2')]:
            ref = load(runs, f'{fixture}-{ref_name}-{spacing_tag}')
            if ref is None:
                continue
            report['runs'][f'{fixture}-{ref_name}-{spacing_tag}'] = ref['meta'] | dict(npzSha256=ref['sha256'])
            names = [f'{fixture}-{spacing_tag}'] + [f'{fixture}-pml{L}-m{m:g}-{spacing_tag}'
                                                     for L, m in [(8, .5), (32, .5), (16, .25), (16, 1.0), (16, 2.0)]
                                                     if f'pml{L}-m{m:g}' != ref_name]
            for name in names:
                run = load(runs, name)
                if run is None:
                    continue
                report['runs'][name] = run['meta'] | dict(npzSha256=run['sha256'])
                if abs(run['meta']['duration'] - ref['meta']['duration']) > 1e-9:
                    continue
                out['absorbingLayer'][name] = dict(pml=run['meta']['pml'], margin=run['meta']['margin'],
                                                   spacing=run['meta']['spacing'],
                                                   whole=rel(run['H'], ref['H']), shadow=rel(run['H'], ref['H'], shadow),
                                                   against=f'{fixture}-{ref_name}-{spacing_tag}')
        report['fixtures'][fixture] = out
    report['frequencySensitivity'] = sensitivity(evidence, runs)
    bench = runs.parent / 'logs/bench.json'
    report['benchmark'] = json.loads(bench.read_text()) if bench.exists() else None
    report['throughput'] = throughput(report['runs'])
    report['dispersion'] = dispersion(evidence, runs)
    report['projection'] = projection(report['throughput']['sustainedGcellPerSecond'], report['dispersion'])
    passed = dict(canopyWholeMap=None, blocksWholeMap=None)
    for fixture in FIXTURES:
        levels = report['fixtures'][fixture]['levels']
        if 'h17' in levels:
            passed[f'{fixture}WholeMap'] = bool(levels['h17']['complexRelativeL2'] <= budget['pass'][f'{fixture}WholeMap'])
    report['verdict'] = passed
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    figures(report, maps, output)
    return report


def sensitivity(evidence, runs, offsets=(-.4, -.2, -.1, 0, .1, .2, .4)):
    """Diagnostic only: the finest canopy field at 125 Hz +/- small offsets vs BEM.

    Shows how strongly the lightly damped room's response depends on frequency;
    it is not used to correct or score the reported field. Derived from the full
    receiver time series and cached so the slim archive can reproduce it.
    """
    cache = runs / 'canopy-h17-sensitivity.npz'
    full = runs / 'canopy-h17.npz'
    if full.exists() and 'pressure' in np.load(full).files:
        d = np.load(full)
        meta = json.loads(str(d['meta']))
        dt, dec = meta['dt'], meta['decimation']
        p, q = d['pressure'].astype(np.float64), d['source']
        fields = []
        for f in 125.0 + np.array(offsets):
            P = np.exp(-2j * np.pi * f * np.arange(len(p)) * dec * dt) @ p * dec * dt
            Q = np.sum(q * np.exp(-2j * np.pi * f * (np.arange(len(q)) + .5) * dt)) * dt
            fields.append(4 * np.pi * P / (1j * 2 * np.sin(np.pi * f * dt) / dt * Q))
        np.savez_compressed(cache, offsets=np.array(offsets), H=np.array(fields))
    if not cache.exists():
        return None
    d = np.load(cache)
    base = evidence / 'canopy/reference'
    scene = json.loads((base / 'scene.json').read_text())
    points = np.array(scene['points'], float).reshape(-1, 3)
    shadow = shadow_mask(scene, points)
    target = np.load(base / 'reference-0.25-125.npz')['pressure']
    at = list(d['offsets']).index(0)
    return dict(note='FDTD (1.67 cm) evaluated off 125 Hz; diagnostic of resonance sensitivity, not a fit.',
                rows=[dict(hz=125 + float(o), whole=rel(H, target), shadow=rel(H, target, shadow),
                           changeVs125=rel(H, d['H'][at])) for o, H in zip(d['offsets'], d['H'])])


def throughput(runs):
    rows = [dict(case=k, cells=v['cells'], steps=v['timing']['steps'], gcellPerSecond=v['timing']['gcellPerSecond'],
                 gpuSeconds=v['timing']['gpuSeconds'], wallSeconds=v['timing']['wallSeconds'],
                 peakDriverBytes=v['timing']['peakDriverBytes'], peakRssBytes=v['timing']['peakRssBytes'])
            for k, v in runs.items()]
    big = [r['gcellPerSecond'] for r in rows if r['cells'] > 3e7]
    return dict(runs=rows, sustainedGcellPerSecond=float(np.median(big)) if big else None,
                scope='GPU-busy time of paced runs (pacing sleeps excluded), all cells incl. CPML and solids; '
                      'includes source injection, 1,166-receiver recording and plane DFT.')


def dispersion(evidence, runs):
    """Free-field phase lag per wavelength travelled and amplitude bias, times ppw^2.

    Fitted on the coarsest free-field runs (asymptotic second-order regime):
    phase(r) = -(a / ppw^2) r/lambda + c and median |H|/|exact| - 1 = -b / ppw^2.
    """
    out = {}
    for fixture, spec in FIXTURES.items():
        run = load(runs, f'{fixture}-free-h50')
        if run is None:
            continue
        scene = json.loads((evidence / fixture / 'reference/scene.json').read_text())
        points = np.array(scene['points'], float).reshape(-1, 3)
        r = np.linalg.norm(points - scene['source'], axis=1) / (fw.SPEED_OF_SOUND / spec['hz'])
        ratio = run['H'] / fw.free_field_pressure(points, scene['source'], spec['hz'])
        slope = np.linalg.lstsq(np.c_[r, np.ones_like(r)], np.angle(ratio), rcond=None)[0][0]
        ppw = run['meta']['pointsPerWavelength']
        out[fixture] = dict(pointsPerWavelength=ppw, wavelengthsTravelled=[float(r.min()), float(r.max())],
                            phaseLagCoefficient=float(-slope * ppw ** 2),
                            amplitudeBiasCoefficient=float(-(np.median(abs(ratio)) - 1) * ppw ** 2))
    if out:
        out['model'] = dict(a=float(max(v['phaseLagCoefficient'] for v in out.values())),
                            b=float(max(v['amplitudeBiasCoefficient'] for v in out.values())),
                            note='complex error at D wavelengths ~ |exp(-i a D/ppw^2)(1 - b/ppw^2) - 1|; '
                                 'direction-averaged over the fixtures, 7-point Yee scheme at Courant 0.99/sqrt(3)')
    return out


def predicted_error(model, ppw, wavelengths):
    return float(abs(np.exp(-1j * model['a'] * wavelengths / ppw ** 2) * (1 - model['b'] / ppw ** 2) - 1))


def projection(gcell, disp):
    if not gcell:
        return None
    model = disp.get('model')
    rows = []
    for hz in [500, 1000]:
        for ppw in [10, 20, 35, 50]:
            h = fw.SPEED_OF_SOUND / hz / ppw
            n = [math.ceil(v / h) + 32 for v in PROJECTION_VOLUME]
            cells = float(np.prod(n))
            dt = fw.COURANT * h / fw.SPEED_OF_SOUND
            for seconds in [.5, 1.0]:
                steps = math.ceil(seconds / dt)
                rows.append(dict(hz=hz, pointsPerWavelength=ppw, spacing=h, cells=cells, steps=steps,
                                 responseSeconds=seconds, memoryGiB=fw.memory_bytes(fw.Grid(h, (0, 0, 0), tuple(n), 16)) / fw.GIB,
                                 gpuMinutes=cells * steps / gcell / 1e9 / 60,
                                 predictedError10Wavelengths=predicted_error(model, ppw, 10) if model else None,
                                 predictedErrorAcrossScene=predicted_error(model, ppw, 17.5 * hz / fw.SPEED_OF_SOUND) if model else None))
    return dict(volume=PROJECTION_VOLUME, gcellPerSecond=gcell, rows=rows,
                note='GPU-busy minutes per source run; multiply by 5 for 20% work pacing. '
                     'Predicted errors are free-field dispersion only, at 10 wavelengths and across the '
                     '17.5 m scene diagonal; geometry and boundary errors add to them.')


def figures(report, maps, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'figure.facecolor': '#10141b', 'axes.facecolor': '#181e28', 'text.color': '#e8edf5',
                         'axes.labelcolor': '#adb9cb', 'xtick.color': '#adb9cb', 'ytick.color': '#adb9cb',
                         'axes.edgecolor': '#343e4d', 'font.size': 10, 'legend.facecolor': '#181e28',
                         'legend.edgecolor': '#343e4d'})
    for fixture, name in [('canopy', 'comparison.png'), ('blocks', 'blocks.png')]:
        if (fixture, 'h17') not in maps:
            continue
        run, scene, points, target, shadow = maps[(fixture, 'h17')]
        H = run['H']
        hz = FIXTURES[fixture]['hz']
        meta = run['meta']
        axis = 1 if fixture == 'canopy' else 2
        u, v = [a for a in range(3) if a != axis]
        fixed = scene['view']['fixedValue'] if 'view' in scene else .2
        # Plane DFT -> same normalisation as receivers.
        dt = meta['dt']
        q = run['source']
        w = 2 * math.pi * hz
        Q = np.sum(q * np.exp(-1j * w * (np.arange(len(q)) + .5) * dt)) * dt
        plane = 4 * np.pi * run['plane'] / (1j * 2 * math.sin(w * dt / 2) / dt * Q)
        L = meta['pml']
        cu = meta['origin'][u] + (np.arange(meta['shape'][u]) + .5) * meta['spacing']
        cv = meta['origin'][v] + (np.arange(meta['shape'][v]) + .5) * meta['spacing']
        plane, cu, cv = plane[L:-L, L:-L], cu[L:-L], cv[L:-L]
        rms = np.sqrt(np.mean(abs(target) ** 2))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), layout='constrained')
        level = lambda x: 20 * np.log10(np.maximum(abs(x), 1e-12))
        image = axes[0].imshow(level(plane).T, origin='lower', extent=[cu[0], cu[-1], cv[0], cv[-1]],
                               cmap='magma', vmin=-40, vmax=10, interpolation='nearest')
        fig.colorbar(image, ax=axes[0], label='Level (dB re 1 m free field)', shrink=.8)
        axes[0].set_title(f'FDTD plane, h = {meta["spacing"]*100:.2f} cm')
        for ax, values, title, residual in [(axes[1], target, 'BEM reference at receivers', False),
                                            (axes[2], H - target, '|FDTD − BEM| / RMS(BEM)', True)]:
            c = abs(values) / rms if residual else level(values)
            sc = ax.scatter(points[:, u], points[:, v], c=c, s=14, marker='s', cmap='magma',
                            vmin=0 if residual else -40, vmax=.05 if residual else 10, linewidths=0)
            fig.colorbar(sc, ax=ax, label='Normalised residual' if residual else 'Level (dB re 1 m free field)', shrink=.8)
            ax.set_title(title)
        for ax in axes:
            for wall in scene['walls']:
                c, h = np.array(wall['center']), np.array(wall['half'])
                inside = abs(c[axis] - fixed) <= h[axis]
                ax.add_patch(Rectangle([c[u] - h[u], c[v] - h[v]], 2 * h[u], 2 * h[v],
                                       facecolor='#83909e' if inside else 'none', edgecolor='#e8edf5',
                                       linestyle='-' if inside else '--'))
            ax.plot(scene['source'][u], scene['source'][v], '*', color='#1accff', markersize=10)
            span = np.vstack([points, scene['source']])
            ax.set(xlabel='xyz'[u] + ' (m)', ylabel='xyz'[v] + ' (m)', aspect='equal',
                   xlim=(span[:, u].min() - .4, span[:, u].max() + .4),
                   ylim=(span[:, v].min() - .4, span[:, v].max() + .4))
        e = report['fixtures'][fixture]['levels']['h17']
        fig.suptitle(f'{hz:g} Hz {fixture} · FDTD vs BEM, no fitting · whole {e["complexRelativeL2"]*100:.2f}%, '
                     f'shadow {e["shadowComplexRelativeL2"]*100:.2f}% · slice {"xyz"[axis]} = {fixed} m', fontsize=13)
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
    # Convergence, truncation and absorbing-layer evidence.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    ax = axes[0]
    colors = dict(canopy='#ffb000', blocks='#1accff')
    for fixture, data in report['fixtures'].items():
        lv = [k for k in LEVELS if k in data['levels']]
        if lv:
            h = [data['levels'][k]['spacing'] * 100 for k in lv]
            ax.plot(h, [data['levels'][k]['complexRelativeL2'] * 100 for k in lv], 'o-', color=colors[fixture],
                    label=f'{fixture} vs BEM, whole')
            ax.plot(h, [data['levels'][k]['shadowComplexRelativeL2'] * 100 for k in lv], 's--', color=colors[fixture],
                    label=f'{fixture} vs BEM, shadow')
        fl = [k for k in LEVELS if k in data['freeField']]
        if fl:
            ax.plot([data['freeField'][k]['spacing'] * 100 for k in fl],
                    [data['freeField'][k]['complexRelativeL2'] * 100 for k in fl], '^:', color=colors[fixture],
                    label=f'{fixture} free field vs analytic')
        ref = data['referenceRefinementChange']['whole'] * 100
        ax.axhline(ref, color=colors[fixture], lw=.8, alpha=.5)
    hh = np.array([1.6, 5])
    ax.plot(hh, .12 * (hh / 5) ** 2, color='#adb9cb', lw=.8)
    ax.plot(hh, 1.2 * (hh / 5), color='#adb9cb', lw=.8, ls=':')
    ax.text(3.2, .12 * (3.2 / 5) ** 2 * .7, 'h²', color='#adb9cb')
    ax.text(3.2, 1.2 * (3.2 / 5) * .78, 'h', color='#adb9cb')
    ax.axhline(5, color='#ff5c5c', lw=1)
    ax.text(1.62, 5.4, '5% budget', color='#ff5c5c')
    ax.set(xscale='log', yscale='log', xlabel='Grid spacing h (cm)', ylabel='Complex relative L2 (%)',
           title='Grid refinement (thin lines: BEM mesh change)')
    ax.legend(fontsize=7.5)
    ax = axes[1]
    tr = report['fixtures']['canopy'].get('truncation')
    if tr:
        ax.semilogy(tr['times'][:-1], np.array(tr['whole'][:-1]) * 100, color='#ffb000', label='whole map')
        ax.semilogy(tr['times'][:-1], np.array(tr['shadow'][:-1]) * 100, '--', color='#ffb000', label='shadow')
        ax.axhline(1, color='#adb9cb', lw=.8)
        ax.set(xlabel='Truncation time T (s)', ylabel=f'Change vs T = {tr["referenceSeconds"]:g} s (%)',
               title='Canopy time support (h = 5 cm)')
        ax.legend()
    ax = axes[2]
    rows = [(k, v) for f in report['fixtures'].values() for k, v in f['absorbingLayer'].items()]
    if rows:
        labels = [f'{v["pml"]} cells, {v["margin"]:g} m\n{v["spacing"]*100:g} cm' for _, v in rows]
        ax.barh(range(len(rows)), [v['whole'] * 100 for _, v in rows], color='#1accff', label='whole')
        ax.barh(range(len(rows)), [v['shadow'] * 100 for _, v in rows], color='none', edgecolor='#ffb000',
                label='shadow')
        ax.set_yticks(range(len(rows)), [f'{k.split("-")[0]} · {lab}' for (k, _), lab in zip(rows, labels)], fontsize=7)
        ax.set(xscale='log', xlabel='Change vs thickest layer at same h (%)', title='Absorbing layer thickness / distance')
        ax.legend()
    fig.savefig(output / 'convergence.png', dpi=150)
    plt.close(fig)


def package(runs, output):
    """evidence.zip: every run without its receiver time series (kept in .context).

    Plane maps are kept for the two reported runs only.

    The slim files carry everything analyze.py reads; the manifest records the
    hash of each full run file and of each logged run.
    """
    import io
    import zipfile
    manifest = {}
    with zipfile.ZipFile(output / 'evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(runs.glob('*.npz')):
            d = np.load(path)
            # Plane maps are only drawn for the reported (finest) runs.
            drop = {'pressure'} | (set() if path.stem in ('canopy-h17', 'blocks-h17') else {'plane'})
            keep = {k: d[k] for k in d.files if k not in drop}
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **keep)
            z.writestr(f'runs/{path.name}', buffer.getvalue())
            if path.with_suffix('.json').exists():
                z.write(path.with_suffix('.json'), f'runs/{path.stem}.json')
            manifest[path.name] = dict(fullSha256=digest(path), slimSha256=hashlib.sha256(buffer.getvalue()).hexdigest())
        logs = runs.parent / 'logs'
        for log in sorted([*logs.glob('*.log'), *logs.glob('bench.json')]):
            z.write(log, f'logs/{log.name}')
        z.writestr('manifest.json', json.dumps(manifest, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evidence', type=Path, default=ROOT / '.context/fdtd-teacher/evidence')
    p.add_argument('--runs', type=Path, default=ROOT / '.context/fdtd-teacher/runs')
    p.add_argument('--output', type=Path, default=HERE)
    p.add_argument('--package', action='store_true', help='Also write evidence.zip')
    a = p.parse_args()
    if a.package:
        package(a.runs, a.output)
    report = analyze(a.evidence, a.runs, a.output)
    summary = {f: dict(levels={k: {m: v[m] for m in ['complexRelativeL2', 'shadowComplexRelativeL2']}
                               for k, v in d['levels'].items()},
                       freeField={k: v['complexRelativeL2'] for k, v in d['freeField'].items()},
                       refinement=d['refinement'])
               for f, d in report['fixtures'].items()}
    print(json.dumps(dict(summary=summary, verdict=report['verdict']), indent=2))


if __name__ == '__main__':
    main()
