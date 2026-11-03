"""Residual-walk convergence and accuracy measurements (2 November 2026).

    PY=.context/fdtd-teacher/venv/bin/python
    $PY docs/benchmarks/residual-walk-2026-11-02/walk.py prepare ROOT          # calibration ladder scenes
    nice -n 19 $PY tools/dlss_jobs.py --data ROOT render --batch 4 [--variant tracer-rigid]
    $PY docs/benchmarks/residual-walk-2026-11-02/walk.py fdtd-prepare ROOT     # pilot rigid FDTD subset
    nice -n 19 $PY tools/dlss_jobs.py --data ROOT render --batch 6 --variant tracer-rigid
    $PY docs/benchmarks/residual-walk-2026-11-02/walk.py report                # report.json, tables.md, figures

Ladder: calibration scenes (generator index 900, outside the pilot) rendered with
128, 512, 2,048 and 8,192 walk rays per bin and 256 complete-route samples, four
seeds (two at 8,192), exactly as the pilot's walk_scaling.py. Accuracy: the
pilot's settled rigid barrier/canopy scenes re-rendered with their own recipe and
compared with the pilot's FDTD fields (read-only).
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_scenes as ds  # noqa: E402

PILOT = ROOT / '.context/audio-dlss-data'
LADDER = [(128, 4), (512, 4), (2048, 4), (8192, 2)]
FAMILIES = ['barriers', 'canopy', 'doorway', 'shoebox']
FDTD_FAMILIES = ('barriers', 'canopy')
TRUNCATION_LIMIT = .05


def prepare(root):
    """Ladder scenes. The pilot's barriers/doorway/shoebox calibration scenes (generator v1) are
    copied byte for byte so its walk table is directly comparable; canopy is generated (v2)."""
    for family in FAMILIES:
        pilot = ROOT / f'.context/audio-dlss-walk/scenes/{family}-0900-walk/scene.json'
        if pilot.exists():
            folder = root / 'scenes' / pilot.parent.name
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pilot, folder / 'scene.json')
            continue
        scene = ds.generate(family, 900)
        scene['sceneId'] += '-walk'
        scene['render'] = dict(budgets=[dict(name=f'walk{r}', raysPerBin=r, completeRouteSamples=256,
                                             seeds=[ds.stable_seed('walk', family, r, k) for k in range(n)])
                                        for r, n in LADDER])
        folder = root / 'scenes' / scene['sceneId']
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'scene.json').write_bytes(ds.dumps(scene))


def fdtd_prepare(root):
    """Copy the pilot's rigid barrier/canopy scenes (exact bytes) into a new data root."""
    for folder in sorted(PILOT.glob('scenes/*/fdtd')):
        scene = folder.parent
        if scene.name.split('-')[0] in FDTD_FAMILIES:
            target = root / 'scenes' / scene.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(scene / 'scene.json', target / 'scene.json')


def load(folder, name):
    run = json.loads((folder / 'run.json').read_text())
    budget = [b for b in run['budgets'] if b['name'] == name][0]
    S, P, Q, F = len(budget['seeds']), run['points'], len(run['frequencies']), len(run['families'])
    x = np.fromfile(folder / f'responses-{name}.f32', '<f4').reshape(S, P, Q, F, 2)
    peaks = np.fromfile(folder / f'peaks-{name}.f32', '<f4').reshape(S, P, Q, F)
    seconds = np.fromfile(folder / f'seconds-{name}.f32', '<f4').reshape(S, P)
    return run, budget, x[..., 0] + 1j * x[..., 1], peaks, seconds


def ladder_rows(root, variant):
    rows = []
    for folder in sorted(root.glob(f'scenes/*/{variant}')):
        run = json.loads((folder / 'run.json').read_text())
        reference = None
        for budget in run['budgets']:
            _, b, x, peaks, seconds = load(folder, budget['name'])
            S = len(b['seeds'])
            early, walk = x[..., :7].sum(-1), x[..., 7]
            if reference is None:
                reference = np.sqrt(np.mean(np.abs(early.mean(0)) ** 2))
            variance = (np.abs(walk - walk.mean(0)) ** 2).sum(0) / (S - 1)
            total = early + walk
            tv = (np.abs(total - total.mean(0)) ** 2).sum(0) / (S - 1)
            rows.append(dict(scene=run['sceneId'], variant=variant, raysPerBin=b['raysPerBin'], seeds=S,
                             walkSingleSeedSD=float(np.sqrt(variance.mean()) / reference),
                             walkMeanRms=float(np.sqrt(np.mean(np.abs(walk.mean(0)) ** 2)) / reference),
                             totalSingleSeedSD=float(np.sqrt(tv.mean()) / reference),
                             peakMax=float(peaks[..., 7].max()), directPeakMax=float(peaks[..., 0].max()),
                             secondsPerSolve=float(np.median(seconds))))
    return rows


def fdtd_rows(root, variant='tracer-rigid'):
    """Early field and total (early + walk) of the rigid label against FDTD, LoS and shadow."""
    out = []
    for folder in sorted(root.glob(f'scenes/*/{variant}')):
        scene_dir = folder.parent
        wave_dir = PILOT / 'scenes' / scene_dir.name / 'fdtd'
        if not wave_dir.exists():
            continue
        scene = json.loads((scene_dir / 'scene.json').read_text())
        fd = json.loads((wave_dir / 'run.json').read_text())
        run, b, x, _, _ = load(folder, 'label')
        P = run['points']
        wave = np.fromfile(wave_dir / 'wave.f32', '<f4').reshape(P, len(fd['frequencies']), 2)
        wave = wave[..., 0] + 1j * wave[..., 1]
        index = [run['frequencies'].index(hz) for hz in fd['frequencies']]
        early = x[..., :7].sum(-1).mean(0)[:, index]
        total = x.sum(-1).mean(0)[:, index]
        los = np.array(scene['lineOfSight'], bool)
        for j, hz in enumerate(fd['frequencies']):
            def err(v, m=slice(None)):
                return float(np.linalg.norm(v[m, j] - wave[m, j]) / np.linalg.norm(wave[m, j]))
            out.append(dict(scene=scene_dir.name, family=scene['family'], hz=hz,
                            settled=bool(fd['truncationChange075'][j] <= TRUNCATION_LIMIT),
                            early=err(early), total=err(total), earlyLos=err(early, los), totalLos=err(total, los),
                            earlyShadow=err(early, ~los) if (~los).any() else None,
                            totalShadow=err(total, ~los) if (~los).any() else None,
                            walkOverEarly=float(np.linalg.norm(total[:, j] - early[:, j]) / np.linalg.norm(early[:, j]))))
    return out


BEM = ROOT / '.context/wave-canopy-2026-10-27/reference-0.25-125.npz'


def bem_rows(directory):
    """125 Hz finite canopy/room map against the numerical BEM reference (solid-boundary fixture)."""
    reference = np.load(BEM)['pressure']
    r = np.genfromtxt(directory / 'renderer.csv', delimiter=',', names=True)
    out = []
    for rays in sorted(set(r['rays'])):
        for seed in sorted(set(r['seed'])):
            a = np.sort(r[(r['diffraction'] == 1) & (r['seed'] == seed) & (r['rays'] == rays)], order='point')
            if len(a) != len(reference):
                continue
            p = a['real'] + 1j * a['imag']
            shadow = a['direct_visible'] == 0
            out.append(dict(rays=int(rays), seed=int(seed), whole=float(np.linalg.norm(p - reference) / np.linalg.norm(reference)),
                            shadow=float(np.linalg.norm((p - reference)[shadow]) / np.linalg.norm(reference[shadow])),
                            visible=float(np.linalg.norm((p - reference)[~shadow]) / np.linalg.norm(reference[~shadow])),
                            medianSolveMs=float(1000 * np.median(a['solve_seconds']))))
    return out


def label_rows(root, variant='tracer', scenes=None):
    """Label standard error with and without the walk (dataset acceptance rule, K label seeds)."""
    import dlss_dataset as dd
    out = []
    for folder in sorted(root.glob(f'scenes/*/{variant}')):
        if scenes and folder.parent.name not in scenes:
            continue
        run = json.loads((folder / 'run.json').read_text())
        b = [x for x in run['budgets'] if x['name'] == 'label'][0]
        raw = np.fromfile(folder / 'responses-label.f32', '<f4').reshape(len(b['seeds']), run['points'], len(run['frequencies']), 8, 2)
        row = dict(scene=folder.parent.name, family=folder.parent.name.split('-')[0])
        for name, families in [('early', dd.FAMILIES[:7]), ('total', dd.FAMILIES)]:
            mean, variance, k = dd.label_statistics(raw, families)
            _, q = dd.acceptance(mean, variance, k)
            row[name] = dict(relativeStandardError=q['relativeStandardError'], validFraction=q['validFraction'], accepted=q['accepted'])
        out.append(row)
    return out


def rms(values):
    values = [v for v in values if v is not None]
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def summarise_fdtd(rows):
    table = {}
    for family in FDTD_FAMILIES:
        for hz in (125.0, 250.0):
            sel = [r for r in rows if r['family'] == family and r['hz'] == hz and r['settled']]
            table[f'{family}-{int(hz)}'] = dict(settled=len(sel), **{k: rms([r[k] for r in sel]) for k in
                                                 ['early', 'total', 'earlyLos', 'totalLos', 'earlyShadow', 'totalShadow', 'walkOverEarly']})
    return table


DATA = ROOT / '.context/residual-walk'
# The final shader is the `after` root. `edgeSchedule` (q = 1/(bounce+2)) and `edgeChoice`
# (q proportional to the diffuse coefficient) are rejected edge-branch probabilities;
# they change only preset walks, so their rigid renders equal `after` and are not repeated.
LADDERS = dict(before=[(ROOT / '.context/audio-dlss-walk', 'tracer'), (DATA / 'before', 'tracer'), (DATA / 'before', 'tracer-rigid')],
               after=[(DATA / 'after', 'tracer'), (DATA / 'after', 'tracer-rigid')],
               edgeSchedule=[(DATA / 'after2', 'tracer')],
               edgeChoice=[(DATA / 'after3', 'tracer')])


def cost_rows(root, variant, budget='input-1-256'):
    out = []
    for folder in sorted(root.glob(f'scenes/*/{variant}')):
        if (folder / f'seconds-{budget}.f32').exists():
            out.append(dict(scene=folder.parent.name, medianMs=float(1000 * np.median(np.fromfile(folder / f'seconds-{budget}.f32', '<f4')))))
    return out


def report():
    ladders = {name: [r for root, variant in roots for r in ladder_rows(root, variant)] for name, roots in LADDERS.items()}
    fdtd_before, fdtd_after = fdtd_rows(PILOT), fdtd_rows(DATA / 'fdtd-after')
    subset = sorted(p.parent.name for p in (DATA / 'label-after').glob('scenes/*/tracer'))
    out = dict(
        ladder=ladders,
        fdtd=dict(before=summarise_fdtd(fdtd_before), after=summarise_fdtd(fdtd_after), rowsBefore=fdtd_before, rowsAfter=fdtd_after),
        bem=dict(before=bem_rows(DATA / 'bem-before/full'), after=bem_rows(DATA / 'bem-after/full')),
        labels=dict(before=label_rows(PILOT, 'tracer', subset), after=label_rows(DATA / 'label-after', 'tracer')),
        cost=dict(rigidBefore=cost_rows(PILOT, 'tracer-rigid'), rigidAfter=cost_rows(DATA / 'fdtd-after', 'tracer-rigid'),
                  presetsBefore=[r for r in cost_rows(PILOT, 'tracer') if r['scene'] in subset],
                  presetsAfter=cost_rows(DATA / 'label-after', 'tracer')),
        diagnosis=json.loads((HERE / 'diagnosis.json').read_text()))
    (HERE / 'report.json').write_text(json.dumps(out, indent=1) + '\n')
    figures(out)
    return out


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parent / 'denoise-pilot-2026-10-29'))
    from analyze import COLORS, MARKERS, INK, MUTED, style
    style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, variant, title in zip(axes, ['tracer', 'tracer-rigid'], ['Material presets', 'Rigid walls (FDTD convention)']):
        for name, ls, alpha in [('before', '--', .55), ('after', '-', 1)]:
            rows = [r for r in out['ladder'][name] if r['variant'] == variant]
            for scene in sorted({r['scene'] for r in rows}):
                sel = sorted((r for r in rows if r['scene'] == scene), key=lambda r: r['raysPerBin'])
                family = scene.split('-')[0]
                ax.plot([r['raysPerBin'] for r in sel], [max(1e-4, 100 * r['walkSingleSeedSD']) for r in sel], ls=ls, alpha=alpha,
                        marker=MARKERS[family], ms=5, color=COLORS[family], label=f'{family} {name}')
        r0 = np.array([128, 8192])
        ax.plot(r0, 30 * np.sqrt(128 / r0), color=MUTED, lw=1, ls=':')
        ax.text(2500, 30 * np.sqrt(128 / 2500) * 1.4, '1/sqrt(N)', color=MUTED, fontsize=8)
        ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel('Residual-walk rays per frequency bin')
        ax.set_title(title, color=INK, loc='left', fontsize=10)
    axes[1].text(150, 2.2e-4, 'after: the walk is exactly zero (drawn at the axis floor)', color=MUTED, fontsize=8)
    axes[0].set_ylabel('Walk seed-to-seed SD / early-field RMS (%)')
    axes[0].legend(fontsize=7, ncol=2, loc='lower left')
    fig.suptitle('Before (dashed) and after (solid) the walk fix', color=INK, x=.01, ha='left', fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / 'fig1-convergence.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    keys = ['barriers-125', 'barriers-250', 'canopy-125', 'canopy-250']
    bars = [('early', 'Early field (both)', '#2a78d6'), ('totalBefore', 'Total before', '#eb6834'), ('totalAfter', 'Total after', '#1baf7a')]
    x = np.arange(len(keys))
    for i, (key, label, color) in enumerate(bars):
        vals = [100 * (out['fdtd']['before'][k]['early'] if key == 'early' else
                       out['fdtd']['before' if key == 'totalBefore' else 'after'][k]['total']) for k in keys]
        ax.bar(x + (i - 1) * .27, vals, .25, color=color, label=label)
        for xi, v in zip(x + (i - 1) * .27, vals):
            ax.text(xi, v * 1.08, f'{v:.0f}', ha='center', fontsize=7, color=MUTED)
    ax.set_yscale('log'); ax.set_ylim(10, 6000); ax.grid(axis='x', visible=False); ax.set_xticks(x); ax.set_xticklabels([k.replace('-', ', ') + ' Hz' for k in keys])
    ax.set_ylabel('Complex error vs FDTD (%)'); ax.legend(fontsize=8, loc='upper right')
    ax.set_title('Rigid tracer vs FDTD on the pilot\'s settled scenes', color=INK, loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(HERE / 'fig3-fdtd.png', dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['prepare', 'fdtd-prepare', 'ladder', 'fdtd', 'bem', 'labels', 'report'])
    p.add_argument('root', type=Path, nargs='?')
    p.add_argument('--variant', default='tracer')
    a = p.parse_args()
    if a.command == 'report':
        report()
    elif a.command == 'prepare':
        prepare(a.root)
    elif a.command == 'fdtd-prepare':
        fdtd_prepare(a.root)
    elif a.command == 'labels':
        print(json.dumps(label_rows(a.root, a.variant), indent=1))
    elif a.command == 'bem':
        print(json.dumps(bem_rows(a.root), indent=1))
    elif a.command == 'ladder':
        print(json.dumps(ladder_rows(a.root, a.variant), indent=1))
    else:
        rows = fdtd_rows(a.root, a.variant)
        print(json.dumps(dict(summary=summarise_fdtd(rows), rows=rows), indent=1))


if __name__ == '__main__':
    main()
