"""Residual-walk convergence check: does the walk family settle as rays increase?

    PY=.context/fdtd-teacher/venv/bin/python
    $PY docs/benchmarks/denoise-pilot-2026-10-29/walk_scaling.py prepare .context/audio-dlss-walk
    nice -n 19 $PY tools/dlss_jobs.py --data .context/audio-dlss-walk render --batch 3
    $PY docs/benchmarks/denoise-pilot-2026-10-29/walk_scaling.py analyze .context/audio-dlss-walk

Three calibration scenes (index 900, outside the pilot) are rendered with 128,
512, 2,048 and 8,192 walk rays per bin and a fixed small complete-route budget.
Writes walk.json and fig6-walk.png next to this file.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'tools'))
import dlss_dataset as dd  # noqa: E402
import dlss_scenes as ds  # noqa: E402

LADDER = [(128, 4), (512, 4), (2048, 4), (8192, 2)]
FAMILIES = ['barriers', 'doorway', 'shoebox']


def prepare(root):
    for family in FAMILIES:
        scene = ds.generate(family, 900)
        scene['sceneId'] += '-walk'
        scene['render'] = dict(budgets=[dict(name=f'walk{r}', raysPerBin=r, completeRouteSamples=256,
                                             seeds=[ds.stable_seed('walk', family, r, k) for k in range(n)])
                                        for r, n in LADDER])
        folder = root / 'scenes' / scene['sceneId']
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'scene.json').write_bytes(ds.dumps(scene))


def analyze(root):
    rows = []
    for folder in sorted(root.glob('scenes/*/tracer')):
        run = json.loads((folder / 'run.json').read_text())
        P, Q, F = run['points'], len(run['frequencies']), len(run['families'])
        reference = None
        for budget in run['budgets']:
            name, S = budget['name'], len(budget['seeds'])
            x = dd.complex_of(np.fromfile(folder / f'responses-{name}.f32', '<f4').reshape(S, P, Q, F, 2))
            early, walk = x[..., :7].sum(-1), x[..., 7]
            if reference is None:
                reference = np.sqrt(np.mean(np.abs(early.mean(0)) ** 2))
            variance = (np.abs(walk - walk.mean(0)) ** 2).sum(0) / (S - 1)
            peaks = np.fromfile(folder / f'peaks-{name}.f32', '<f4').reshape(S, P, Q, F)[..., 7]
            counts = np.fromfile(folder / f'counts-{name}.i64', '<i8').reshape(S, P, Q, F)[..., 7]
            direct = np.fromfile(folder / f'peaks-{name}.f32', '<f4').reshape(S, P, Q, F)[..., 0]
            rows.append(dict(scene=run['sceneId'], raysPerBin=budget['raysPerBin'], seeds=S,
                             walkSingleSeedSD=float(np.sqrt(variance.mean()) / reference),
                             walkMeanRms=float(np.sqrt(np.mean(np.abs(walk.mean(0)) ** 2)) / reference),
                             peakMax=float(peaks.max()), peakP99=float(np.percentile(peaks, 99)),
                             directPeakMax=float(direct.max()), arrivalsPerBin=float(counts.mean()),
                             secondsPerSolve=float(np.fromfile(folder / f'seconds-{name}.f32', '<f4').mean())))
    (HERE / 'walk.json').write_text(json.dumps(dict(rows=rows, note='SD and mean are relative to the RMS early field '
                                                    '(all other families) of the same scene'), indent=1) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE))
    from analyze import COLORS, MARKERS, INK, MUTED, style
    style(plt)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for scene in sorted({r['scene'] for r in rows}):
        sel = [r for r in rows if r['scene'] == scene]
        family = scene.split('-')[0]
        ax.plot([r['raysPerBin'] for r in sel], [100 * r['walkSingleSeedSD'] for r in sel], marker=MARKERS[family],
                color=COLORS[family], label=scene.replace('-walk', ''))
    r0 = np.array([128, 8192])
    start = np.median([100 * r['walkSingleSeedSD'] for r in rows if r['raysPerBin'] == 128])
    ax.plot(r0, start * np.sqrt(128 / r0), color=MUTED, lw=1, ls='--')
    ax.text(1500, start * np.sqrt(128 / 1500) * .55, '1/sqrt(N) expected', color=MUTED, fontsize=8)
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel('Residual-walk rays per frequency bin'); ax.set_ylabel('Walk seed-to-seed SD / early-field RMS (%)')
    ax.set_title('The residual walk does not converge with more rays', color=INK, loc='left'); ax.legend()
    fig.tight_layout(); fig.savefig(HERE / 'fig6-walk.png', dpi=160); plt.close(fig)
    for r in rows:
        print(r)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['prepare', 'analyze'])
    p.add_argument('root', type=Path)
    a = p.parse_args()
    (prepare if a.command == 'prepare' else analyze)(a.root)


if __name__ == '__main__':
    main()
