"""Check the renderer's family gains against the offline law on real dataset scenes.

    PY=.context/fdtd-teacher/venv/bin/python
    B=docs/benchmarks/family-gains-2026-11-03
    $PY $B/verify.py prepare .context/family-gains/verify barriers-0001 canopy-0002 ...
    nice -n 19 $PY tools/dlss_jobs.py --data .context/family-gains/verify render --batch 12
    $PY $B/verify.py compare .context/family-gains/verify

Each pilot scene is rendered twice at the four input budgets with its pilot seeds:
gains off and gains on (scene.json "familyGains"). Checks: (1) gains off reproduces the
pilot's early families bit for bit; (2) gains on equals gains off times the offline law
(analyze.py routes_per_sample, the shipped FamilyGains.json) per family, with the walk
and direct families unchanged; (3) the solve time per receiver, off against on.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

PILOT = A.dd.DATA / 'scenes'


def prepare(out, names):
    for name in names:
        scene = json.loads((PILOT / name / 'scene.json').read_text())
        scene['render']['budgets'] = [b for b in scene['render']['budgets'] if b['name'] != 'label']
        for mode in ('off', 'on'):
            d = out / 'scenes' / f'{name}-{mode}'
            d.mkdir(parents=True, exist_ok=True)
            (d / 'scene.json').write_text(json.dumps(dict(scene, familyGains=mode == 'on'), indent=1) + '\n')


def read(folder, name, shape):
    a = np.fromfile(folder / f'responses-{name}.f32', '<f4').reshape(*shape, A.F, 2)
    return a[..., 0].astype(np.float64) + 1j * a[..., 1]


def compare(root):
    asset = json.loads(A.ASSET.read_text())
    spec = dict(kind='rho', k=asset['k'], power=asset['power'])
    rows = []
    for off_dir in sorted((root / 'scenes').glob('*-off')):
        name = off_dir.name[:-4]
        on_dir = root / 'scenes' / f'{name}-on'
        run_off = json.loads((off_dir / 'tracer/run.json').read_text())
        run_on = json.loads((on_dir / 'tracer/run.json').read_text())
        assert run_on['familyGains'] and not run_off['familyGains']
        edges = A.edge_count(json.loads((off_dir / 'tracer/boundary.json').read_text()))
        g = A.gains_for(spec, dict(rho=A.routes_per_sample(edges, run_off['primitives'])))
        row = dict(scene=name, edges=edges, primitives=run_off['primitives'], familyGainsHash=run_on['familyGainsHash'],
                   budgets=[])
        for b, v in enumerate(A.VARIANTS):
            S, P = len(run_off['budgets'][b]['seeds']), run_off['points']
            off, on = read(off_dir / 'tracer', v, (S, P, 32)), read(on_dir / 'tracer', v, (S, P, 32))
            pilot = read(PILOT / name / 'tracer', v, (S, P, 32))
            # Relative deviation per family, against that family's RMS (zero families excluded).
            dev = []
            for f in range(A.F):
                ref = g[b, f] * off[..., f]
                scale = math.sqrt(np.mean(np.abs(ref) ** 2))
                dev.append(float(np.max(np.abs(on[..., f] - ref)) / scale) if scale > 0 else float(np.max(np.abs(on[..., f]))))
            t_off = np.fromfile(off_dir / f'tracer/seconds-{v}.f32', '<f4')
            t_on = np.fromfile(on_dir / f'tracer/seconds-{v}.f32', '<f4')
            row['budgets'].append(dict(
                budget=v, gains=g[b].tolist(),
                offEarlyBitIdenticalToPilot=bool(np.array_equal(off[..., :A.EARLY], pilot[..., :A.EARLY])),
                walkUnchanged=bool(np.array_equal(on[..., A.EARLY], off[..., A.EARLY])),
                directUnchanged=bool(np.array_equal(on[..., 0], off[..., 0])),
                maxRelativeDeviation=dev,
                medianMsOff=1e3 * float(np.median(t_off)), medianMsOn=1e3 * float(np.median(t_on))))
        rows.append(row)
    out = dict(asset=str(A.ASSET.relative_to(A.ROOT)), assetSha256=A.dd.sha256(A.ASSET), scenes=rows)
    (HERE / 'verify.json').write_text(json.dumps(out, indent=1) + '\n')
    for r in rows:
        for x in r['budgets']:
            print(f"{r['scene']:15s} {x['budget'][6:]:6s} pilot-identical {x['offEarlyBitIdenticalToPilot']} walk {x['walkUnchanged']} "
                  f"direct {x['directUnchanged']} max dev {max(x['maxRelativeDeviation']):.2e}  ms off/on {x['medianMsOff']:.2f}/{x['medianMsOn']:.2f}")


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare(Path(sys.argv[2]), sys.argv[3:])
    else:
        compare(Path(sys.argv[2]))
