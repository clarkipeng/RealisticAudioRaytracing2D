"""Score path-method variants against the wave ground truth, side by side (same scorer as the scoreboard).

    .context/fdtd-teacher/venv/bin/python docs/benchmarks/path-denoise-2026-11-12/variants.py

Each variant is a set of raw method runs (tools/method_runs.py layout) packed under its own method
name into the comparison root, scored, then removed again so the scoreboard's methods are untouched.
"""
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import method_compare as mc  # noqa: E402
import method_runs as mr  # noqa: E402

RAW = mr.RAW
V1 = RAW.parent / 'raw-path-v1'
S16 = RAW.parent / 'raw-path-surv16'
# name -> (raw folder of a scene's runs given the scene id, budget folder, cost note)
VARIANTS = {
    'energy': (lambda s: RAW / s / 'energy', 'r02048'),
    'path-v1': (lambda s: V1 / s / 'path', 'r02048'),                  # 12 Nov first version: every path to the horizon
    'path-surv16': (lambda s: S16 / s / 'path', 'r02048'),             # survival in time, T0 = 16 ms
    'path-surv16-16k': (lambda s: S16 / s / 'path', 'r16384'),
    'path-surv100': (lambda s: S16 / f'{s}-s100' / 'path', 'r02048'),  # survival in time, T0 = 100 ms
    'path-default': (lambda s: RAW / s / 'path', 'r02048'),            # the default after this change (T0 = 100 ms)
}


def pack(name, folder, budget, scene):
    made = []
    for seed_dir in sorted((folder / budget).glob('[0-9]*')):
        if not (seed_dir / 'run.json').exists():
            continue
        run, data = mr.read_raw(seed_dir)
        out = mc.DEFAULT_ROOT / scene / name / budget / seed_dir.name
        meta = dict(method=name, budget=budget, seed=int(seed_dir.name), budgetValue=run['budget'], budgetUnit='rays', sceneId=scene,
                    sceneSha256=run['sceneSha256'], code=dict(name=run['method'], sha256=run['codeSha256']), wallSeconds=run['busySeconds'],
                    validSeconds=run['maximumDelaySeconds'], bandLimited=False, wallModel='absorption', notes=name)
        mc.write_output(out, meta, data['energy'].astype(np.float32), None, data['direction'])
        made.append(out.parent.parent)
    return made


def main(scenes=('shoebox-0020-presets', 'shoebox-0010-presets', 'doorway-0008-presets', 'corridor-0013-presets', 'furnished-0005-presets')):
    made = []
    try:
        for name, (folder, budget) in VARIANTS.items():
            if name == 'energy':
                continue
            for scene in scenes:
                if folder(scene).exists():
                    made += pack(name, folder(scene), budget, scene)
        rows = mc.score(mc.DEFAULT_ROOT, Path('/tmp/path-variants'))['rows']
        result = {}
        for name, (folder, budget) in VARIANTS.items():
            sel = [r for r in rows if r['method'] == (name if name == 'energy' else name) and r['scene'] in scenes and r['group'] == 'all'
                   and (name != 'energy' or r['budget'] == budget)]
            cost = []
            for scene in scenes:
                for seed_dir in (folder(scene) / budget).glob('[0-9]*'):
                    if (seed_dir / 'run.json').exists():
                        run = json.loads((seed_dir / 'run.json').read_text())
                        cost.append(run['busySeconds'] / run['receivers'])
            result[name] = {k: mc.aggregate(sel, ('method',), k).get((name,)) for k in ('level', 'levelBias', 'EDT', 'T30', 'etc')}
            result[name]['msPerUpdate'] = 1000 * float(np.mean(cost)) if cost else None
            print(name.ljust(14), ' '.join(f'{k} {v:.2f}' if v is not None else f'{k} -' for k, v in result[name].items()))
        return result
    finally:
        for folder in set(made):
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == '__main__':
    main()
