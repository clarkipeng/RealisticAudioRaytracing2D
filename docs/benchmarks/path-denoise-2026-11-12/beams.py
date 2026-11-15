"""Beam mode of the Path method against its exact specular connection on the wave-ground-truth suite.

    PY=.context/fdtd-teacher/venv/bin/python
    B=docs/benchmarks/path-denoise-2026-11-12
    nice -n 19 $PY $B/beams.py render 2048 4096    # guarded Unity runs of every suite scene, beam mode, two seeds
    $PY $B/beams.py score                          # both modes beside each other, same scorer as the scoreboard

Beam runs go to .context/methods-2026-11-06/raw-path-beam (same layout as tools/method_runs.py); the
exact connection's runs are the suite's own (raw/<scene>/path). Scoring packs each variant under its
own method name into the comparison root, scores, and removes it again (as variants.py).
"""
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(HERE))
import method_compare as mc  # noqa: E402
import method_runs as mr  # noqa: E402
import variants  # noqa: E402

BEAM = mr.WORK / 'raw-path-beam'
SEEDS = [0, 1]


def scenes():
    root = mc.DEFAULT_ROOT
    return [e['id'] for e in json.loads((root / 'suite.json').read_text())['scenes'] if (root / e['id'] / 'gt' / 'ref' / '0').exists()]


def render(budgets):
    path = mr.make_jobs('path-beam-suite', scenes(), ['path'], budgets, SEEDS)
    jobs = json.loads(path.read_text())['jobs']
    for job in jobs:
        job['output'] = str(BEAM); job['pathBeams'] = True
        scene = json.loads(Path(job['sceneFile']).read_text())['sceneId']
        if all((BEAM / scene / 'path' / f'r{b:05d}' / str(s) / 'run.json').exists() for b in budgets for s in SEEDS):
            continue
        part = mr.WORK / 'jobs' / f'path-beam-suite.{scene}.json'
        part.write_text(json.dumps(dict(jobs=[job]), indent=1) + '\n')
        mr.LOGS.mkdir(parents=True, exist_ok=True)
        mr.launch(f'path-beam-{scene}', ['-runTests', '-testPlatform', 'EditMode', '-testFilter', 'TakeChecks.MeasureMethods',
                                         '-testResults', str(mr.LOGS / f'.path-beam-{scene}.xml')], dict(ACOUSTIC_METHOD_JOBS=str(part)))


def busy(folder):
    seconds = [json.loads((d / 'run.json').read_text())['busySeconds'] for d in folder.glob('[0-9]*') if (d / 'run.json').exists()]
    return float(np.mean(seconds)) if seconds else None


def score():
    ids = scenes()
    names = {'path-exact': (mr.RAW, 'r02048')}
    for budget in sorted({p.name for s in ids for p in (BEAM / s / 'path').glob('r*')} if BEAM.exists() else []):
        names[f'path-beam-{int(budget[1:])}'] = (BEAM, budget)
    made = []
    try:
        cost = {}
        for name, (raw, budget) in names.items():
            cost[name] = {}
            for s in ids:
                folder = raw / s / 'path' / budget
                if folder.exists():
                    made += variants.pack(name, raw / s / 'path', budget, s)
                    cost[name][s] = busy(folder)
        rows = mc.score(mc.DEFAULT_ROOT, Path('/tmp/path-beams'), methods=list(names))['rows']
        metrics = ('level', 'EDT', 'T30', 'C80', 'etc')
        table = {}
        for name in names:
            sel = [r for r in rows if r['method'] == name]
            table[name] = {g: {k: mc.aggregate([r for r in sel if r['group'] == g], ('method',), k).get((name,)) for k in metrics}
                           for g in ('all', 'view', 'shadow')}
            table[name]['closed'] = {k: mc.aggregate([r for r in sel if r['group'] == 'all' and r['closed']], ('method',), k).get((name,)) for k in metrics}
            table[name]['seconds'] = float(np.mean([v for v in cost[name].values() if v])) if cost[name] else None
            table[name]['scenes'] = {s: {k: mc.aggregate([r for r in sel if r['group'] == 'all' and r['scene'] == s], ('method',), k).get((name,))
                                         for k in metrics} for s in ids}
        (HERE / 'beams.json').write_text(json.dumps(table, indent=1) + '\n')
        for name, t in table.items():
            print(f"{name:16s} {t['seconds'] or 0:6.2f} s/run  " + '  '.join(f"{g}: " + ' '.join(f"{k} {t[g][k]:.2f}" if t[g][k] is not None else f'{k} -' for k in metrics)
                                                                for g in ('all', 'shadow', 'closed')))
    finally:
        for d in set(made):
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'render':
        render([int(b) for b in sys.argv[2:]] or [2048])
    elif command == 'score':
        score()
    else:
        raise SystemExit(__doc__)
