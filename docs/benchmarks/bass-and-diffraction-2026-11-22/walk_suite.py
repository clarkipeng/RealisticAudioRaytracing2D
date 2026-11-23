"""Part B: diffraction as a walk event on the wave suite, at equal time (README.md, part B).

    PY=.context/fdtd-teacher/venv/bin/python
    B=docs/benchmarks/bass-and-diffraction-2026-11-22
    nice -n 19 $PY $B/walk_suite.py profile current walk beams     # GPU time per solve of every suite scene, 2,048 rays
    $PY $B/walk_suite.py budgets                                   # rays per scene and configuration for the current method's time
    nice -n 19 $PY $B/walk_suite.py render current 2,3             # beams.py renders (raw-walk-<config>), the given seeds
    nice -n 19 $PY $B/walk_suite.py render walk 0,1,2,3
    $PY $B/walk_suite.py score                                     # every configuration with four seeds and the static-IR baseline

The current method's seeds 0 and 1 are the scoreboard's runs (raw-path-shortcut); its seeds 2 and 3 and every walk run are
rendered with this code. Equal time, scene by scene: the GPU time per solve (MeasurePathProfile: edge distribution, walk
and readback, four receivers), scaled linearly in rays.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import method_compare as mc  # noqa: E402
import method_runs as mr  # noqa: E402

BEAMS = ROOT / 'docs/benchmarks/path-denoise-2026-11-12/beams.py'
OUT = ROOT / '.context/walk-diffraction'
CONFIGS = {'current': {}, 'walk': {'WALK_DIFFRACTION': '1'}, 'beams': {'WALK_DIFFRACTION': '1', 'BEAMS_ONLY': '1'}}
RAYS = 2048


def scenes():
    root = mc.DEFAULT_ROOT
    return [e['id'] for e in json.loads((root / 'suite.json').read_text())['scenes'] if (root / e['id'] / 'gt' / 'ref' / '0').exists()]


def profile(configs):
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [str(mr.scene_spec(mc.DEFAULT_ROOT / s, mr.WORK / 'specs' / f'{s}.json')) for s in scenes()]
    for name in configs:
        out = OUT / f'suite-profile-{name}.json'
        env = dict(ACOUSTIC_PATH_PROFILE=str(out), ACOUSTIC_PATH_PROFILE_SCENES=','.join(specs), ACOUSTIC_PATH_PROFILE_RAYS=str(RAYS),
                   ACOUSTIC_PATH_PROFILE_FLAGS='0', **{f'ACOUSTIC_PATH_PROFILE_{k}': v for k, v in CONFIGS[name].items()})
        mr.tests('TakeChecks.MeasurePathProfile', env)


def seconds(name):
    """GPU seconds per solve by scene (edge distribution + walk + readback)."""
    data = json.loads((OUT / f'suite-profile-{name}.json').read_text())
    return {s['scene'].split('|')[0]: sum(s['variants'][0][k] for k in ('edge', 'walk', 'readback')) for s in data['scenes']}


def budgets():
    """Rays per scene and configuration for the current method's time at 2,048 rays (the suite-wide ratio printed too)."""
    base = seconds('current')
    out = {}
    for name in CONFIGS:
        t = seconds(name)
        out[name] = {s: int(round(RAYS * base[s] / t[s])) for s in t}
        print(f'{name}: {sum(t.values()) * 1000:.1f} ms per solve summed over the suite (x{sum(base.values()) / sum(t.values()):.3f}); rays at equal time: '
              + ', '.join(f'{s} {out[name][s]}' for s in t))
    (OUT / 'suite-budgets.json').write_text(json.dumps(out, indent=1) + '\n')


def render(name, seeds):
    """Every scene at its own equal-time budget."""
    rays = json.loads((OUT / 'suite-budgets.json').read_text())[name]
    for scene, budget in rays.items():
        env = dict(os.environ, PATH_BEAMS_RAW=f'raw-walk-{name}', PATH_SEEDS=seeds, PATH_SCENES=scene, **{f'PATH_{k}': v for k, v in CONFIGS[name].items()})
        subprocess.run([sys.executable, str(BEAMS), 'render', str(2048 if name == 'current' else budget)], env=env, check=True)


def score():
    """Each configuration, every scene at its equal-time budget with four seeds, beside the static-IR baseline; beams.py's
    scorer and groups (all / in view / shadow / closed rooms). The current method's seeds 0 and 1 are the scoreboard's runs."""
    sys.path.insert(0, str(BEAMS.parent))
    import variants
    rays = json.loads((OUT / 'suite-budgets.json').read_text())
    made, table = [], {}
    metrics = ('level', 'EDT', 'T30', 'etc', 'levelBias')
    try:
        for name in CONFIGS:
            for s in scenes():
                budget = f'r{2048 if name == "current" else rays[name][s]:05d}'
                folder = mr.WORK / f'raw-walk-{name}' / s / 'path'
                if name == 'current':
                    for seed in ('0', '1'):
                        source, target = mr.WORK / 'raw-path-shortcut' / s / 'path' / budget / seed, folder / budget / seed
                        if source.exists() and not target.exists():
                            shutil.copytree(source, target)
                made += variants.pack(f'walk-{name}', folder, budget, s)
        rows = mc.score(mc.DEFAULT_ROOT, Path('/tmp/walk-suite'), methods=[f'walk-{n}' for n in CONFIGS] + ['baseline-static-ir'])['rows']
        for method in [f'walk-{n}' for n in CONFIGS] + ['baseline-static-ir']:
            sel = [r for r in rows if r['method'] == method]
            table[method] = {g: {k: mc.aggregate([r for r in sel if r['group'] == g], ('method',), k).get((method,)) for k in metrics} for g in ('all', 'view', 'shadow')}
            table[method]['closed'] = {k: mc.aggregate([r for r in sel if r['group'] == 'all' and r['closed']], ('method',), k).get((method,)) for k in metrics}
            table[method]['scenes'] = {s: {k: mc.aggregate([r for r in sel if r['group'] == 'all' and r['scene'] == s], ('method',), k).get((method,)) for k in metrics}
                                       for s in scenes()}
            print(f'{method:20s} ' + '  '.join(f'{g}: ' + ' / '.join(f'{table[method][g][k]:.2f}' if table[method][g][k] is not None else '-' for k in metrics)
                                               for g in ('all', 'view', 'shadow', 'closed')))
        (OUT / 'suite-score.json').write_text(json.dumps(table, indent=1) + '\n')
    finally:
        for d in set(made):
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'profile':
        profile(sys.argv[2:] or list(CONFIGS))
    elif command == 'budgets':
        budgets()
    elif command == 'render':
        render(sys.argv[2], sys.argv[3])
    elif command == 'score':
        score()
    else:
        raise SystemExit(__doc__)
