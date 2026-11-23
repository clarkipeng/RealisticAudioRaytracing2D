"""Part B: diffraction as a walk event on the wave suite, at equal time (README-part-b.md).

    PY=.context/fdtd-teacher/venv/bin/python
    B=docs/benchmarks/bass-and-diffraction-2026-11-22
    nice -n 19 $PY $B/walk_suite.py profile current walk beams     # GPU time per solve of every suite scene, 2,048 rays
    $PY $B/walk_suite.py budgets                                   # rays per configuration for the current method's time
    nice -n 19 $PY $B/walk_suite.py render current 2048 2,3        # beams.py renders (raw-walk-<config>), the given seeds
    $PY $B/walk_suite.py score                                     # every configuration with four seeds, beams.py's scorer

The current method's seeds 0 and 1 are the scoreboard's runs (raw-path-shortcut); its seeds 2 and 3 and every walk run are
rendered with this code. Equal time: the summed GPU time per solve over the suite's scenes (MeasurePathProfile: edge
distribution, walk and readback, four receivers per scene), scaled linearly in rays.
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
    base = seconds('current')
    out = {}
    for name in CONFIGS:
        t = seconds(name)
        ratio = sum(base.values()) / sum(t.values())
        out[name] = dict(ratio=ratio, rays=int(round(RAYS * ratio)), seconds=t)
        print(f'{name}: {sum(t.values()) * 1000:.1f} ms per solve summed over the suite; equal time at {out[name]["rays"]} rays (x{ratio:.3f}); '
              + ', '.join(f'{s} {1000 * v:.1f}' for s, v in t.items()))
    (OUT / 'suite-budgets.json').write_text(json.dumps(out, indent=1) + '\n')


def render(name, budget, seeds):
    env = dict(os.environ, PATH_BEAMS_RAW=f'raw-walk-{name}', PATH_SEEDS=seeds, **{f'PATH_{k}': v for k, v in CONFIGS[name].items()})
    subprocess.run([sys.executable, str(BEAMS), 'render', budget], env=env, check=True)


def score():
    """Each configuration's four seeds in one folder (the current method's 0 and 1 copied from the scoreboard's runs)."""
    current = mr.WORK / 'raw-walk-current'
    for s in scenes():
        for seed in ('0', '1'):
            source, target = mr.WORK / 'raw-path-shortcut' / s / 'path' / 'r02048' / seed, current / s / 'path' / 'r02048' / seed
            if source.exists() and not target.exists():
                shutil.copytree(source, target)
    for name in CONFIGS:
        env = dict(os.environ, PATH_BEAMS_RAW=f'raw-walk-{name}', PATH_SCORE_JSON=str(OUT / f'suite-score-{name}.json'))
        subprocess.run([sys.executable, str(BEAMS), 'score'], env=env, check=True)


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'profile':
        profile(sys.argv[2:] or list(CONFIGS))
    elif command == 'budgets':
        budgets()
    elif command == 'render':
        render(sys.argv[2], sys.argv[3], sys.argv[4])
    elif command == 'score':
        score()
    else:
        raise SystemExit(__doc__)
