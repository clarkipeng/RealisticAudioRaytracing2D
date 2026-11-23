"""Part B: diffraction as a walk event in the Church, at equal time per update (README-part-b.md).

    PY=.context/fdtd-teacher/venv/bin/python
    B=docs/benchmarks/bass-and-diffraction-2026-11-22
    nice -n 19 $PY $B/walk_church.py profile current walk beams 4000,8000,16000   # GPU time per update against N in flight
    nice -n 19 $PY $B/walk_church.py levels current:N walk:N beams:N              # 64 updates at the outside and inside points
    $PY $B/walk_church.py table                                                   # part 20's table format

Configurations: `current` (shortcuts, the source's edge rays and edge connections), `walk` (shortcuts with walk
diffraction), `beams` (beams only with walk diffraction). Profiles use the Church walk's persistent render settings
(path-denoise-2026-11-12/church.py, 6 s horizon) through MeasurePathProfile, the levels MeasureChurchLevels (2 s horizon,
persistent rays under population control, 1,024-sample updates, a horizon of warm-up, then 64 updates); both as part 20.
"""
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'docs/benchmarks/path-denoise-2026-11-12'))
import method_runs as mr  # noqa: E402

OUT = ROOT / '.context/walk-diffraction'
CONFIGS = {'current': {}, 'walk': {'WALK_DIFFRACTION': '1'}, 'beams': {'WALK_DIFFRACTION': '1', 'BEAMS_ONLY': '1'}}
POINTS = {'outside': 16.5, 'inside': 3.0}   # seconds into the take (part 20)
RATE = 44100


def job():
    import church
    path = OUT / 'church-persistent.job.json'
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(take=str(church.listen.TAKE), output=str(OUT / 'unused'), shader='Assets/Shaders/Transport.compute',
                                    settings=church.RENDERS['path-persistent'], workFraction=.5), indent=1) + '\n')
    return path


def profile(configs, populations):
    path = job()
    for name in configs:
        out = OUT / f'profile-{name}{os.environ.get("WALK_PROFILE_TAG", "")}.json'
        env = dict(ACOUSTIC_PATH_PROFILE=str(out), ACOUSTIC_PATH_PROFILE_TAKE=str(path), ACOUSTIC_PATH_PROFILE_RAYS=populations,
                   ACOUSTIC_PATH_PROFILE_PERSISTENT='1', ACOUSTIC_PATH_PROFILE_POPULATION='1', ACOUSTIC_PATH_PROFILE_FLAGS=os.environ.get('WALK_PROFILE_FLAGS', '0'),
                   **{f'ACOUSTIC_PATH_PROFILE_{k}': v for k, v in CONFIGS[name].items()})
        mr.tests('TakeChecks.MeasurePathProfile', env)
        for scene in json.loads(out.read_text())['scenes']:
            for v in scene['variants']:
                print(f"{name}: N {scene['rays']}, flags {v['flags']}, in flight {v['inFlight']}, walk {1000 * v['walk']:.1f} ms per update, "
                      f"edge phase {1000 * v['edge']:.1f} ms, events {v['events']}, deposits {v.get('edgeConnections')}, longest chain {v['candidates']}")


def levels(pairs):
    import church
    take = church.listen.TAKE
    for pair in pairs:
        name, population = pair.split(':')
        for point, seconds in POINTS.items():
            out = OUT / f'levels-{name}-{population}-{point}.json'
            env = dict(ACOUSTIC_CHURCH_LEVELS=str(out), ACOUSTIC_CHURCH_LEVELS_TAKE=str(take), ACOUSTIC_CHURCH_LEVELS_POINTS='take',
                       ACOUSTIC_CHURCH_LEVELS_SAMPLE=str(int(seconds * RATE)), ACOUSTIC_CHURCH_LEVELS_ENERGY='', ACOUSTIC_CHURCH_LEVELS_PATH='0.25:0',
                       ACOUSTIC_CHURCH_LEVELS_POPULATION=population, ACOUSTIC_CHURCH_LEVELS_UPDATES='64',
                       **{f'ACOUSTIC_CHURCH_LEVELS_{k}': v for k, v in CONFIGS[name].items()})
            mr.tests('TakeChecks.MeasureChurchLevels', env)


def table():
    rows = {}
    for path in sorted(OUT.glob('levels-*.json')):
        name, population, point = path.stem.split('-')[1:]
        name = f'{name} N={population}'
        updates = np.array(json.loads(path.read_text())['Path'])   # [update][band][window]
        late = updates[:, [0, 2, 4], 2]
        mean = late.mean(0)
        rows[(point, name)] = dict(updates=len(late), level=float(mean[0]), sd=(late.std(0) / mean).tolist(), median=(np.median(late, 0) / mean).tolist(),
                                   levels=mean.tolist())
        print(f"{point:8s} {name:16s} {len(late)} updates: level (125 Hz) {mean[0]:.2e}; sd / mean " + ' / '.join(f'{x:.2f}' for x in late.std(0) / mean) +
              '; median / mean ' + ' / '.join(f'{x:.2f}' for x in np.median(late, 0) / mean) + '; mean 125/500/2000 ' + ' / '.join(f'{x:.2e}' for x in mean))
    (OUT / 'levels.json').write_text(json.dumps({f'{p}/{n}': v for (p, n), v in rows.items()}, indent=1) + '\n')


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'profile':
        profile(sys.argv[2:-1], sys.argv[-1])
    elif command == 'levels':
        levels(sys.argv[2:])
    elif command == 'table':
        table()
    else:
        raise SystemExit(__doc__)
