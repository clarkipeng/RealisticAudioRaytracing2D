"""Wave-suite scores of path-method code variants side by side (README parts 49 onward; the scoreboard's scorer).

    .context/fdtd-teacher/venv/bin/python docs/benchmarks/path-denoise-2026-11-12/suite.py NAME=RAWROOT[:SUFFIX] ...

Each NAME is a set of full-solve runs at 2,048 rays (tools/method_runs.py layout: RAWROOT/<scene>[-SUFFIX]/path/r02048/<seed>),
e.g. today=.context/methods-2026-11-06/raw-path-bvh and new=.context/methods-2026-11-06/raw:p49 (jobs made with --suffix p49).
They are packed under their names into the comparison root, scored, and removed again, so the scoreboard is untouched.
Printed per variant and group: level dB / EDT % / T30 % / ETC dB, RMS over receivers and bands, and the seeds found.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import method_compare as mc  # noqa: E402
import method_runs as mr  # noqa: E402

SCENES = [p.name for p in sorted(mc.DEFAULT_ROOT.iterdir()) if (p / 'scene.json').exists()]
GROUPS = [('all', lambda r: r['group'] == 'all'), ('in view', lambda r: r['group'] == 'view'),
          ('in shadow', lambda r: r['group'] == 'shadow'), ('closed rooms', lambda r: r['group'] == 'all' and r['closed'])]


def main(specs):
    made, seeds = [], {}
    try:
        for spec in specs:
            name, location = spec.split('=', 1)
            raw, _, suffix = location.partition(':')
            for scene in SCENES:
                folder = ROOT / raw / (scene + ('-' + suffix if suffix else '')) / 'path' / 'r02048'
                for seed_dir in sorted(folder.glob('[0-9]*')):
                    if not (seed_dir / 'run.json').exists():
                        continue
                    run, data = mr.read_raw(seed_dir)
                    out = mc.DEFAULT_ROOT / scene / name / 'r02048' / seed_dir.name
                    meta = dict(method=name, budget='r02048', seed=int(seed_dir.name), budgetValue=run['budget'], budgetUnit='rays',
                                sceneId=scene, sceneSha256=run['sceneSha256'], code=dict(name=run['method'], sha256=run['codeSha256']),
                                wallSeconds=run['busySeconds'], validSeconds=run['maximumDelaySeconds'], bandLimited=False,
                                wallModel='absorption', notes=name)
                    mc.write_output(out, meta, data['energy'].astype('float32'), None, data['direction'])
                    made.append(out.parent.parent)
                    seeds.setdefault(name, set()).add(int(seed_dir.name))
        rows = mc.score(mc.DEFAULT_ROOT, Path('/tmp/path-suite-score'))['rows']
        for spec in specs:
            name = spec.split('=', 1)[0]
            line = []
            for group, keep in GROUPS:
                sel = [r for r in rows if r['method'] == name and keep(r)]
                values = [mc.aggregate(sel, ('method',), k).get((name,)) for k in ('level', 'EDT', 'T30', 'etc')]
                line.append(group + ' ' + ' / '.join('-' if v is None else f'{v:.2f}' for v in values))
            print(f"{name} (seeds {sorted(seeds.get(name, []))}): " + '; '.join(line))
    finally:
        for folder in set(made):
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == '__main__':
    main(sys.argv[1:])
