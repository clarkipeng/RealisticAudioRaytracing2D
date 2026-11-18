"""Data for the persistent-ray denoiser (docs/benchmarks/path-denoise-2026-11-12, part 25).

    PY=.context/fdtd-teacher/venv/bin/python
    $PY tools/path_persistent_dataset.py plan            # scenes, listener tracks and the Unity items
    nice -n 19 $PY tools/path_persistent_dataset.py render   # guarded Unity (MeasurePathPersistentSequences)
    $PY tools/path_persistent_dataset.py pack            # one .npz per track, index.json

Every track is replayed update by update (1,024 samples at 44.1 kHz) with persistent rays under population
control. Inputs: the input population (each scene's 23 ms budget, calibrated in Unity) with two seeds; targets:
eight times that population with two independent seeds, along the same updates, so a moving listener's target
is the same retarded-time quantity. Tracks per generated scene: two static listeners and one walking at
1.4 m/s; the Church: its take's own walk, recorded in three windows (standing, walking, outside).
Scenes: three per family of tools/dlss_scenes.py (frozen split rule: corridor validation; canopy, wide
doorways and the Church test).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_scenes as ds  # noqa: E402
import path_dataset as pd  # noqa: E402

WORK = ROOT / '.context/path-dataset-persistent'
STEP = 1024 / 44100
HORIZON = 2.0
WARM, RECORD = int(math.ceil(HORIZON / STEP)) + 2, 64
SPEED = 1.4
PER_FAMILY = 3
SEEDS, HIGH_SEEDS, HIGH_FACTOR = [0, 1], [100, 101], 8
CHURCH_JOB = ROOT / '.context/energy-audio/listen/path-realtime.job.json'
CHURCH_POPULATION = 11000          # the Church's 23 ms point with shortcuts (part 20)
CHURCH_WINDOWS = {'standing': 1.0, 'walking': 7.5, 'outside': 14.5}


def walk(scene, rng, updates):
    """`updates` listener positions walking at SPEED back and forth along a straight segment (at least 1.5 m,
    at most the whole walk) that keeps clear of walls and the source."""
    points = np.asarray(scene['points'], float).reshape(-1, 3)
    source = np.asarray(scene['source'], float)
    distance = np.arange(updates) * SPEED * STEP
    for _ in range(4000):
        a = points[rng.integers(len(points))]
        u = rng.normal(size=3); u[1] = 0; u /= np.linalg.norm(u)
        line = a + np.outer(np.linspace(0, distance[-1], 200), u)
        bad = (ds.clearance(line, scene['walls']) < .3) | (np.linalg.norm(line - source, axis=1) < .5)
        length = distance[-1] * (np.argmax(bad) / 199 if bad.any() else 1)
        if length >= 1.5:
            s = np.abs((distance + length) % (2 * length) - length)
            return a + np.outer(s, u)
    raise RuntimeError(f"{scene['sceneId']}: no clear walk")


def pick(family, tries=30):
    """PER_FAMILY scenes of a family; where the split rule holds part of a family out (wide doorways, crowded
    rooms), one of them from the held-out part."""
    scenes = []
    for i in range(tries):
        try:
            scenes.append(ds.generate(family, i))
        except RuntimeError:
            continue
    chosen = [scenes[0]]
    other = [b for b in scenes if b['split'] != chosen[0]['split']]
    chosen += (other or [b for b in scenes if b is not chosen[0]])[:PER_FAMILY - 1]
    return chosen


def plan():
    items, index = [], []
    updates = WARM + RECORD
    for family in ds.FAMILIES:
        for base in pick(family):
            rng = np.random.default_rng(ds.stable_seed(base['sceneId'], 'path-persistent'))
            points = np.asarray(base['points'], float).reshape(-1, 3)
            static = points[rng.choice(len(points), 2, replace=False)]
            tracks = [('static-0', np.repeat(static[:1], updates, 0)), ('static-1', np.repeat(static[1:], updates, 0)), ('walk', walk(base, rng, updates))]
            table = pd.materials(base['surfaceMaterials'])
            names = sorted(table)
            folder = WORK / 'scenes' / base['sceneId']
            folder.mkdir(parents=True, exist_ok=True)
            spec = dict(sceneId=base['sceneId'], sceneSha256=hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest(), source=base['source'],
                        points=np.round(static, 6).ravel().tolist(), walls=base['walls'], surfaceMaterials=base['surfaceMaterials'], materialNames=names,
                        absorption=[v for n in names for v in table[n]['absorption']], scattering=[v for n in names for v in table[n]['scattering']],
                        transmission=[v for n in names for v in table[n]['transmission']], speedOfSound=base['speedOfSound'], frequencies=[])
            (folder / 'spec.json').write_text(json.dumps(spec) + '\n')
            for name, track in tracks:
                out = WORK / 'raw' / base['sceneId'] / name
                items.append(dict(kind='scene', spec=str(folder / 'spec.json'), job='', output=str(out), track=np.round(track, 6).ravel().tolist(),
                                  samples=[], warm=WARM, record=RECORD, population=0, highFactor=HIGH_FACTOR, seeds=SEEDS, highSeeds=HIGH_SEEDS))
                index.append(dict(scene=base['sceneId'], family=family, split=base['split'], track=name, moving=name == 'walk', raw=str(out)))
    meta = json.loads((pd.CHURCH_TAKE / 'take.json').read_text())
    step = 1024
    warm = int(math.ceil(json.loads(CHURCH_JOB.read_text())['settings']['maximumDelaySeconds'] / STEP)) + 2   # the job's 6 s horizon
    for name, start in CHURCH_WINDOWS.items():
        first = int(round(start * meta['sampleRate'] / step))
        samples = [max(0, first - warm + u) * step for u in range(warm + RECORD)]   # standing still before the take starts
        out = WORK / 'raw' / 'church' / name
        items.append(dict(kind='take', spec='', job=str(CHURCH_JOB), output=str(out), track=[], samples=samples, warm=warm, record=RECORD,
                          population=CHURCH_POPULATION, highFactor=HIGH_FACTOR, seeds=SEEDS, highSeeds=HIGH_SEEDS))
        index.append(dict(scene='church', family='church', split='test', track=name, moving=name != 'standing', raw=str(out)))
    (WORK / 'items.json').write_text(json.dumps(dict(items=items)) + '\n')
    (WORK / 'plan.json').write_text(json.dumps(index, indent=1) + '\n')
    print(f'{len(items)} tracks in {len({i["scene"] for i in index})} scenes')


def render():
    import method_runs as mr
    mr.LOGS = WORK / 'logs'
    mr.LOGS.mkdir(parents=True, exist_ok=True)
    mr.launch('path-persistent-dataset', ['-runTests', '-testPlatform', 'EditMode', '-testFilter', 'TakeChecks.MeasurePathPersistentSequences',
                                          '-testResults', str(WORK / 'logs' / '.dataset.xml')], dict(ACOUSTIC_PERSISTENT_DATASET=str(WORK / 'items.json')))


def pack():
    index = json.loads((WORK / 'plan.json').read_text())
    out = []
    for e in index:
        raw = Path(e['raw'])
        if not (raw / 'done').exists():
            continue
        runs = {}
        for d in sorted(raw.glob('*-*')):
            r = json.loads((d / 'run.json').read_text())
            B = r['bins']
            h = np.fromfile(d / 'histogram.f32', '<f4').reshape(r['updates'], 48, B)
            c = np.fromfile(d / 'counts.u32', '<u4').reshape(r['updates'], B).astype(np.float32)
            runs[d.name] = (h, c, r)
        low = sorted(k for k in runs if k.startswith('low')); high = sorted(k for k in runs if k.startswith('high'))
        edges = np.fromfile(raw / low[0] / 'bin-edges.f32', '<f4')
        file = WORK / 'packed' / f"{e['scene']}-{e['track']}.npz"
        file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(file, low=np.stack([runs[k][0] for k in low]), lowCounts=np.stack([runs[k][1] for k in low]),
                 high=np.stack([runs[k][0] for k in high]), edges=edges)
        out.append(dict(e, file=str(file), population=runs[low[0]][2]['population'], newRays=runs[low[0]][2]['newRays'],
                        updateMs=runs[low[0]][2]['updateMs'], highPopulation=runs[high[0]][2]['population'], highUpdateMs=runs[high[0]][2]['updateMs']))
    (WORK / 'index.json').write_text(json.dumps(out, indent=1) + '\n')
    print(f'{len(out)} tracks packed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['plan', 'render', 'pack'])
    a = p.parse_args()
    dict(plan=plan, render=render, pack=pack)[a.command]()
