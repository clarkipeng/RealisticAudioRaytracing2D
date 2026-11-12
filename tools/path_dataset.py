"""Denoiser pilot data for the Path method (spec: docs/benchmarks/path-tracer-2026-11-12/README.md, "Denoiser data").

    PY=.context/fdtd-teacher/venv/bin/python
    $PY tools/path_dataset.py plan --per-family 8        # scene specs, receivers and Unity jobs
    nice -n 19 $PY tools/path_dataset.py render          # box scenes (guarded Unity, batches of scenes)
    nice -n 19 $PY tools/path_dataset.py church          # Church walk frames (guarded Unity)
    $PY tools/path_dataset.py pack                       # dataset index, targets, noise, hashes, splits
    $PY tools/path_dataset.py verify

One example is one listener pose (a static receiver, or one frame of a moving-listener sequence)
in one scene. Inputs: the Path method's histogram [direction 6][band 8][bin] and its contribution
counts [bin] at the realtime budget, several seeds; every frame of a sequence has its own seed, as
live solves do. Target: the same pose at a high budget, averaged over independent seeds whose spread
measures the target's own noise. Inputs and targets are separate solves; nothing is decimated.
Scenes: tools/dlss_scenes.py families (frozen split rule, two held-out families) and the Church walk.
Unity work goes through tools/method_runs.launch (isolated runner, tools/heavy_guard.py).
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

WORK = ROOT / '.context/path-dataset'
RAW = WORK / 'raw'
NAME = 'path-pilot-v1'
LOW = dict(rays=1024, seeds=[0, 1, 2, 3])       # realtime budget, input seeds
HIGH = dict(rays=16384, seeds=[100, 101])       # target budget, independent seeds
STATIC, FRAMES = 8, 16                          # receivers per scene: static poses, frames of one walk
STEP_SECONDS, SPEED = 1024 / 44100, 1.4         # response step of the renders, walking speed (m/s)
HORIZON, RADIUS = 2.0, .25
CHURCH_TAKE = ROOT / '.context/energy-audio/church/speech-walk/take'
CHURCH_WINDOWS = {'standing': 1.0, 'walking': 7.5, 'outside': 14.5}   # start (s) of 16-frame windows


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ plan

def materials(names):
    """Per-band absorption and the presets' own scattering (no wave ground truth needs zero scattering here)."""
    import method_compare_suite as suite
    table = suite.material_table(names)
    for value in table.values():
        value['scattering'] = value.pop('presetScattering')
    return table


def moving_frames(scene, rng):
    """FRAMES consecutive listener positions walking at SPEED along a grid direction, clear of boxes."""
    points = np.asarray(scene['points'], float).reshape(-1, 3)
    source = np.asarray(scene['source'], float)
    stride = SPEED * STEP_SECONDS
    for _ in range(2000):
        a, b = rng.choice(len(points), 2, replace=False)
        u = points[b] - points[a]
        if np.linalg.norm(u) < 1e-6:
            continue
        u /= np.linalg.norm(u)
        frames = points[a] + np.outer(np.arange(FRAMES) * stride, u)
        if ds.clearance(frames, scene['walls']).min() >= .3 and np.linalg.norm(frames - source, axis=1).min() >= .5:
            return frames
    raise RuntimeError(f"{scene['sceneId']}: no clear walk")


def plan(per_family):
    import method_runs as mr
    scenes, jobs_low, jobs_high = [], [], []
    for family in ds.FAMILIES:
        index, made = 0, 0
        while made < per_family and index < 10 * per_family:
            try:
                base = ds.generate(family, index)
            except RuntimeError:
                index += 1
                continue
            index += 1
            made += 1
            rng = ds.stream(base['sceneId'], 'path-pilot')
            points = np.asarray(base['points'], float).reshape(-1, 3)
            static = points[np.sort(rng.choice(len(points), STATIC, replace=False))]
            moving = moving_frames(base, rng)
            receivers = np.concatenate([static, moving])
            table = materials(base['surfaceMaterials'])
            names = sorted(table)
            scene = dict(base, schema='path-pilot-scene-v1', points=np.round(receivers, 6).ravel().tolist(), materials=table,
                         receiverKinds=['static'] * STATIC + ['moving'] * FRAMES, sequence=dict(frames=FRAMES, stepSeconds=STEP_SECONDS, speed=SPEED),
                         lineOfSight=ds.line_of_sight(base['source'], receivers, base['walls']).astype(int).tolist(), airAbsorption=True)
            for key in ('render', 'fdtd', 'receiverGrid'):
                scene.pop(key, None)
            folder = WORK / 'scenes' / base['sceneId']
            folder.mkdir(parents=True, exist_ok=True)
            raw = (json.dumps(scene, indent=1, sort_keys=True) + '\n').encode()
            (folder / 'scene.json').write_bytes(raw)
            spec = dict(sceneId=base['sceneId'], sceneSha256=hashlib.sha256(raw).hexdigest(), source=base['source'], points=scene['points'],
                        walls=base['walls'], surfaceMaterials=base['surfaceMaterials'], materialNames=names,
                        absorption=[v for n in names for v in table[n]['absorption']],
                        scattering=[v for n in names for v in table[n]['scattering']],
                        transmission=[v for n in names for v in table[n]['transmission']], speedOfSound=base['speedOfSound'], frequencies=[])
            (folder / 'spec.json').write_text(json.dumps(spec) + '\n')
            common = dict(sceneFile=str(folder / 'spec.json'), output=str(RAW), methods=['path'], maximumDelaySeconds=HORIZON,
                          detectionRadius=RADIUS, rouletteFloor=1e-6, receivers=-1, captureReceiver=-1, seedPerReceiver=True, airAbsorption=True)
            jobs_low.append(dict(common, budgets=[LOW['rays']], seeds=LOW['seeds']))
            jobs_high.append(dict(common, budgets=[HIGH['rays']], seeds=HIGH['seeds']))
            scenes.append(base['sceneId'])
    (WORK / 'jobs').mkdir(parents=True, exist_ok=True)
    (WORK / 'jobs' / 'plan.json').write_text(json.dumps(dict(scenes=scenes, low=jobs_low, high=jobs_high), indent=1) + '\n')
    print(f'{len(scenes)} scenes planned in {WORK / "scenes"}')


# ------------------------------------------------------------------ render

def done(scene_id, budget, seeds):
    return all((RAW / scene_id / 'path' / f'r{budget:05d}' / str(s) / 'run.json').exists() for s in seeds)


def render(batch, work_fraction):
    import method_runs as mr
    plan_ = json.loads((WORK / 'jobs' / 'plan.json').read_text())
    pending = []
    for low, high in zip(plan_['low'], plan_['high']):
        scene_id = Path(low['sceneFile']).parent.name
        for job in (low, high):
            if not done(scene_id, job['budgets'][0], job['seeds']):
                pending.append(job)
    print(f'{len(pending)} jobs pending', flush=True)
    for first in range(0, len(pending), batch):
        part = WORK / 'jobs' / f'batch-{first:04d}.json'
        part.write_text(json.dumps(dict(jobs=pending[first:first + batch]), indent=1) + '\n')
        mr.LOGS = WORK / 'logs'
        logs, resources = mr.launch(f'path-dataset-{first:04d}', ['-runTests', '-testPlatform', 'EditMode', '-testFilter',
                                    'TakeChecks.MeasureMethods', '-testResults', str(WORK / 'logs' / f'.batch-{first}.xml')],
                                    dict(ACOUSTIC_METHOD_JOBS=str(part)), work_fraction)


def church(work_fraction):
    import method_runs as mr
    meta = json.loads((CHURCH_TAKE / 'take.json').read_text())
    rate, step = meta['sampleRate'], 1024
    samples = []
    for start in CHURCH_WINDOWS.values():
        first = int(round(start * rate / step)) * step
        samples += [first + f * step for f in range(FRAMES)]
    settings = dict(responseStepSamples=step, speedOfSound=343.0, referenceDistance=1.0, minimumDistance=.1, maximumDelaySeconds=HORIZON,
                    useBvh=True, transport=1, synthesis=1, audioBackend=1, seed=0, diffractionFactor=1.0, airAbsorption=True,
                    fftSize=1024, spectralHop=256, maxBounces=64, method=3, methodRays=LOW['rays'], specularDetectionRadius=RADIUS,
                    methodRouletteFloor=1e-6, methodEdgeProbability=.5)
    out = RAW / 'church-walk' / 'path'
    job = dict(take=str(CHURCH_TAKE), output=str(out), shader='Assets/Shaders/Transport.compute', settings=settings)
    path = WORK / 'jobs' / 'church.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=1) + '\n')
    runs = f"{LOW['rays']}:{LOW['seeds'][0]}:{len(LOW['seeds'])},{HIGH['rays']}:{HIGH['seeds'][0]}:{len(HIGH['seeds'])}"
    mr.LOGS = WORK / 'logs'
    mr.launch('path-dataset-church', ['-runTests', '-testPlatform', 'EditMode', '-testFilter', 'TakeChecks.MeasurePathTake',
                                      '-testResults', str(WORK / 'logs' / '.church.xml')],
              dict(ACOUSTIC_PATH_TAKE=str(path), ACOUSTIC_PATH_SAMPLES=','.join(map(str, samples)), ACOUSTIC_PATH_RUNS=runs), work_fraction)


# ------------------------------------------------------------------ pack

def read_run(folder):
    run = json.loads((folder / 'run.json').read_text())
    B = run['histogramBins'] if 'histogramBins' in run else run['bins']
    R = run['receivers'] if 'receivers' in run else run['frames']
    h = np.fromfile(folder / 'histogram.f32', '<f4').reshape(R, 6, 8, B)
    c = np.fromfile(folder / 'counts.u32', '<u4').reshape(R, B)
    seconds = run.get('busySeconds') or run.get('wallSeconds')
    return run, h, c, np.fromfile(folder / 'bin-edges.f32', '<f4'), seconds


def descriptor(path, dtype, shape, axes):
    return dict(file=str(path.relative_to(WORK)), dtype=dtype, shape=list(shape), axes=axes, order='C', sha256=sha256(path), bytes=path.stat().st_size)


def pack():
    out = WORK / 'datasets' / NAME
    (out / 'derived').mkdir(parents=True, exist_ok=True)
    bundles, examples, costs, rejected = [], [], {}, []
    scene_ids = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    edges_ref = None
    for scene_id in scene_ids:
        church = scene_id == 'church-walk'
        if church:
            family, split, group, kinds = 'church', 'test', 'church', None
        else:
            scene = json.loads((WORK / 'scenes' / scene_id / 'scene.json').read_text())
            family, split, group, kinds = scene['family'], scene['split'], scene['splitGroup'], scene['receiverKinds']
        base = RAW / scene_id / 'path'
        try:
            low = [read_run(base / f"r{LOW['rays']:05d}" / str(s)) for s in LOW['seeds']]
            high = [read_run(base / f"r{HIGH['rays']:05d}" / str(s)) for s in HIGH['seeds']]
        except FileNotFoundError as missing:
            rejected.append(dict(scene=scene_id, reason=f'missing run: {missing.filename}'))
            continue
        edges = low[0][3]
        edges_ref = edges if edges_ref is None else edges_ref
        if not np.array_equal(edges, edges_ref):
            rejected.append(dict(scene=scene_id, reason='bin edges differ'))
            continue
        target = np.stack([r[1] for r in high]).astype(np.float64)
        mean, var = target.mean(0), target.var(0, ddof=1)
        if not (np.isfinite(mean).all() and np.isfinite(var).all()):
            rejected.append(dict(scene=scene_id, reason='non-finite target'))
            continue
        folder = out / 'derived' / scene_id
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, array, dtype in (('target-mean.f32', mean, '<f4'), ('target-variance.f32', var, '<f4'),
                                   ('target-counts.u32', np.stack([r[2] for r in high]).sum(0), '<u4')):
            array.astype(dtype).tofile(folder / name)
        R = mean.shape[0]
        files['targetMean'] = descriptor(folder / 'target-mean.f32', 'float32', mean.shape, ['pose', 'direction', 'band', 'bin'])
        files['targetVariance'] = descriptor(folder / 'target-variance.f32', 'float32', var.shape, ['pose', 'direction', 'band', 'bin'])
        files['targetCounts'] = descriptor(folder / 'target-counts.u32', 'uint32', (R, mean.shape[-1]), ['pose', 'bin'])
        files['inputs'] = [dict(seed=s, histogram=descriptor(base / f"r{LOW['rays']:05d}" / str(s) / 'histogram.f32', 'float32', r[1].shape, ['pose', 'direction', 'band', 'bin']),
                                counts=descriptor(base / f"r{LOW['rays']:05d}" / str(s) / 'counts.u32', 'uint32', r[2].shape, ['pose', 'bin']))
                           for s, r in zip(LOW['seeds'], low)]
        # Noise: energy-weighted relative standard error per bin, and the band level's standard error (dB).
        band_mean = mean.sum(1)                                     # [pose, band, bin]
        band_var = var.sum(1)
        se_rel = np.sqrt(band_var.sum() / len(HIGH['seeds'])) / max(band_mean.sum(), 1e-30)
        weight = band_mean / max(band_mean.sum(), 1e-30)
        bin_rel = np.sqrt(band_var / len(HIGH['seeds'])) / np.maximum(band_mean, 1e-30)
        target_bin_noise = float((weight * np.minimum(bin_rel, 10)).sum())
        level_se = np.sqrt(var.sum((1, 3)) / len(HIGH['seeds'])) / np.maximum(mean.sum((1, 3)), 1e-30)
        inputs = np.stack([r[1] for r in low]).astype(np.float64).sum(2)   # [seed, pose, band, bin]
        input_rel = np.sqrt(((inputs - band_mean[None]) ** 2).mean(0)) / np.maximum(band_mean, 1e-30)
        input_bin_noise = float((weight * np.minimum(input_rel, 10)).sum())
        cost = dict(lowSeconds=float(sum(r[4] for r in low)), highSeconds=float(sum(r[4] for r in high)))
        costs[scene_id] = cost
        bundle = dict(scene=scene_id, family=family, split=split, splitGroup=group, poses=R, files=files, cost=cost,
                      quality=dict(targetBinNoise=target_bin_noise, inputBinNoise=input_bin_noise, targetRelativeStandardError=float(se_rel),
                                   targetLevelStandardErrorDb=float(10 * np.log10(1 + np.median(level_se)))))
        if not church:
            bundle['sceneSha256'] = sha256(WORK / 'scenes' / scene_id / 'scene.json')
        else:
            bundle['take'] = str(CHURCH_TAKE.relative_to(ROOT))
            bundle['poses_file'] = descriptor(base / f"r{LOW['rays']:05d}" / str(LOW['seeds'][0]) / 'poses.f32', 'float32', (R, 6), ['pose', 'listener xyz, source xyz'])
        bundles.append(bundle)
        for p in range(R):
            if church:
                window = p // FRAMES
                kind, sequence, frame = ('static' if list(CHURCH_WINDOWS)[window] == 'standing' else 'moving'), f'{scene_id}/{list(CHURCH_WINDOWS)[window]}', p % FRAMES
            else:
                kind = kinds[p]
                sequence, frame = (f'{scene_id}/walk', p - STATIC) if kind == 'moving' else (None, None)
            examples.append(dict(scene=scene_id, split=split, pose=p, kind=kind, sequence=sequence, frame=frame))
    edges_path = out / 'bin-edges.f32'
    np.asarray(edges_ref, '<f4').tofile(edges_path)
    with open(out / 'bundles.jsonl', 'w') as f:
        for b in bundles:
            f.write(json.dumps(b, sort_keys=True) + '\n')
    with open(out / 'examples.jsonl', 'w') as f:
        for e in examples:
            f.write(json.dumps(e, sort_keys=True) + '\n')
    splits = {}
    for b in bundles:
        splits.setdefault(b['split'], []).append(b['family'])
    summary = dict(schema='path-pilot-dataset-v1', name=NAME, generator=ds.GENERATOR_VERSION, horizonSeconds=HORIZON, detectionRadius=RADIUS,
                   inputs=dict(rays=LOW['rays'], seeds=LOW['seeds']), targets=dict(rays=HIGH['rays'], seeds=HIGH['seeds']),
                   poses=dict(static=STATIC, movingFrames=FRAMES, stepSeconds=STEP_SECONDS, speed=SPEED),
                   binEdges=descriptor(edges_path, 'float32', (len(edges_ref),), ['edge seconds']),
                   splitRule='dlss_scenes.split_for (frozen); corridor = validation and canopy = test are held-out families; the Church is test',
                   scenes=len(bundles), examples=len(examples), splits={k: dict(scenes=len(v), families=sorted(set(v))) for k, v in splits.items()},
                   rejected=rejected,
                   cost=dict(perScene=costs, meanLowSeconds=float(np.mean([c['lowSeconds'] for c in costs.values()])) if costs else None,
                             meanHighSeconds=float(np.mean([c['highSeconds'] for c in costs.values()])) if costs else None),
                   noise=dict(targetBinNoiseMedian=float(np.median([b['quality']['targetBinNoise'] for b in bundles])) if bundles else None,
                              inputBinNoiseMedian=float(np.median([b['quality']['inputBinNoise'] for b in bundles])) if bundles else None),
                   bundlesSha256=sha256(out / 'bundles.jsonl'), examplesSha256=sha256(out / 'examples.jsonl'))
    (out / 'dataset.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in ('scenes', 'examples', 'splits', 'noise', 'rejected')}, indent=1))
    print('identity', sha256(out / 'dataset.json'))


def verify():
    out = WORK / 'datasets' / NAME
    summary = json.loads((out / 'dataset.json').read_text())
    assert sha256(out / 'bundles.jsonl') == summary['bundlesSha256'] and sha256(out / 'examples.jsonl') == summary['examplesSha256']
    groups, checked = {}, 0
    for line in open(out / 'bundles.jsonl'):
        b = json.loads(line)
        descriptors = [b['files']['targetMean'], b['files']['targetVariance'], b['files']['targetCounts']]
        descriptors += [d for i in b['files']['inputs'] for d in (i['histogram'], i['counts'])]
        for d in descriptors:
            path = WORK / d['file']
            assert path.stat().st_size == d['bytes'] and sha256(path) == d['sha256'], d['file']
            checked += 1
        groups.setdefault(b['splitGroup'], set()).add(b['split'])
        if b['family'] == 'corridor':
            assert b['split'] == 'validation'
        if b['family'] in ('canopy', 'church'):
            assert b['split'] == 'test'
    assert all(len(s) == 1 for s in groups.values()), groups
    print(f'verified {checked} payloads, {len(groups)} split groups, each in one split')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan'); p.add_argument('--per-family', type=int, default=8)
    r = sub.add_parser('render'); r.add_argument('--batch', type=int, default=24); r.add_argument('--work-fraction', type=float, default=.5)
    c = sub.add_parser('church'); c.add_argument('--work-fraction', type=float, default=.5)
    sub.add_parser('pack'); sub.add_parser('verify')
    args = parser.parse_args(argv)
    if args.command == 'plan':
        plan(args.per_family)
    elif args.command == 'render':
        render(args.batch, args.work_fraction)
    elif args.command == 'church':
        church(args.work_fraction)
    elif args.command == 'pack':
        pack()
    else:
        verify()


if __name__ == '__main__':
    main()
