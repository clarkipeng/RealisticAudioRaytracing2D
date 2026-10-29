"""Audio-DLSS denoiser dataset: pack rendered scenes, verify, load and check identity.

Layout under the data root (default .context/audio-dlss-data), following
docs/simulation-contracts.md section 9:

    scenes/<scene-id>/scene.json        one description for Unity and FDTD
    scenes/<scene-id>/tracer/           Unity run payloads + run.json (never copied)
    scenes/<scene-id>/fdtd/             FDTD wave payloads + run.json (subset)
    datasets/<name>/dataset.json        identity = SHA-256 of this file
    datasets/<name>/generation.json     resolved recipe and quality configuration
    datasets/<name>/bundles.jsonl       one bundle per scene: hashes, split, descriptors
    datasets/<name>/examples.jsonl      one row per (scene, low variant, seed, receiver)
    datasets/<name>/derived/<scene-id>/ label mean/variance/validity and context

Every numeric descriptor declares file, dtype, shape, axes, units, C order,
byte_offset, byte_length and whole-file SHA-256; paths are relative to
artifact_root, itself relative to the dataset directory.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY tools/dlss_dataset.py pack denoise-pilot-v1
    $PY tools/dlss_dataset.py verify .context/audio-dlss-data/datasets/denoise-pilot-v1
    $PY tools/dlss_dataset.py identity
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

DATA = ROOT / '.context/audio-dlss-data'
DATASET_SCHEMA = 'acoustic-training-dataset-v1'
EXPERIMENT = 'ray-denoise-complex-v1'
CONTEXT_SCHEMA = 'static-geometry-v1'
CONTEXT_COLUMNS = ['sourceMinusListenerX', 'sourceMinusListenerY', 'sourceMinusListenerZ', 'distance',
                   'lineOfSight', 'directVisible']
DTYPES = {'<f4': 4, '<i8': 8, '|u1': 1}
FAMILIES = ['direct', 'R', 'D', 'DD', 'RD', 'DR', 'RR', 'walk']
# Primary label: the converged early field. The residual walk is kept as an input
# buffer and an auxiliary total label, but it does not converge at affordable
# budgets (docs/benchmarks/denoise-pilot-2026-10-29), so it is not the target.
TARGET_FAMILIES = ['direct', 'R', 'D', 'DD', 'RD', 'DR', 'RR']
UNITS = 'pressure relative to a unit-amplitude point source at 1 m; time dependence exp(+i*w*t)'
# Label acceptance, fixed with the protocol before the pilot (generation.json).
ACCEPTANCE = dict(relativeStandardError=.1, nearZeroFloor=.01, minimumValidFraction=.9,
                  definition='entry valid if sqrt(var/K) <= relativeStandardError*|mean| + '
                             'nearZeroFloor*rms_receivers(|mean|) at that frequency; '
                             'scene accepted if all values are finite and the valid fraction >= minimumValidFraction')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def descriptor(base, path, dtype, shape, axes, units):
    path = Path(path)
    size = int(np.prod(shape)) * DTYPES[dtype]
    if path.stat().st_size != size:
        raise ValueError(f'{path}: {path.stat().st_size} bytes, expected {size} for {shape} {dtype}.')
    return dict(file=str(path.relative_to(base)), dtype=dtype, shape=[int(s) for s in shape], axes=axes, units=units,
                order='C', byte_offset=0, byte_length=size, sha256=sha256(path))


def read(base, d, check=False):
    """Read-only memory map of a descriptor (optionally hash-checked first)."""
    path = Path(base) / d['file']
    if d['byte_length'] != int(np.prod(d['shape'])) * DTYPES[d['dtype']]:
        raise ValueError(f'{path}: descriptor size does not match its shape.')
    if d['byte_offset'] + d['byte_length'] > path.stat().st_size:
        raise ValueError(f'{path}: descriptor addresses past the end of the file.')
    if check and sha256(path) != d['sha256']:
        raise ValueError(f'{path}: SHA-256 differs from the descriptor.')
    return np.memmap(path, dtype=np.dtype(d['dtype']), mode='r', offset=d['byte_offset'], shape=tuple(d['shape']))


def complex_of(array):
    return array[..., 0].astype(np.float64) + 1j * array[..., 1].astype(np.float64)


def tracer_descriptors(base, folder, run, variant='tracer'):
    families = run['families']
    points, freqs = run['points'], len(run['frequencies'])
    out = {}
    for budget in run['budgets']:
        name, seeds = budget['name'], len(budget['seeds'])
        t = folder / variant
        out[name] = dict(
            raysPerBin=budget['raysPerBin'], completeRouteSamples=budget['completeRouteSamples'], seeds=budget['seeds'],
            responses=descriptor(base, t / f'responses-{name}.f32', '<f4', [seeds, points, freqs, len(families), 2],
                                 ['seed', 'receiver', 'frequency', 'family', 'realImag'], UNITS),
            counts=descriptor(base, t / f'counts-{name}.i64', '<i8', [seeds, points, freqs, len(families)],
                              ['seed', 'receiver', 'frequency', 'family'], 'nonzero path arrivals'),
            peaks=descriptor(base, t / f'peaks-{name}.f32', '<f4', [seeds, points, freqs, len(families)],
                             ['seed', 'receiver', 'frequency', 'family'], 'largest single-path |gain|'),
            seconds=descriptor(base, t / f'seconds-{name}.f32', '<f4', [seeds, points], ['seed', 'receiver'],
                               'seconds, synchronous transport solve and path readback'))
    return out


def label_statistics(responses, families=None):
    """Mean and unbiased complex sample variance over K label seeds (float64)."""
    values = complex_of(np.asarray(responses))                      # [K, P, Q, F]
    if families is not None:
        values = values[..., [FAMILIES.index(f) for f in families]]
    total = values.sum(axis=3)                                      # [K, P, Q]
    k = total.shape[0]
    if k < 2:
        raise ValueError('Label statistics need K >= 2 independent seeds.')
    mean = total.mean(0)
    variance = (np.abs(total - mean) ** 2).sum(0) / (k - 1)
    return mean, variance, k


def acceptance(mean, variance, k):
    se = np.sqrt(variance / k)
    floor = ACCEPTANCE['nearZeroFloor'] * np.sqrt(np.mean(np.abs(mean) ** 2, axis=0, keepdims=True))
    valid = se <= ACCEPTANCE['relativeStandardError'] * np.abs(mean) + floor
    finite = bool(np.all(np.isfinite(mean)) and np.all(np.isfinite(variance)))
    fraction = float(valid.mean())
    reasons = [] if finite else ['non-finite label']
    if fraction < ACCEPTANCE['minimumValidFraction']:
        reasons.append(f'valid label fraction {fraction:.3f} < {ACCEPTANCE["minimumValidFraction"]}')
    return valid, dict(accepted=not reasons, reasons=reasons, validFraction=fraction,
                       relativeStandardError=float(np.sqrt(np.sum(se ** 2) / np.sum(np.abs(mean) ** 2))))


def pack(data, name, label='label'):
    """Write a complete dataset over every rendered scene under the data root."""
    target = data / 'datasets' / name
    if (target / 'dataset.json').exists():
        raise FileExistsError(f'{target} exists; datasets are immutable once complete.')
    (target / 'derived').mkdir(parents=True, exist_ok=True)
    base = data
    bundles, examples, split_groups, rejected = [], [], {}, []
    scenes, frequencies, low_names = [], None, None
    for folder in sorted(p.parent for p in (data / 'scenes').glob('*/scene.json')):
        scene_path = folder / 'scene.json'
        scene = json.loads(scene_path.read_text())
        scenes.append(dict(sceneId=scene['sceneId'], sha256=sha256(scene_path), rendered=(folder / 'tracer').exists()))
        if not (folder / 'tracer/run.json').exists():
            continue
        run = json.loads((folder / 'tracer/run.json').read_text())
        if run['sceneSha256'] != scenes[-1]['sha256']:
            raise ValueError(f'{folder.name}: tracer run was rendered from different scene bytes.')
        frequencies = frequencies or run['frequencies']
        if run['frequencies'] != frequencies:
            raise ValueError('All bundles must share one frequency grid.')
        budgets = tracer_descriptors(base, folder, run)
        names = [b for b in budgets if b != label]
        low_names = low_names or names
        if names != low_names:
            raise ValueError('All bundles must share the same low variants.')
        if run['families'] != FAMILIES:
            raise ValueError(f'{folder.name}: unexpected family order {run["families"]}.')
        label_responses = read(base, budgets[label]['responses'])
        mean, variance, k = label_statistics(label_responses, TARGET_FAMILIES)
        total_mean, total_variance, _ = label_statistics(label_responses)
        valid, quality = acceptance(mean, variance, k)
        quality['totalRelativeStandardError'] = acceptance(total_mean, total_variance, k)[1]['relativeStandardError']
        derived = target / 'derived' / scene['sceneId']
        derived.mkdir(exist_ok=True)
        np.stack([mean.real, mean.imag], -1).astype('<f4').tofile(derived / 'target-mean.f32')
        variance.astype('<f4').tofile(derived / 'target-variance.f32')
        valid.astype('|u1').tofile(derived / 'target-validity.u8')
        np.stack([total_mean.real, total_mean.imag], -1).astype('<f4').tofile(derived / 'total-target-mean.f32')
        total_variance.astype('<f4').tofile(derived / 'total-target-variance.f32')
        points = np.array(scene['points']).reshape(-1, 3)
        delta = np.asarray(scene['source']) - points
        visible = np.fromfile(folder / 'tracer/direct-visible.u8', np.uint8)
        context = np.c_[delta, np.linalg.norm(delta, axis=1), scene['lineOfSight'], visible].astype('<f4')
        context.tofile(derived / 'context.f32')
        P, Q = mean.shape
        descriptors = dict(budgets=budgets, label=label,
                           target_mean=descriptor(base, derived / 'target-mean.f32', '<f4', [P, Q, 2],
                                                  ['receiver', 'frequency', 'realImag'], UNITS),
                           target_variance=descriptor(base, derived / 'target-variance.f32', '<f4', [P, Q],
                                                      ['receiver', 'frequency'], 'squared pressure; standard error sqrt(v/K)'),
                           target_validity=descriptor(base, derived / 'target-validity.u8', '|u1', [P, Q],
                                                      ['receiver', 'frequency'], 'mask'),
                           total_target_mean=descriptor(base, derived / 'total-target-mean.f32', '<f4', [P, Q, 2],
                                                        ['receiver', 'frequency', 'realImag'], UNITS),
                           total_target_variance=descriptor(base, derived / 'total-target-variance.f32', '<f4', [P, Q],
                                                            ['receiver', 'frequency'], 'squared pressure'),
                           context=descriptor(base, derived / 'context.f32', '<f4', [P, len(CONTEXT_COLUMNS)],
                                              ['receiver', 'feature'], 'metres / flags'),
                           direct_visible=descriptor(base, folder / 'tracer/direct-visible.u8', '|u1', [P],
                                                     ['receiver'], 'mask'))
        rigid = None
        if (folder / 'tracer-rigid/run.json').exists():
            rigid_run = json.loads((folder / 'tracer-rigid/run.json').read_text())
            if rigid_run['sceneSha256'] != scenes[-1]['sha256'] or rigid_run['frequencies'] != frequencies:
                raise ValueError(f'{folder.name}: rigid tracer run identity differs.')
            rigid_budgets = tracer_descriptors(base, folder, rigid_run, 'tracer-rigid')
            r_mean, r_variance, r_k = label_statistics(read(base, rigid_budgets[label]['responses']), TARGET_FAMILIES)
            np.stack([r_mean.real, r_mean.imag], -1).astype('<f4').tofile(derived / 'rigid-target-mean.f32')
            r_variance.astype('<f4').tofile(derived / 'rigid-target-variance.f32')
            rigid = dict(run=dict(file=str((folder / 'tracer-rigid/run.json').relative_to(base)),
                                  sha256=sha256(folder / 'tracer-rigid/run.json')),
                         budgets=rigid_budgets, label_seeds=r_k,
                         target_mean=descriptor(base, derived / 'rigid-target-mean.f32', '<f4', [P, Q, 2],
                                                ['receiver', 'frequency', 'realImag'], UNITS),
                         target_variance=descriptor(base, derived / 'rigid-target-variance.f32', '<f4', [P, Q],
                                                    ['receiver', 'frequency'], 'squared pressure; standard error sqrt(v/K)'))
        wave = None
        if (folder / 'fdtd/run.json').exists():
            fd = json.loads((folder / 'fdtd/run.json').read_text())
            if fd['sceneSha256'] != scenes[-1]['sha256']:
                raise ValueError(f'{folder.name}: FDTD run was solved from different scene bytes.')
            wave = dict(run=dict(file=str((folder / 'fdtd/run.json').relative_to(base)), sha256=sha256(folder / 'fdtd/run.json')),
                        frequencies=fd['frequencies'], spacing=fd['spacing'], truncationChange075=fd['truncationChange075'],
                        wave=descriptor(base, folder / 'fdtd/wave.f32', '<f4', [P, len(fd['frequencies']), 2],
                                        ['receiver', 'frequency', 'realImag'], UNITS))
        group = scene['splitGroup']
        if split_groups.setdefault(group, scene['split']) != scene['split']:
            raise ValueError(f'Split group {group} spans several splits.')
        bundle = dict(bundle_id=scene['sceneId'], family=scene['family'], split=scene['split'], split_group=group,
                      parameters=scene['parameters'], closed=scene['closed'], receivers=P,
                      scene=dict(file=str(scene_path.relative_to(base)), sha256=scenes[-1]['sha256']),
                      tracer_run=dict(file=str((folder / 'tracer/run.json').relative_to(base)),
                                      sha256=sha256(folder / 'tracer/run.json')),
                      tracer_provenance=dict(file=str((folder / 'tracer/provenance.json').relative_to(base)),
                                             sha256=sha256(folder / 'tracer/provenance.json'))
                      if (folder / 'tracer/provenance.json').exists() else None,
                      boundary=dict(file=str((folder / 'tracer/boundary.json').relative_to(base)),
                                    sha256=sha256(folder / 'tracer/boundary.json')),
                      label_seeds=k, quality=quality, descriptors=descriptors, rigid=rigid, wave=wave)
        bundles.append(bundle)
        if not quality['accepted']:
            rejected.append(dict(bundle_id=scene['sceneId'], family=scene['family'], reasons=quality['reasons']))
            continue
        for variant in low_names:
            for s in range(len(budgets[variant]['seeds'])):
                for p in range(P):
                    examples.append(dict(example_id=f'{scene["sceneId"]}-{variant}-s{s}-p{p}', bundle_id=scene['sceneId'],
                                         low_variant=variant, low_seed=s, pair_index=p))
    write_lines(target / 'bundles.jsonl', bundles)
    write_lines(target / 'examples.jsonl', examples)
    generation = dict(generator=ds.GENERATOR_VERSION, generatorSha256=sha256(ROOT / 'tools/dlss_scenes.py'),
                      packerSha256=sha256(Path(__file__)), labelBudget=ds.LABEL_BUDGET, labelSeeds=ds.LABEL_SEEDS,
                      inputFractions=list(ds.INPUT_FRACTIONS), inputSeeds=ds.INPUT_SEEDS, acceptance=ACCEPTANCE,
                      targetFamilies=TARGET_FAMILIES,
                      families=list(ds.FAMILIES), splitRule=ds.split_for.__doc__.strip(), scenes=scenes,
                      rejected=rejected, rejectedByFamily={f: sum(r['family'] == f for r in rejected) for f in ds.FAMILIES})
    (target / 'generation.json').write_text(json.dumps(generation, indent=1) + '\n')
    manifest = dict(
        schema=DATASET_SCHEMA, experiment=EXPERIMENT, quantity='ComplexPressure', dimension='Spatial3D',
        artifact_root='../..', sample_rate=ds.SAMPLE_RATE, history_frames=1,
        grid=dict(frequencies_hz=frequencies, fft_size=ds.FFT_SIZE, units=UNITS, origin='source emission time',
                  direct_inclusion='included as family "direct"', families=['direct', 'R', 'D', 'DD', 'RD', 'DR', 'RR', 'walk']),
        context_schema=CONTEXT_SCHEMA, context_columns=CONTEXT_COLUMNS, query_schema='none',
        availability=dict(mode='static-offline-v1', note='single static solve per example; no temporal history'),
        label=dict(budget=label, statistic='mean of K seeds', uncertainty='unbiased complex sample variance',
                   target_families=TARGET_FAMILIES, auxiliary='total_target_mean includes the residual walk'),
        low_variants=low_names, split_groups=split_groups,
        bundles=dict(file='bundles.jsonl', rows=len(bundles), sha256=sha256(target / 'bundles.jsonl')),
        examples=dict(file='examples.jsonl', rows=len(examples), sha256=sha256(target / 'examples.jsonl')),
        generation=dict(file='generation.json', sha256=sha256(target / 'generation.json')),
        complete=True)
    (target / 'dataset.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return target


def write_lines(path, rows):
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':'), sort_keys=True) + '\n')


def read_lines(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def open_dataset(target, check=True):
    """Manifest, bundles and artifact base, after hash checks on the index files."""
    target = Path(target)
    manifest = json.loads((target / 'dataset.json').read_text())
    if manifest.get('schema') != DATASET_SCHEMA or manifest.get('experiment') != EXPERIMENT:
        raise ValueError(f'Unsupported dataset {manifest.get("schema")}/{manifest.get("experiment")}.')
    if not manifest.get('complete'):
        raise ValueError('Dataset is incomplete.')
    for key in ['bundles', 'examples', 'generation']:
        if check and sha256(target / manifest[key]['file']) != manifest[key]['sha256']:
            raise ValueError(f'{key} file hash differs from dataset.json.')
    bundles = read_lines(target / manifest['bundles']['file'])
    return manifest, bundles, (target / manifest['artifact_root']).resolve()


def load_bundle(base, bundle, check=False):
    """Complex arrays of one bundle: inputs per variant, label runs and statistics."""
    d = bundle['descriptors']
    out = dict(inputs={}, counts={}, peaks={}, seconds={})
    for name, budget in d['budgets'].items():
        out['inputs'][name] = complex_of(read(base, budget['responses'], check))   # [S, P, Q, F]
        out['counts'][name] = np.asarray(read(base, budget['counts'], check))
        out['peaks'][name] = np.asarray(read(base, budget['peaks'], check))
        out['seconds'][name] = np.asarray(read(base, budget['seconds'], check))
    out['label_family_runs'] = out['inputs'].pop(d['label'])                        # [K, P, Q, F]
    out['label_runs'] = out['label_family_runs'].sum(axis=3)                        # [K, P, Q]
    out['target'] = complex_of(read(base, d['target_mean'], check))
    out['variance'] = np.asarray(read(base, d['target_variance'], check), np.float64)
    out['valid'] = np.asarray(read(base, d['target_validity'], check)).astype(bool)
    out['context'] = np.asarray(read(base, d['context'], check))
    if bundle.get('wave'):
        out['wave'] = complex_of(read(base, bundle['wave']['wave'], check))
    if bundle.get('rigid'):
        r = bundle['rigid']
        out['rigid_inputs'] = {n: complex_of(read(base, b['responses'], check)) for n, b in r['budgets'].items()}
        out['rigid_label_runs'] = out['rigid_inputs'].pop(d['label'])               # [K, P, Q, F]
        out['rigid_target'] = complex_of(read(base, r['target_mean'], check))
        out['rigid_variance'] = np.asarray(read(base, r['target_variance'], check), np.float64)
    return out


def verify(target, deep=True):
    """Hashes, sizes, identities and split isolation. Raises on the first failure."""
    manifest, bundles, base = open_dataset(target)
    examples = read_lines(Path(target) / manifest['examples']['file'])
    report = dict(bundles=len(bundles), examples=len(examples), checkedFiles=0)
    if len(bundles) != manifest['bundles']['rows'] or len(examples) != manifest['examples']['rows']:
        raise ValueError('Row counts differ from dataset.json.')
    ids = [b['bundle_id'] for b in bundles]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate bundle ids.')
    groups = manifest['split_groups']
    by_split = {}
    for b in bundles:
        if groups.get(b['split_group']) != b['split']:
            raise ValueError(f'{b["bundle_id"]}: split {b["split"]} differs from its group mapping.')
        scene_bytes = (base / b['scene']['file']).read_bytes()
        if hashlib.sha256(scene_bytes).hexdigest() != b['scene']['sha256']:
            raise ValueError(f'{b["bundle_id"]}: scene.json changed.')
        scene = json.loads(scene_bytes)
        if ds.split_for(scene['family'], scene['parameters']) != b['split']:
            raise ValueError(f'{b["bundle_id"]}: split differs from the frozen rule.')
        run = json.loads((base / b['tracer_run']['file']).read_text())
        if sha256(base / b['tracer_run']['file']) != b['tracer_run']['sha256'] or run['sceneSha256'] != b['scene']['sha256']:
            raise ValueError(f'{b["bundle_id"]}: tracer run identity differs.')
        if b['label_seeds'] < 2:
            raise ValueError(f'{b["bundle_id"]}: fewer than two label seeds.')
        if deep:
            items = [d for budget in b['descriptors']['budgets'].values()
                     for d in (budget['responses'], budget['counts'], budget['peaks'], budget['seconds'])]
            items += [b['descriptors'][k] for k in ['target_mean', 'target_variance', 'target_validity', 'total_target_mean',
                                                     'total_target_variance', 'context', 'direct_visible']]
            items += [b['wave']['wave']] if b.get('wave') else []
            if b.get('rigid'):
                items += [d for budget in b['rigid']['budgets'].values()
                          for d in (budget['responses'], budget['counts'], budget['peaks'], budget['seconds'])]
                items += [b['rigid']['target_mean'], b['rigid']['target_variance']]
            for d in items:
                read(base, d, check=True)
                report['checkedFiles'] += 1
        by_split.setdefault(b['split'], set()).add(b['family'])
    families = {b['family'] for b in bundles}
    held_out = sorted(f for f in families if by_split.get('test', set()) >= {f} and
                      f not in by_split.get('train', set()) and f not in by_split.get('validation', set()))
    if not held_out:
        raise ValueError('No family is held out entirely in the test split.')
    split_of = {b['bundle_id']: b['split'] for b in bundles}
    receivers = {b['bundle_id']: b['receivers'] for b in bundles}
    accepted = {b['bundle_id'] for b in bundles if b['quality']['accepted']}
    variants = set(manifest['low_variants'])
    seen = set()
    for row in examples:
        if row['bundle_id'] not in accepted or row['low_variant'] not in variants:
            raise ValueError(f'{row["example_id"]}: unknown or rejected bundle/variant.')
        if not 0 <= row['pair_index'] < receivers[row['bundle_id']]:
            raise ValueError(f'{row["example_id"]}: receiver index out of range.')
        if row['example_id'] in seen:
            raise ValueError(f'{row["example_id"]}: duplicate example.')
        seen.add(row['example_id'])
    report.update(heldOutFamilies=held_out, splits={s: sorted(f) for s, f in by_split.items()},
                  bundlesPerSplit={s: sum(v == s for v in split_of.values()) for s in by_split},
                  rejected=len(bundles) - len(accepted), datasetSha256=sha256(Path(target) / 'dataset.json'))
    return report


def mesh_volume(boundary):
    v = np.array(boundary['vertices'], float).reshape(-1, 3)
    t = np.array(boundary['triangles'], int).reshape(-1, 3)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6)


def union_volume(walls, spacing):
    """Exact union volume of lattice boxes by rasterising at `spacing`."""
    low, high = ds.box_arrays(walls)
    origin = low.min(0)
    lo = np.round((low - origin) / spacing).astype(int)
    hi = np.round((high - origin) / spacing).astype(int)
    grid = np.zeros(hi.max(0), bool)
    for l, h in zip(lo, hi):
        grid[l[0]:h[0], l[1]:h[1], l[2]:h[2]] = True
    return float(grid.sum() * spacing ** 3), grid


def identity(data, variant='tracer'):
    """Same scene, same solid: Unity bake volume vs FDTD voxels vs exact box union.

    The bake has one solid group per material preset; groups do not overlap in
    generated scenes, so their volumes add to the union of all boxes.
    """
    import fdtd_wave as fw
    rows = []
    for folder in sorted(p.parent.parent for p in (data / 'scenes').glob(f'*/{variant}/boundary.json')):
        scene = json.loads((folder / 'scene.json').read_text())
        baked = json.loads((folder / variant / 'boundary.json').read_text())
        exact, _ = union_volume(scene['walls'], ds.LATTICE)
        names = baked['groups']['names']
        materials = json.loads((folder / variant / 'run.json').read_text())['materials']
        grouped = sum(union_volume([w for w, m in zip(scene['walls'], materials) if m == name], ds.LATTICE)[0]
                      for name in names)
        bake = sum(mesh_volume(b) for b in baked['boundaries'])
        row = dict(sceneId=scene['sceneId'], exact=exact, groupedExact=grouped, bake=bake,
                   reportedBake=sum(b['volume'] for b in baked['boundaries']), groups=len(names),
                   components=sum(b['components'] for b in baked['boundaries']))
        for h in (ds.LATTICE, ds.FDTD_SPACING):
            low, high = fw.scene_bounds(scene, ds.FDTD_MARGIN)
            grid = fw.make_grid(low, high, h, 0)
            solid, snap = fw.voxelize(grid, scene['walls'])
            row[f'voxels{h * 1000:g}mm'] = float(solid.sum() * h ** 3)
            row[f'snap{h * 1000:g}mm'] = snap
        row['groupOverlap'] = abs(grouped - exact) / exact
        row['bakeRelative'] = abs(bake - exact) / exact
        row['voxelRelative'] = max(abs(row[f'voxels{h * 1000:g}mm'] - exact) / exact for h in (ds.LATTICE, ds.FDTD_SPACING))
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=DATA)
    sub = p.add_subparsers(dest='command', required=True)
    k = sub.add_parser('pack')
    k.add_argument('name')
    v = sub.add_parser('verify')
    v.add_argument('dataset', type=Path)
    v.add_argument('--shallow', action='store_true')
    sub.add_parser('identity')
    a = p.parse_args()
    data = a.data.resolve()
    if a.command == 'pack':
        print(pack(data, a.name))
    elif a.command == 'verify':
        print(json.dumps(verify(a.dataset, not a.shallow), indent=1))
    else:
        rows = identity(data)
        worst = dict(bake=max(r['bakeRelative'] for r in rows), voxel=max(r['voxelRelative'] for r in rows),
                     groupOverlap=max(r['groupOverlap'] for r in rows),
                     snap=max(max(r[k] for k in r if k.startswith('snap')) for r in rows))
        print(json.dumps(dict(scenes=len(rows), worst=worst), indent=1))
        (data / 'identity.json').write_text(json.dumps(dict(rows=rows, worst=worst), indent=1) + '\n')


if __name__ == '__main__':
    main()
