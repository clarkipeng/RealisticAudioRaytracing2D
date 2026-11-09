"""Benchmark suite and wave ground truth for the acoustic method comparison.

Builds the scenes (two BEM fixtures and generated dataset scenes, rigid and with
the existing material presets), solves the ground truth with the FDTD solver
(tools/fdtd_wave.py) one octave band at a time, and writes it in the common
format (docs/benchmarks/method-compare/FORMAT.md) as method `gt`, budget `ref`,
seed 0. The BEM fixtures' own complex pressure is written as method `bem`.

Wall absorption: each band runs with every wall a locally reacting surface of
real impedance whose Paris (random-incidence) absorption equals the preset's
absorption at that band centre; the run is then band-filtered with that band's
filter. See README "Wall absorption".

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY tools/method_compare_suite.py build
    nice -n 19 $PY tools/method_compare_suite.py plan
    nice -n 19 $PY tools/method_compare_suite.py gt [scene ...]       # throttled, resumable
    nice -n 19 $PY tools/method_compare_suite.py assemble
    nice -n 19 $PY tools/method_compare_suite.py validate reflection|shoebox|refine
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

import dlss_scenes as ds
import fdtd_wave as fw
import heavy_guard as hg
import method_compare as mc

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / '.context/method-compare'
DATA = WORK / 'data'
FDTD = WORK / 'fdtd'
VALIDATION = WORK / 'validation'
EVIDENCE = ROOT / 'docs/benchmarks/wave-review-2026-10-27/evidence.zip'
MATERIALS = ROOT / 'Assets/Materials'
GIB = fw.GIB

PPW_AT_SUPPORT_TOP = 10.0     # points per wavelength at 2 fc, the top of a band filter's support
COMPLEX_PPW = 20.0            # complex ground truth only where the grid has at least this many
COMPLEX_STEP = 15.625         # Hz
SOURCE_SPECTRUM_FLOOR = .2    # complex from a run only where |Q(f)| >= this fraction of its peak
VALID_GUARD = .02             # s trimmed before the end of a run (filter smearing at the cut)
FIXTURE_RECEIVERS = 96

# The suite. Scene ids are <family>-<index>-<variant> for generated scenes.
GENERATED = [
    ('shoebox', 20, 'presets'), ('shoebox', 10, 'presets'),
    ('doorway', 8, 'presets'), ('corridor', 13, 'presets'), ('furnished', 5, 'presets'),
    ('canopy', 9, 'rigid'), ('canopy', 9, 'presets'),
    ('barriers', 20, 'rigid'), ('barriers', 20, 'presets'),
]
FIXTURES = dict(
    canopy=dict(scene='canopy/reference/scene.json', bem='canopy/reference/reference-0.25-125.npz', hz=125.0,
                lattice=.05, view=[0, 2], duration=1.25),
    blocks=dict(scene='blocks/reference/scene.json', bem='blocks/reference/reference-0.14-250.npz', hz=250.0,
                lattice=.1, view=[0, 1], duration=.3),
)
DURATION = dict(shoebox=2.05, furnished=2.05, doorway=2.05, corridor=2.05, canopy=.8, barriers=.5)
GT_BANDS = [125.0, 250.0, 500.0]
EXTRA_BANDS = {'blocks-bem': [1000.0], 'shoebox-0020-presets': [1000.0]}


# --------------------------------------------------------------------------- materials

def preset_curve(name):
    """Energy-curve knots (Hz, absorption, scattering) of Assets/Materials/Church <name>.asset."""
    text = (MATERIALS / f'Church {name}.asset').read_text()
    block = text.split('energyCurve:')[1].split('roughness:')[0]
    knots = re.findall(r'frequencyHz:\s*([\d.eE+-]+)\s*absorption:\s*([\d.eE+-]+)\s*scattering:\s*([\d.eE+-]+)', block)
    return [tuple(map(float, k)) for k in knots]


def preset_bands(name):
    """Absorption and scattering at the octave centres, linear in Hz between knots and clamped (renderer rule)."""
    knots = sorted(preset_curve(name))
    hz = [k[0] for k in knots]
    absorption = np.interp(mc.BANDS, hz, [k[1] for k in knots])
    scattering = np.interp(mc.BANDS, hz, [k[2] for k in knots])
    return absorption, scattering


def material_table(names):
    out = {}
    for name in sorted(set(names)):
        if name == 'rigid':
            zero = [0.0] * len(mc.BANDS)
            out[name] = dict(absorption=zero, scattering=zero, transmission=zero, presetScattering=zero,
                             admittance=zero, source='rigid (no absorption)')
            continue
        a, s = preset_bands(name)
        path = MATERIALS / f'Church {name}.asset'
        a = np.round(a, 8)
        out[name] = dict(absorption=a.tolist(), scattering=[0.0] * len(mc.BANDS),
                         transmission=[0.0] * len(mc.BANDS), presetScattering=np.round(s, 8).tolist(),
                         admittance=[round(fw.admittance_from_absorption(v, 'random'), 10) for v in a],
                         source=f'Assets/Materials/Church {name}.asset energyCurve (sha256 {mc.sha256_file(path)[:16]})')
    return out


# --------------------------------------------------------------------------- scenes

def scene_id(family, index, variant):
    return f'{family}-{index:04d}-{variant}'


def generated_scene(family, index, variant):
    base = ds.generate(family, index)
    names = base['surfaceMaterials'] if variant == 'presets' else ['rigid'] * len(base['walls'])
    sid = scene_id(family, index, variant)
    los = np.asarray(base['lineOfSight'], int)
    scene = dict(base, schema=mc.SCENE_SCHEMA, sceneId=sid, variant=variant, surfaceMaterials=names,
                 materials=material_table(names), shadow=(1 - los).tolist(), bands=mc.BANDS.tolist(),
                 airAbsorption=False, gtBands=gt_bands(sid), viewAxes=[0, 2],
                 origin=dict(generator=ds.GENERATOR_VERSION, family=family, index=index,
                             datasetSceneSha256=hashlib.sha256(ds.dumps(base)).hexdigest()))
    scene.pop('render', None)
    scene.pop('fdtd', None)
    scene['comparison'] = dict(lattice=ds.LATTICE, duration=DURATION[family])
    return scene


def gt_bands(sid):
    return GT_BANDS + EXTRA_BANDS.get(sid, [])


def extract_evidence(target=WORK / 'evidence'):
    """The two BEM fixtures from the committed wave-review archive (scene + reference only)."""
    with zipfile.ZipFile(EVIDENCE) as z:
        for spec in FIXTURES.values():
            for name in (spec['scene'], spec['bem']):
                out = target / name
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))
    return target


def fixture_scene(name, evidence):
    spec = FIXTURES[name]
    raw = json.loads((evidence / spec['scene']).read_text())
    bem = np.load(evidence / spec['bem'])
    points = np.array(raw['points'], float).reshape(-1, 3)
    if not np.allclose(points, bem['points']):
        raise ValueError(f'{name}: BEM points differ from the scene')
    los = ds.line_of_sight(raw['source'], points, raw['walls'])
    # Deterministic stratified subset: half in view, half in shadow, evenly spaced in index.
    pick = []
    for flag in (True, False):
        idx = np.flatnonzero(los == flag)
        take = min(len(idx), FIXTURE_RECEIVERS // 2)
        pick += idx[np.round(np.linspace(0, len(idx) - 1, take)).astype(int)].tolist()
    pick = np.array(sorted(set(pick)))
    sid = f'{name}-bem'
    names = ['rigid'] * len(raw['walls'])
    scene = dict(schema=mc.SCENE_SCHEMA, sceneId=sid, family=sid, variant='rigid', closed=False,
                 source=raw['source'], walls=raw['walls'], surfaceMaterials=names, materials=material_table(names),
                 points=np.round(points[pick], 9).ravel().tolist(), lineOfSight=los[pick].astype(int).tolist(),
                 shadow=(~los[pick]).astype(int).tolist(), speedOfSound=raw['speedOfSound'],
                 bands=mc.BANDS.tolist(), airAbsorption=False, gtBands=gt_bands(sid), viewAxes=spec['view'],
                 referenceConvention='exp(+i*w*t)', lattice=spec['lattice'],
                 comparison=dict(lattice=spec['lattice'], duration=spec['duration'], bemHz=spec['hz']),
                 origin=dict(fixture=spec['scene'], bem=spec['bem'], archive=str(EVIDENCE.relative_to(ROOT)),
                             fixtureSceneSha256=mc.sha256_file(evidence / spec['scene']),
                             bemSha256=mc.sha256_file(evidence / spec['bem']),
                             receiverIndices=pick.tolist()))
    return scene, bem['pressure'][pick]


def write_scene(scene):
    folder = DATA / scene['sceneId']
    folder.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(scene, indent=1, sort_keys=True, allow_nan=False) + '\n').encode()
    path = folder / 'scene.json'
    if path.exists() and path.read_bytes() != data:
        # bem is rewritten by build and gt by assemble; any other method's output would be stale.
        stale = [p for p in folder.iterdir() if p.is_dir() and p.name not in ('bem', 'gt')]
        if stale:
            raise RuntimeError(f'{scene["sceneId"]}: scene.json changed but outputs exist: {stale}')
        if (folder / 'gt').exists():
            shutil.rmtree(folder / 'gt')
    tmp = path.with_suffix('.json.partial')
    tmp.write_bytes(data)
    tmp.rename(path)
    return path


def build():
    """Write suite.json, every scene.json and the BEM fixtures' `bem` outputs."""
    evidence = extract_evidence()
    entries = []
    for name in FIXTURES:
        scene, pressure = fixture_scene(name, evidence)
        path = write_scene(scene)
        hz = FIXTURES[name]['hz']
        R = len(pressure)
        energy = np.full((R, len(mc.BANDS), mc.TIME_BINS), np.nan, np.float32)
        meta = dict(method='bem', budget='ref', seed=0, budgetValue=1, sceneId=scene['sceneId'],
                    sceneSha256=mc.sha256_file(path), wallSeconds=0.0, validSeconds=mc.BIN_SECONDS * mc.TIME_BINS,
                    bandsProvided=[], frequenciesHz=[hz], bandLimited=True,
                    code=dict(name='bempp 0.4.2 BEM, wave-review 2026-10-27', sha256=scene['origin']['bemSha256']),
                    notes='Single-frequency BEM reference (complex only); energy is NaN.')
        mc.write_output(DATA / scene['sceneId'] / 'bem/ref/0', meta, energy, pressure[:, None])
        entries.append(scene)
    for family, index, variant in GENERATED:
        scene = generated_scene(family, index, variant)
        write_scene(scene)
        entries.append(scene)
    suite = dict(schema=mc.SUITE_SCHEMA, scenes=[
        dict(id=s['sceneId'], family=s['family'], variant=s['variant'], closed=bool(s['closed']),
             gtBands=s['gtBands'], receivers=len(s['points']) // 3, shadowReceivers=int(np.sum(s['shadow'])))
        for s in entries])
    (DATA / 'suite.json').write_text(json.dumps(suite, indent=1) + '\n')
    return suite


# --------------------------------------------------------------------------- FDTD plan

def band_spacing(scene, fc):
    """Largest lattice/n spacing with PPW_AT_SUPPORT_TOP points per wavelength at 2 fc."""
    lattice = scene['comparison']['lattice']
    c = scene['speedOfSound']
    n = max(1, math.ceil(lattice / (c / (2 * fc) / PPW_AT_SUPPORT_TOP) - 1e-9))
    return lattice / n


def run_settings(scene, fc):
    closed = bool(scene.get('closed'))
    return dict(spacing=band_spacing(scene, fc), duration=scene['comparison']['duration'],
                peak=1.2 * fc, pml=0 if closed else 16, margin=0.0 if closed else .5)


def wall_admittance(scene, b):
    """Per-wall normalised admittance for band index b (random-incidence match)."""
    out = []
    for name in scene['surfaceMaterials']:
        alpha = scene['materials'][name]['absorption'][b]
        out.append(fw.admittance_from_absorption(alpha, 'random'))
    return out


def plan(scenes=None):
    rows = []
    for scene in iter_scenes(scenes):
        for fc in scene['gtBands']:
            s = run_settings(scene, fc)
            low, high = fw.scene_bounds(scene, s['margin'])
            grid = fw.make_grid(low, high, s['spacing'], s['pml'])
            dt = fw.COURANT * s['spacing'] / scene['speedOfSound']
            steps = math.ceil(s['duration'] / dt)
            rows.append(dict(scene=scene['sceneId'], band=fc, **s, cells=grid.cells, steps=steps,
                             gcell=grid.cells * steps / 1e9, gib=fw.memory_bytes(grid) / GIB,
                             done=(FDTD / scene['sceneId'] / f'{int(fc)}.json').exists()))
    return rows


def iter_scenes(ids=None):
    suite = json.loads((DATA / 'suite.json').read_text())
    for entry in suite['scenes']:
        if ids and entry['id'] not in ids:
            continue
        yield json.loads((DATA / entry['id'] / 'scene.json').read_text())


# --------------------------------------------------------------------------- FDTD ground truth

def fdtd_transfer_spectrum(pressure, q, dt, pad_seconds=.3):
    """H(f) = 4 pi P / (i w~ Q) on a uniform rfft grid (exp(+iwt), free field exp(-ikr)/r).

    pressure [N+1, R] at n dt, q [N] at (n+1/2) dt. Frequencies where |Q| is
    below 1e-6 of its peak are set to zero (no band filter reaches them).
    """
    N = pressure.shape[0]
    n = N + int(pad_seconds / dt)
    n += n % 2
    f = np.fft.rfftfreq(n, dt)
    P = np.fft.rfft(pressure.astype(np.float64), n, axis=0) * dt
    Q = np.fft.rfft(q.astype(np.float64), n) * dt * np.exp(-1j * np.pi * f * dt)
    w_disc = 2 * np.sin(np.pi * f * dt) / dt
    ok = np.abs(Q) > 1e-6 * np.abs(Q).max()
    H = np.zeros_like(P)
    H[ok] = 4 * np.pi * P[ok] / (1j * w_disc[ok] * Q[ok])[:, None]
    return H.T, f[1] - f[0]


def run_band(scene, fc, work_fraction=hg.WORK_FRACTION, memory_gib=2.0, power=None):
    """Solve one band and store its band energy and raw receiver record."""
    folder = FDTD / scene['sceneId']
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f'{int(fc)}'
    if (folder / f'{int(fc)}.json').exists():
        return json.loads((folder / f'{int(fc)}.json').read_text())
    b = int(np.flatnonzero(mc.BANDS == fc)[0])
    s = run_settings(scene, fc)
    admit = wall_admittance(scene, b)
    started = time.time()

    def progress(n, total, last=[0.0]):
        if time.time() - last[0] > 120:
            last[0] = time.time()
            print(f'  {scene["sceneId"]} {fc:g} Hz: {n}/{total} steps, {time.time() - started:.0f} s', flush=True)
    out = fw.solve(scene, s['spacing'], [fc], s['duration'], s['peak'], pml=s['pml'], margin=s['margin'],
                   work_fraction=work_fraction, memory_limit=memory_gib * GIB, progress=progress,
                   pause=power, wall_admittance=admit)
    meta = out['meta']
    H, df = fdtd_transfer_spectrum(out['pressure'], out['source'], meta['dt'])
    energy, beyond = mc.energy_from_spectrum(H, df, [fc])
    valid = s['duration'] - VALID_GUARD
    vb = int(math.floor(valid / mc.BIN_SECONDS))
    e = energy[:, 0]
    e[:, vb:] = 0
    total = e.sum(1)
    tail = e[:, max(0, vb - 50):vb].sum(1) / np.maximum(total, 1e-300)
    np.save(f'{stem}-energy.npy', e.astype(np.float32))
    np.save(f'{stem}-pressure.npy', out['pressure'].astype(np.float32))
    np.save(f'{stem}-source.npy', out['source'].astype(np.float64))
    info = dict(sceneId=scene['sceneId'], band=fc, admittance=sorted(set(admit)), validSeconds=vb * mc.BIN_SECONDS,
                lastFiftyMsFraction=dict(max=float(tail.max()), median=float(np.median(tail))),
                solverSha256=mc.sha256_file(ROOT / 'tools/fdtd_wave.py'),
                suiteSha256=mc.sha256_file(Path(__file__)), elapsedSeconds=time.time() - started,
                workFraction=work_fraction, **s, **{k: v for k, v in meta.items() if k not in s and k != 'hz'})
    (folder / f'{int(fc)}.json').write_text(json.dumps(info, indent=1, default=mc._json) + '\n')
    t = meta['timing']
    print(f'  {scene["sceneId"]} {fc:g} Hz: done h={s["spacing"]*100:.4g} cm, {meta["cells"]/1e6:.1f} M cells, '
          f'GPU {t["gpuSeconds"]:.0f} s, wall {t["wallSeconds"]:.0f} s, {t["gcellPerSecond"]:.2f} Gcell/s, '
          f'last-50-ms energy fraction max {tail.max():.1e}', flush=True)
    return info


def run_gt(ids=None, work_fraction=hg.WORK_FRACTION, memory_gib=2.0, bands=None):
    import dlss_jobs as dj
    power = dj.PowerCheck()
    order = sorted(plan(ids), key=lambda r: r['gcell'])          # cheapest first
    for row in order:
        if row['done'] or (bands and row['band'] not in bands):
            continue
        dj.wait_for_mains()
        scene = json.loads((DATA / row['scene'] / 'scene.json').read_text())
        print(f'{row["scene"]} {row["band"]:g} Hz: {row["gcell"]:.0f} Gcell updates, {row["gib"]:.2f} GiB', flush=True)
        run_band(scene, row['band'], work_fraction, memory_gib, power)
        fw.torch.mps.empty_cache()
        time.sleep(10)


def complex_plan(scene, infos):
    """For each frequency of the complex grid: (band run to use, points per wavelength), or None.

    Rigid scenes may use the finest run whose source spectrum covers the
    frequency (walls are band independent); scenes with materials use the run
    of the band whose nominal range holds the frequency.
    """
    c = scene['speedOfSound']
    rigid = all(n == 'rigid' for n in scene['surfaceMaterials'])
    plan_ = {}
    for fc in scene['gtBands']:
        lo, hi = fc / math.sqrt(2), fc * math.sqrt(2)
        for f in np.arange(math.ceil(lo / COMPLEX_STEP) * COMPLEX_STEP, hi, COMPLEX_STEP):
            candidates = [fc]
            if rigid:
                candidates = sorted(infos, key=lambda k: infos[k]['spacing'])
            for k in candidates:
                info = infos[k]
                peak = info['peak']
                ratio = (f / peak) * math.exp(.5 * (1 - (f / peak) ** 2))     # |Q(f)| / |Q(peak)|
                ppw = c / f / info['spacing']
                if ratio >= SOURCE_SPECTRUM_FLOOR and ppw >= COMPLEX_PPW:
                    plan_[float(f)] = (k, ppw)
                    break
    return plan_


def assemble(ids=None):
    """Combine finished band runs into gt/ref/0 for every scene whose bands are all done."""
    written = []
    for scene in iter_scenes(ids):
        folder = FDTD / scene['sceneId']
        infos = {}
        for fc in scene['gtBands']:
            path = folder / f'{int(fc)}.json'
            if path.exists():
                infos[fc] = json.loads(path.read_text())
        if len(infos) < len(scene['gtBands']):
            print(f'{scene["sceneId"]}: {len(infos)}/{len(scene["gtBands"])} bands done, skipped')
            continue
        R = len(scene['points']) // 3
        energy = np.full((R, len(mc.BANDS), mc.TIME_BINS), np.nan, np.float32)
        for fc in infos:
            b = int(np.flatnonzero(mc.BANDS == fc)[0])
            energy[:, b] = np.load(folder / f'{int(fc)}-energy.npy')
        # Runs can be longer than the format's 2 s window; only that window is stored.
        valid = min(min(i['validSeconds'] for i in infos.values()), mc.TIME_BINS * mc.BIN_SECONDS)
        energy[:, :, int(round(valid / mc.BIN_SECONDS)):] = np.where(
            np.isnan(energy[:, :, int(round(valid / mc.BIN_SECONDS)):]), np.nan, 0)
        cplan = complex_plan(scene, infos)
        hz = sorted(cplan)
        pressure = np.zeros((R, len(hz)), complex)
        cache = {}
        for j, f in enumerate(hz):
            k, _ = cplan[f]
            if k not in cache:
                cache[k] = (np.load(folder / f'{int(k)}-pressure.npy'), np.load(folder / f'{int(k)}-source.npy'))
            p, q = cache[k]
            pressure[:, j] = fw.transfer(p, q, infos[k]['dt'], f, scene['speedOfSound'])[0]
        gpu = sum(i['timing']['gpuSeconds'] for i in infos.values())
        wall = sum(i['timing']['wallSeconds'] for i in infos.values())
        meta = dict(method='gt', budget='ref', seed=0, budgetValue=1, sceneId=scene['sceneId'],
                    sceneSha256=mc.sha256_file(DATA / scene['sceneId'] / 'scene.json'),
                    code=dict(name='tools/fdtd_wave.py (per-band locally reacting real impedance)',
                              sha256=sorted({i['solverSha256'] for i in infos.values()})),
                    wallSeconds=wall, gpuSeconds=gpu, validSeconds=valid, bandLimited=True,
                    bandsProvided=sorted(infos), frequenciesHz=hz,
                    complexSource={f'{f:g}': dict(bandRun=cplan[f][0], pointsPerWavelength=round(cplan[f][1], 2))
                                   for f in hz},
                    bandRuns={f'{k:g}': dict(spacing=i['spacing'], duration=i['duration'], cells=i['cells'],
                                             steps=i['timing']['steps'], admittance=i['admittance'],
                                             gpuSeconds=i['timing']['gpuSeconds'],
                                             lastFiftyMsFraction=i['lastFiftyMsFraction'])
                              for k, i in sorted(infos.items())},
                    notes='FDTD, one run per octave band with that band\'s wall impedance, zero-phase band filter; '
                          'no ground truth above the listed bands.')
        mc.write_output(DATA / scene['sceneId'] / 'gt/ref/0', meta, energy, pressure)
        written.append(scene['sceneId'])
        print(f'{scene["sceneId"]}: gt written ({len(hz)} complex frequencies, valid {valid:.3f} s)')
    return written


# --------------------------------------------------------------------------- harness self-test

SELFTEST = WORK / 'selftest'
SELFTEST_BIAS_DB = 1.0
SELFTEST_BUDGETS = (16, 256, 4096)


def selftest(seeds=3):
    """Copy every scene's gt into a separate root and add a perturbed method with known errors.

    `synthetic` = gt energy x 10^(bias/10) x max(0, 1 + n * 2/sqrt(N)) per bin (independent
    Gaussian n per seed) and gt complex x 10^(bias/20) x (1 + complex noise * 0.5/sqrt(N)).
    Expected: level bias -> +1 dB, complex error -> 12.2%, every other metric and the
    seed spread -> 0 as N grows; gt against itself scores exactly zero.
    """
    import shutil as sh
    if SELFTEST.exists():
        sh.rmtree(SELFTEST)
    found = mc.discover(DATA)
    scenes = []
    for sid, runs in found.items():
        if 'gt' not in runs:
            continue
        (SELFTEST / sid).mkdir(parents=True)
        sh.copy2(DATA / sid / 'scene.json', SELFTEST / sid / 'scene.json')
        sh.copytree(runs['gt']['ref'][0], SELFTEST / sid / 'gt/ref/0')
        meta, e, cp, _ = mc.read_output(runs['gt']['ref'][0])
        scene_path = SELFTEST / sid / 'scene.json'
        for N in SELFTEST_BUDGETS:
            for seed in range(seeds):
                rng = np.random.default_rng(mc_seed(sid, N, seed))
                gain = 10 ** (SELFTEST_BIAS_DB / 10)
                noisy = e * gain * np.maximum(0, 1 + rng.standard_normal(e.shape) * 2 / math.sqrt(N))
                c = None
                if cp is not None:
                    c = cp * 10 ** (SELFTEST_BIAS_DB / 20) * (1 + (rng.standard_normal(cp.shape) +
                                                                     1j * rng.standard_normal(cp.shape)) * .5 / math.sqrt(N))
                m = dict(method='synthetic', budget=f'r{N:05d}', seed=seed, budgetValue=N, budgetUnit='synthetic rays',
                         sceneId=sid, sceneSha256=mc.sha256_file(scene_path), wallSeconds=N * 1e-4,
                         validSeconds=meta['validSeconds'], bandsProvided=meta['bandsProvided'], bandLimited=True,
                         code=dict(name='method_compare_suite.selftest', sha256=mc.sha256_file(Path(__file__))),
                         notes=f'gt x {SELFTEST_BIAS_DB} dB with 2/sqrt(N) per-bin noise')
                if c is not None:
                    m['frequenciesHz'] = meta['frequenciesHz']
                mc.write_output(SELFTEST / sid / f'synthetic/r{N:05d}/{seed}', m, noisy, c)
        scenes.append(dict(id=sid))
    suite = json.loads((DATA / 'suite.json').read_text())
    suite['scenes'] = [s for s in suite['scenes'] if s['id'] in {x['id'] for x in scenes}]
    (SELFTEST / 'suite.json').write_text(json.dumps(suite, indent=1) + '\n')
    return mc.score(SELFTEST, SELFTEST.parent / 'selftest-score')


def mc_seed(*parts):
    return int.from_bytes(hashlib.sha256('/'.join(map(str, parts)).encode()).digest()[:4], 'little')


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('build', help='Write suite.json, scene.json files and the bem outputs')
    pl = sub.add_parser('plan', help='Print the ground-truth runs and their cost')
    pl.add_argument('scenes', nargs='*')
    g = sub.add_parser('gt', help='Run missing ground-truth bands (cheapest first)')
    g.add_argument('scenes', nargs='*')
    g.add_argument('--bands', type=float, nargs='*')
    g.add_argument('--work-fraction', type=float, default=hg.WORK_FRACTION)
    g.add_argument('--memory-gib', type=float, default=2.0)
    a_ = sub.add_parser('assemble', help='Write gt/ref/0 from finished band runs')
    a_.add_argument('scenes', nargs='*')
    sub.add_parser('selftest', help='Harness self-test: gt vs gt and a perturbed method, in a separate root')
    v = sub.add_parser('validate', help='Absorbing-wall validation studies')
    v.add_argument('study', choices=['reflection', 'shoebox', 'refine', 'report'])
    v.add_argument('--work-fraction', type=float, default=hg.WORK_FRACTION)
    a = p.parse_args(argv)
    if a.command == 'build':
        suite = build()
        print(json.dumps(suite, indent=1))
    elif a.command == 'plan':
        rows = plan(a.scenes or None)
        total = 0
        for r in rows:
            total += r['gcell'] * (not r['done'])
            print(f'{r["scene"]:24s} {r["band"]:6g} Hz h={r["spacing"]*100:6.4g} cm {r["cells"]/1e6:7.1f} M cells '
                  f'{r["steps"]:6d} steps {r["gcell"]:8.0f} Gcell {r["gib"]:5.2f} GiB {"done" if r["done"] else ""}')
        print(f'remaining {total:.0f} Gcell updates = {total / 1.9 / 60:.0f} GPU min at 1.9 Gcell/s '
              f'({total / 1.9 / 60 / hg.WORK_FRACTION / 60:.1f} h at {hg.WORK_FRACTION:.0%} pacing)')
    elif a.command == 'gt':
        run_gt(a.scenes or None, a.work_fraction, a.memory_gib, a.bands)
    elif a.command == 'assemble':
        assemble(a.scenes or None)
    elif a.command == 'selftest':
        report = selftest()
        print((SELFTEST.parent / 'selftest-score' / 'summary.md').read_text())
    elif a.command == 'validate':
        import method_compare_validate as mv
        mv.main(a.study, a.work_fraction)


if __name__ == '__main__':
    main()
