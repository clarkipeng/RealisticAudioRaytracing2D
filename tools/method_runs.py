"""Runs of the three acoustic methods (hybrid, pressure, energy) for the method comparison.

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY tools/method_runs.py tests "TakeChecks.MethodsMatchTheFreeField"   # Unity checks
    $PY tools/method_runs.py jobs NAME --scenes canopy-0009-rigid ... --budgets 128 512 --seeds 0 1
    nice -n 19 $PY tools/method_runs.py render NAME                                     # Unity renders (raw)
    $PY tools/method_runs.py pack                                                       # raw -> method-compare-v1

Unity work goes through tools/dlss_jobs.py: isolated runner (.context/gpu-audio/runner),
nice 19, one job worker, tools/heavy_guard.py (machine-wide lock, memory-pressure
start/stop, 3 GiB footprint limit, 50% pacing), suspension on battery. Raw outputs live in .context/methods-2026-11-06/raw/<scene>/<method>/<budget>/<seed>/
and are packed into the comparison root (.context/method-compare/data) in the format of
docs/benchmarks/method-compare/FORMAT.md.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as xml

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_jobs as dj  # noqa: E402
import heavy_guard as hg  # noqa: E402

WORK = ROOT / '.context/methods-2026-11-06'
RAW = WORK / 'raw'
LOGS = WORK / 'logs'
COMPARE = ROOT / '.context/method-compare/data'
BANDS = [125., 250., 500., 1000., 2000., 4000., 8000., 16000.]
GT_GRID = 15.625


def clear_stale_lock():
    lock = dj.RUNNER / 'Temp/UnityLockfile'
    if lock.exists():
        processes = subprocess.run(['ps', '-A', '-o', 'command='], capture_output=True, text=True).stdout
        if str(dj.RUNNER) not in processes:
            lock.unlink()


def prune_runner():
    """Remove runner sources that were deleted from the repository (sync only copies)."""
    for folder in [f for f in dj.SYNCED if f.startswith('Assets/')]:
        for target in (dj.RUNNER / folder).rglob('*'):
            if target.is_file() and not (ROOT / target.relative_to(dj.RUNNER)).exists():
                source = target.with_suffix('') if target.suffix == '.meta' else target
                if not (ROOT / source.relative_to(dj.RUNNER)).exists():
                    target.unlink()


def launch(name, arguments, env=None, work_fraction=hg.WORK_FRACTION):
    logs = LOGS / time.strftime(f'{name}-%Y%m%d-%H%M%S')
    logs.mkdir(parents=True)
    dj.wait_for_mains()
    clear_stale_lock()
    prune_runner()
    (logs / 'provenance.json').write_text(json.dumps(dict(files=dj.sync_runner(dj.RUNNER)), indent=1) + '\n')
    command = ['nice', '-n', '19', str(dj.UNITY), '-batchmode', '-force-metal', '-disable-audio', '-job-worker-count', '1',
               '-projectPath', str(dj.RUNNER)] + arguments + ['-logFile', str(logs / 'unity.log')]
    environment = dict(os.environ, ACOUSTIC_TEST_WORK_FRACTION=str(work_fraction),
                       ACOUSTIC_SOLID_PYTHON=str(dj.SOLID_PYTHON), **(env or {}))
    print(f'{logs.name}: {" ".join(arguments)}', flush=True)
    resources = dj.Worker(command, environment, lambda m: print(m, flush=True)).run()
    (logs / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    print(f'{logs.name}: exit {resources["exitCode"]}, {resources["wallSeconds"]:.0f} s, '
          f'peak footprint {resources["peakTreeFootprintBytes"] / dj.GIB:.2f} GiB', flush=True)
    return logs, resources


def tests(filter_, env, work_fraction=hg.WORK_FRACTION):
    results = LOGS / '.results.xml'
    logs, resources = launch('tests', ['-runTests', '-testPlatform', 'EditMode', '-testFilter', filter_,
                                       '-testResults', str(results)], env, work_fraction)
    if not results.exists():
        print(f'No test results; see {logs}/unity.log', flush=True)
        return None
    results.rename(logs / 'tests.xml')
    root = xml.parse(logs / 'tests.xml').getroot()
    summary = {k: root.get(k) for k in ['result', 'total', 'passed', 'failed', 'skipped']}
    print(json.dumps(dict(log=str(logs), summary=summary)), flush=True)
    for case in root.iter('test-case'):
        output = case.find('output')
        text = (output.text or '').strip() if output is not None else ''
        message = case.find('failure/message')
        print(f"{case.get('result'):7s} {case.get('fullname')} {case.get('duration')} s", flush=True)
        if text:
            print('   ' + text.replace('\n', '\n   '), flush=True)
        if message is not None and message.text:
            print('   FAIL: ' + message.text.strip()[:800], flush=True)
    return summary


# ------------------------------------------------------------------ jobs

def scene_frequencies(scene):
    """Complex output grid: the ground truth's 15.625 Hz grid over the support of every GT band."""
    bands = scene.get('gtBands') or [125., 250., 500.]
    lo, hi = min(bands) / 2, max(bands) * 2
    return [GT_GRID * k for k in range(int(math.ceil(lo / GT_GRID)), int(math.floor(hi / GT_GRID)) + 1)]


def scene_spec(scene_dir, out):
    """JsonUtility-friendly copy of a method-compare scene for the Unity harness."""
    raw = (scene_dir / 'scene.json').read_bytes()
    scene = json.loads(raw)
    names = sorted(scene['materials'])
    spec = dict(sceneId=scene['sceneId'], sceneSha256=hashlib.sha256(raw).hexdigest(), source=scene['source'],
                points=scene['points'], walls=scene['walls'], surfaceMaterials=scene['surfaceMaterials'], materialNames=names,
                absorption=[v for n in names for v in scene['materials'][n]['absorption']],
                scattering=[v for n in names for v in scene['materials'][n]['scattering']],
                transmission=[v for n in names for v in scene['materials'][n]['transmission']],
                speedOfSound=scene.get('speedOfSound', 343.0), frequencies=scene_frequencies(scene))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec) + '\n')
    return out


def make_jobs(name, scenes, methods, budgets, seeds, root=COMPARE, capture=-1, receivers=-1, materials=None, radius=.25, capture_scenes=None,
              preset_scattering=False):
    """Write .context/methods-2026-11-06/jobs/NAME.json. `materials` optionally overrides
    scattering (e.g. {"scattering": 0.3}) for the divergence study; such scenes get a suffix."""
    jobs = []
    for scene_id in scenes:
        scene_dir = Path(root) / scene_id
        if not (scene_dir / 'scene.json').exists():
            scene_dir = Path(scene_id)
        spec = scene_spec(scene_dir, WORK / 'specs' / f'{scene_dir.name}.json')
        if preset_scattering:
            # Study variant: the presets' own scattering (the suite sets it to zero).
            scene = json.loads((scene_dir / 'scene.json').read_text())
            data = json.loads(spec.read_text())
            data['sceneId'] += '-scattering'
            data['scattering'] = [v for n in data['materialNames'] for v in scene['materials'][n]['presetScattering']]
            spec = spec.with_name(data['sceneId'] + '.json')
            spec.write_text(json.dumps(data) + '\n')
        if materials:
            data = json.loads(spec.read_text())
            suffix = '-' + '-'.join(f'{k[0]}{v:g}' for k, v in sorted(materials.items()))
            data['sceneId'] += suffix
            for key, value in materials.items():
                data[key] = [value] * len(data[key])
            spec = spec.with_name(data['sceneId'] + '.json')
            spec.write_text(json.dumps(data) + '\n')
        jobs.append(dict(sceneFile=str(spec), output=str(RAW), methods=methods, budgets=budgets, seeds=seeds,
                         maximumDelaySeconds=2.0, detectionRadius=radius, receivers=receivers,
                         captureReceiver=capture if capture_scenes is None or scene_id in capture_scenes else -1))
    path = WORK / 'jobs' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(jobs=jobs), indent=1) + '\n')
    print(path)
    return path


def render(name, work_fraction=hg.WORK_FRACTION):
    """One Unity process per (method, scene): each process compiles one method's shader
    and holds one scene (memory grew across scenes within one process). Finished runs
    (run.json present) are skipped; a failed process is reported and the rest continue."""
    path = WORK / 'jobs' / f'{name}.json'
    jobs = json.loads(path.read_text())['jobs']
    failures = []
    for method in ['hybrid', 'pressure', 'energy']:
        for job in jobs:
            if method not in job['methods']:
                continue
            scene = json.loads(Path(job['sceneFile']).read_text())['sceneId']
            done = all((RAW / scene / method / f'r{b:05d}' / str(s) / 'run.json').exists() for b in job['budgets'] for s in job['seeds'])
            if done:
                continue
            part = WORK / 'jobs' / f'{name}.{method}.{scene}.json'
            part.write_text(json.dumps(dict(jobs=[dict(job, methods=[method])]), indent=1) + '\n')
            results = LOGS / f'.render-{name}-{method}-{scene}.xml'
            try:
                logs, resources = launch(f'render-{name}-{method}-{scene}', ['-runTests', '-testPlatform', 'EditMode', '-testFilter',
                                         'TakeChecks.MeasureMethods', '-testResults', str(results)],
                                         dict(ACOUSTIC_METHOD_JOBS=str(part)), work_fraction)
            except MemoryError as error:
                failures.append(f'{method} {scene}: {error}')
                print('FAILED', failures[-1], flush=True)
                continue
            if results.exists():
                results.rename(logs / 'tests.xml')
                root = xml.parse(logs / 'tests.xml').getroot()
                for case in root.iter('test-case'):
                    message = case.find('failure/message')
                    if message is not None and message.text:
                        failures.append(f'{method} {scene}: ' + message.text.strip()[:500])
                        print('FAILED', failures[-1], flush=True)
    print(json.dumps(dict(failures=failures), indent=1), flush=True)
    return failures


# ------------------------------------------------------------------ packing

def read_raw(folder):
    run = json.loads((folder / 'run.json').read_text())
    R, T = run['receivers'], run['bins']
    load = lambda name, shape: np.fromfile(folder / name, '<f4').reshape(shape) if (folder / name).exists() else None
    F = len(run['frequencies'])
    return run, dict(energy=load('energy.f32', (R, 8, T)), late=load('late.f32', (R, 8, T)),
                     direction=load('direction.f32', (R, 8, T, 6)), complex=load('complex.f32', (R, F, 2)),
                     stats=load('stats.f32', (R, 8, 64, 4)))


def pack_one(folder, root=COMPARE):
    import method_compare as mc
    run, data = read_raw(folder)
    scene_id = run['sceneId']
    if not (root / scene_id / 'scene.json').exists():
        return None   # a study scene that is not part of the comparison suite
    points = len(json.loads((root / scene_id / 'scene.json').read_text())['points']) // 3
    if run['receivers'] != points:
        return None   # a receiver subset (convergence study), not a comparison output
    energy = data['energy'].astype(np.float64)
    banded = run['method'] != 'energy'
    if data['late'] is not None:
        energy = energy + mc.smooth(data['late'].astype(np.float64))
    T = energy.shape[2]
    if T < mc.TIME_BINS:
        energy = np.concatenate([energy, np.zeros(energy.shape[:2] + (mc.TIME_BINS - T,))], 2)
    direction = data['direction']
    if direction is not None and T < mc.TIME_BINS:
        direction = np.concatenate([direction, np.zeros(direction.shape[:2] + (mc.TIME_BINS - T, 6))], 2)
    complex_pressure = None
    if data['complex'] is not None:
        complex_pressure = data['complex'][..., 0] + 1j * data['complex'][..., 1]
    budget_unit = 'late rays per band (complete-route samples equal)' if run['method'] == 'hybrid' else 'rays (all bands per ray)'
    meta = dict(method=run['method'], budget=f"r{run['budget']:05d}", seed=run['seed'], budgetValue=run['budget'], budgetUnit=budget_unit,
                sceneId=scene_id, sceneSha256=run['sceneSha256'],
                code=dict(name={'hybrid': 'Transport.hlsl+LateEnergy.hlsl', 'pressure': 'MethodPressure.hlsl',
                                'energy': 'MethodEnergy.hlsl'}[run['method']], sha256=run['codeSha256']),
                wallSeconds=run['busySeconds'], gpuSeconds=run['gpuSeconds'] or None, validSeconds=run['maximumDelaySeconds'],
                bandLimited=banded, wallModel='absorption', notes=run['notes'] + f"; specular detection radius {run['detectionRadius']} m" * (run['method'] != 'hybrid'))
    if meta['gpuSeconds'] is None:
        del meta['gpuSeconds']
    if complex_pressure is not None:
        meta['frequenciesHz'] = run['frequencies']
    out = root / scene_id / run['method'] / meta['budget'] / str(run['seed'])
    mc.write_output(out, meta, energy.astype(np.float32), complex_pressure, direction)
    return out


def pack(root=COMPARE):
    count = 0
    for run in sorted(RAW.glob('*/*/*/*/run.json')):
        out = pack_one(run.parent, Path(root))
        if out is not None:
            count += 1
    print(f'packed {count} outputs into {root}')


# ------------------------------------------------------------------ analysis

def study(scenes=None, out=ROOT / 'docs/benchmarks/methods-2026-11-06'):
    """Convergence, divergence and cost from the raw runs: report.json and figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(hybrid='#0072B2', pressure='#D55E00', energy='#009E73')
    report = {}
    for scene_dir in sorted(p for p in RAW.iterdir() if p.is_dir() and (scenes is None or p.name in scenes)):
        for method_dir in sorted(scene_dir.iterdir()):
            if not method_dir.is_dir() or method_dir.name.startswith('.'):
                continue
            rows = []
            for budget_dir in sorted(method_dir.iterdir()):
                runs = [read_raw(r.parent) for r in sorted(budget_dir.glob('*/run.json'))]
                if not runs:
                    continue
                energies = []
                for run, data in runs:
                    e = data['energy'].astype(np.float64)
                    if data['late'] is not None:
                        e = e + data['late']
                    energies.append(e)
                E = np.stack(energies)                       # [seed, R, band, T]
                mean, sd = E.mean(0), E.std(0, ddof=1) if len(E) > 1 else np.zeros(E.shape[1:])
                level = 10 * np.log10(np.maximum(E.sum(-1), 1e-30))   # [seed, R, band]
                row = dict(budget=runs[0][0]['budget'], seeds=len(runs),
                           spread=float(sd.sum() / max(mean.sum(), 1e-30)) if len(E) > 1 else None,
                           spreadByBand=[float(sd[:, b].sum() / max(mean[:, b].sum(), 1e-30)) for b in range(8)] if len(E) > 1 else None,
                           levelSdDb=float(level.std(0, ddof=1).mean()) if len(E) > 1 else None,
                           meanLevelDb=[float(10 * np.log10(max(mean[:, b].sum(-1).mean(), 1e-30))) for b in range(8)],
                           secondsPerSolve=float(np.mean([r['busySeconds'] / r['receivers'] for r, _ in runs])),
                           gpuSecondsPerSolve=float(np.mean([(r['gpuSeconds'] or 0) / r['receivers'] for r, _ in runs])))
                if runs[0][1]['stats'] is not None:
                    S = np.stack([d['stats'] for _, d in runs]).astype(np.float64)   # [seed, R, band, depth, 4]
                    count, total = S[..., 0].sum((0, 1)), S[..., 1].sum((0, 1))
                    rays = runs[0][0]['methodRays'] * len(runs) * runs[0][0]['receivers']
                    row['meanAbsPerRayByDepth'] = (total / rays).tolist()             # [band][depth] E|c| per ray
                    row['maxAbsByDepth'] = S[..., 3].max((0, 1)).tolist()
                    row['contributionsByDepth'] = count.tolist()
                    counters = np.sum([r['counters'] for r, _ in runs], 0)
                    row['counters'] = dict(zip(runs[0][0]['counterNames'], counters.tolist()))
                rows.append(row)
            report.setdefault(scene_dir.name, {})[method_dir.name] = rows
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    scenes_with_budgets = [s for s in report if any(len(r) > 1 for r in report[s].values())]
    if scenes_with_budgets:
        fig, axes = plt.subplots(1, len(scenes_with_budgets), figsize=(4.6 * len(scenes_with_budgets), 3.8), squeeze=False)
        for ax, s in zip(axes[0], scenes_with_budgets):
            for m, rows in report[s].items():
                pts = [(r['budget'], r['spread']) for r in rows if r['spread'] is not None]
                if pts:
                    ax.loglog(*zip(*pts), 'o-', color=colors.get(m, 'k'), label=m)
            b = np.array([128, 8192.])
            ax.loglog(b, .5 * np.sqrt(128 / b), ':', color='#888888', label='1/sqrt(N)')
            ax.set_title(s, fontsize=9); ax.set_xlabel('budget (rays)'); ax.grid(alpha=.3, which='both')
        axes[0, 0].set_ylabel('seed spread (energy-weighted relative SD)'); axes[0, 0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / 'fig-convergence.png', dpi=130); plt.close(fig)
    depth = [(s, m, rows[-1]) for s in report for m, rows in report[s].items() if rows and 'meanAbsPerRayByDepth' in rows[-1]]
    if depth:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, m in zip(axes, ['pressure', 'energy']):
            for s, mm, row in depth:
                if mm != m:
                    continue
                y = np.array(row['meanAbsPerRayByDepth'])[2]    # 500 Hz band
                d = np.nonzero(y > 0)[0]
                ax.semilogy(d, y[d], '.-', label=f"{s} r{row['budget']}", lw=.8)
            ax.set_title(f'{m}: E|contribution| per ray by interaction depth (500 Hz)', fontsize=9, color=colors[m])
            ax.set_xlabel('interactions before the listener connection'); ax.grid(alpha=.3, which='both')
            ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(out / 'fig-depth.png', dpi=130); plt.close(fig)
    print(out / 'report.json')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('tests')
    t.add_argument('filter')
    t.add_argument('env', nargs='*')
    t.add_argument('--work-fraction', type=float, default=hg.WORK_FRACTION)
    j = sub.add_parser('jobs')
    j.add_argument('name')
    j.add_argument('--scenes', nargs='+', required=True)
    j.add_argument('--methods', nargs='+', default=['hybrid', 'pressure', 'energy'])
    j.add_argument('--budgets', nargs='+', type=int, default=[128, 512, 2048, 8192])
    j.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    j.add_argument('--capture', type=int, default=-1)
    j.add_argument('--receivers', type=int, default=-1)
    j.add_argument('--scattering', type=float)
    j.add_argument('--radius', type=float, default=.25)
    j.add_argument('--capture-scenes', nargs='*')
    j.add_argument('--preset-scattering', action='store_true', help="use each material's presetScattering (study scenes)")
    r = sub.add_parser('render')
    r.add_argument('name')
    r.add_argument('--work-fraction', type=float, default=hg.WORK_FRACTION)
    sub.add_parser('pack')
    o = sub.add_parser('outputs', help='print case results and TestContext output of the latest unity_tests.py run')
    o.add_argument('path', nargs='?')
    fp = sub.add_parser('footprint', help='peak process-tree footprint per test case of the latest unity_tests.py run')
    fp.add_argument('path', nargs='?')
    st = sub.add_parser('study')
    st.add_argument('scenes', nargs='*')
    args = parser.parse_args(argv)
    if args.command == 'tests':
        tests(args.filter, dict(a.split('=', 1) for a in args.env), args.work_fraction)
    elif args.command == 'jobs':
        make_jobs(args.name, args.scenes, args.methods, args.budgets, args.seeds, capture=args.capture, receivers=args.receivers,
                  materials=None if args.scattering is None else dict(scattering=args.scattering), radius=args.radius,
                  capture_scenes=args.capture_scenes, preset_scattering=args.preset_scattering)
    elif args.command == 'render':
        render(args.name, args.work_fraction)
    elif args.command == 'pack':
        pack()
    elif args.command == 'outputs':
        path = Path(args.path) if args.path else sorted((ROOT / '.context/unity-tests').glob('*/tests.xml'))[-1]
        for case in xml.parse(path).getroot().iter('test-case'):
            output, message = case.find('output'), case.find('failure/message')
            print(f"{case.get('result'):7s} {case.get('name')} {float(case.get('duration') or 0):.1f} s")
            if output is not None and output.text:
                print('   ' + output.text.strip().replace('\n', '\n   '))
            if message is not None and message.text:
                print('   FAIL: ' + message.text.strip()[:600])
    elif args.command == 'footprint':
        import datetime
        folder = Path(args.path) if args.path else sorted((ROOT / '.context/unity-tests').glob('*/tests.xml'))[-1].parent
        resources = json.loads((folder / 'resources.json').read_text())
        end = (folder / 'resources.json').stat().st_mtime
        begin = end - resources['wallSeconds']
        samples = [(begin + s['seconds'], s['treeFootprintBytes']) for s in resources['samples']]
        stamp = lambda t: datetime.datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp()
        print(f"run peak {resources['peakTreeFootprintBytes'] / 2**30:.2f} GiB (samples every ~{resources['samples'][1]['seconds'] - resources['samples'][0]['seconds']:.0f} s)")
        for case in xml.parse(folder / 'tests.xml').getroot().iter('test-case'):
            t0, t1 = stamp(case.get('start-time')), stamp(case.get('end-time'))
            inside = [b for t, b in samples if t0 - 2 <= t <= t1 + 2]
            peak = max(inside) / 2**30 if inside else float('nan')
            print(f"{case.get('result'):7s} {case.get('name'):55s} {float(case.get('duration')):6.1f} s  peak {peak:.2f} GiB ({len(inside)} samples)")
    elif args.command == 'study':
        study(args.scenes or None)


if __name__ == '__main__':
    main()
