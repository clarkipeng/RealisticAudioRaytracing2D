"""Guarded workers for the audio-DLSS dataset: scene generation, Unity tracer
renders and the FDTD wave subset.

Every heavy job runs at nice 19 with one worker, 20% work pacing, a memory guard
and a pause while the Mac runs on battery (the FDTD study's power hook; Unity
workers are suspended with SIGSTOP and resumed with SIGCONT). Bulk outputs stay
under the data root (default .context/audio-dlss-data), never in Assets/.

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY tools/dlss_jobs.py generate --count 12
    nice -n 19 $PY tools/dlss_jobs.py render --batch 6
    nice -n 19 $PY tools/dlss_jobs.py fdtd --select 24
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as xml

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_scenes as ds  # noqa: E402

DATA = ROOT / '.context/audio-dlss-data'
RUNNER = ROOT / '.context/gpu-audio/runner'
UNITY = Path('/Applications/Unity/Hub/Editor/6000.4.1f1/Unity.app/Contents/MacOS/Unity')
SOLID_PYTHON = ROOT / '.context/solid-python/bin/python'
GIB = 1024 ** 3
SYNCED = ['Assets/Scripts', 'Assets/Shaders', 'Assets/Resources/Acoustics', 'Assets/Tests/Editor', 'Assets/Editor', 'tools']
PRESETS = [f'Assets/Materials/Church {name}.asset' for name in ('stone', 'wood', 'glass', 'ground')]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def on_mains():
    out = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout
    return 'AC Power' in out


def wait_for_mains(log=print):
    """Block while on battery: heavy work only on AC power (FDTD study hook)."""
    told = False
    while not on_mains():
        if not told:
            log(f'paused on battery power at {time.strftime("%H:%M:%S")}')
            told = True
        time.sleep(60)
    if told:
        log(f'resumed on AC power at {time.strftime("%H:%M:%S")}')


class PowerCheck:
    """Throttled pause hook for in-process GPU work."""

    def __init__(self, period=30):
        self.period, self.last = period, 0.0

    def __call__(self):
        if time.time() - self.last > self.period:
            self.last = time.time()
            wait_for_mains()


def scene_dirs(data):
    return sorted(p.parent for p in (data / 'scenes').glob('*/scene.json'))


# ---------------------------------------------------------------- generation

def generate(data, counts, start=0):
    """Write scene.json for each (family, index); never rewrite different bytes."""
    rows = []
    for family, count in counts.items():
        for index in range(start, start + count):
            scene = ds.generate(family, index)
            folder = data / 'scenes' / scene['sceneId']
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / 'scene.json'
            payload = ds.dumps(scene)
            if path.exists() and path.read_bytes() != payload:
                raise FileExistsError(f'{path} differs from the generator output; use a new data root.')
            if not path.exists():
                path.write_bytes(payload)
            rows.append(dict(sceneId=scene['sceneId'], family=family, split=scene['split'],
                             splitGroup=scene['splitGroup'], sha256=hashlib.sha256(payload).hexdigest(),
                             receivers=len(scene['points']) // 3, fdtdEligible=scene['fdtd']['eligible']))
    manifest = dict(generator=ds.GENERATOR_VERSION, generatorSha256=sha256(ROOT / 'tools/dlss_scenes.py'),
                    counts=counts, start=start, scenes=rows)
    (data / 'generation.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


# ---------------------------------------------------------------- guarded process

class Worker:
    """Run one low-priority child with an RSS guard and battery suspension."""

    def __init__(self, command, env, log, rss_limit=4 * GIB):
        self.command, self.env, self.log, self.rss_limit = command, env, log, rss_limit
        self.samples, self.paused_seconds = [], 0.0

    def tree_rss(self, pid):
        out = subprocess.run(['ps', '-A', '-o', 'pid=,ppid=,rss='], capture_output=True, text=True).stdout.split()
        table = np.array(out, dtype=np.int64).reshape(-1, 3) if out else np.zeros((0, 3), np.int64)
        members, frontier = {pid}, [pid]
        while frontier:
            children = table[np.isin(table[:, 1], frontier), 0].tolist()
            frontier = [c for c in children if c not in members]
            members.update(frontier)
        return int(table[np.isin(table[:, 0], list(members)), 2].sum()) * 1024

    def run(self):
        process = subprocess.Popen(self.command, env=self.env, start_new_session=True)
        awake = subprocess.Popen(['caffeinate', '-i', '-w', str(process.pid)])
        started, suspended = time.monotonic(), False
        previous = {s: signal.signal(s, self._interrupt) for s in [signal.SIGINT, signal.SIGTERM]}
        try:
            last_power = 0.0
            while process.poll() is None:
                rss = self.tree_rss(process.pid)
                self.samples.append(dict(seconds=round(time.monotonic() - started, 1), treeRssBytes=rss))
                if rss > self.rss_limit:
                    raise MemoryError(f'Worker tree RSS {rss / GIB:.2f} GiB exceeded the guard.')
                if time.time() - last_power > 30:
                    last_power = time.time()
                    mains = on_mains()
                    if not mains and not suspended:
                        os.killpg(process.pid, signal.SIGSTOP)
                        suspended, paused_at = True, time.monotonic()
                        self.log(f'suspended worker on battery at {time.strftime("%H:%M:%S")}')
                    elif mains and suspended:
                        os.killpg(process.pid, signal.SIGCONT)
                        suspended = False
                        self.paused_seconds += time.monotonic() - paused_at
                        self.log(f'resumed worker on AC power at {time.strftime("%H:%M:%S")}')
                time.sleep(5)
        finally:
            if process.poll() is None:
                if suspended:
                    os.killpg(process.pid, signal.SIGCONT)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            awake.wait(timeout=10)
            for s, h in previous.items():
                signal.signal(s, h)
        return dict(exitCode=process.returncode, wallSeconds=time.monotonic() - started,
                    pausedSeconds=self.paused_seconds, peakTreeRssBytes=max([s['treeRssBytes'] for s in self.samples] or [0]),
                    samples=self.samples[::6])

    @staticmethod
    def _interrupt(signum, frame):
        raise KeyboardInterrupt


# ---------------------------------------------------------------- Unity tracer

def sync_runner(runner):
    """Copy runtime, shader, test and tool sources into the isolated runner."""
    if runner.resolve() == ROOT or (runner / 'Temp/UnityLockfile').exists():
        raise RuntimeError('Use a closed, isolated Unity runner; never the working editor.')
    files = {}
    for folder in SYNCED:
        for source in (ROOT / folder).rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts:
                continue
            target = runner / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or source.read_bytes() != target.read_bytes():
                shutil.copy2(source, target)
            if source.suffix != '.meta':
                files[str(source.relative_to(ROOT))] = sha256(source)
    for preset in PRESETS:   # Material presets are read, never synced: they must already match.
        if sha256(ROOT / preset) != sha256(runner / preset):
            raise RuntimeError(f'Runner material preset differs: {preset}')
        files[preset] = sha256(ROOT / preset)
    return files


def render(data, scenes, batch, work_fraction, variant='tracer', runner=RUNNER, unity=UNITY, rss_limit=4 * GIB):
    """Render pending scenes in batches; each batch is one Unity launch."""
    pending = [s for s in scenes if not (s / variant).exists()]
    logs = data / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    print(f'{len(pending)} scenes pending', flush=True)
    for first in range(0, len(pending), batch):
        wait_for_mains()
        group = pending[first:first + batch]
        name = time.strftime('render-%Y%m%d-%H%M%S')
        out = logs / name
        out.mkdir()
        provenance = dict(files=sync_runner(runner), runner=str(runner), unity=str(unity),
                          scenes={s.name: sha256(s / 'scene.json') for s in group})
        (out / 'provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')
        jobs = out / 'jobs.txt'
        jobs.write_text(''.join(f'{s.resolve()}\t{variant}\n' for s in group))
        command = ['nice', '-n', '19', str(unity), '-batchmode', '-force-metal', '-disable-audio',
                   '-job-worker-count', '1', '-projectPath', str(runner), '-runTests', '-testPlatform', 'EditMode',
                   '-testFilter', 'TakeChecks.MeasureDatasetScenes', '-testResults', str(out / 'tests.xml'),
                   '-logFile', str(out / 'unity.log')]
        env = dict(os.environ, ACOUSTIC_DATASET_JOBS=str(jobs), ACOUSTIC_DATASET_WORK_FRACTION=str(work_fraction),
                   ACOUSTIC_TEST_WORK_FRACTION=str(work_fraction), ACOUSTIC_SOLID_PYTHON=str(SOLID_PYTHON))
        print(f'{name}: {variant}, {len(group)} scenes {[s.name for s in group]}', flush=True)
        resources = Worker(command, env, lambda m: print(m, flush=True), rss_limit).run()
        passed = (out / 'tests.xml').exists() and xml.parse(out / 'tests.xml').getroot().get('result') == 'Passed'
        resources.update(passed=passed, workFraction=work_fraction)
        (out / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
        done = [s for s in group if (s / variant).exists()]
        for s in done:
            if not (s / variant / 'provenance.json').exists():
                record = dict(batch=name, provenanceSha256=sha256(out / 'provenance.json'),
                              resourcesSha256=sha256(out / 'resources.json'))
                (s / variant / 'provenance.json').write_text(json.dumps(record, indent=1) + '\n')
        print(f'{name}: exit {resources["exitCode"]}, passed={passed}, {len(done)}/{len(group)} done, '
              f'{resources["wallSeconds"]:.0f} s, peak RSS {resources["peakTreeRssBytes"] / GIB:.2f} GiB', flush=True)
        if not passed:
            raise RuntimeError(f'Unity batch {name} failed; see {out / "unity.log"}')
        time.sleep(20)


def run_test(name, filter_, work_fraction, runner=RUNNER, unity=UNITY):
    """Run focused Unity checks in the isolated runner under the same guards."""
    out = DATA / 'logs' / time.strftime(f'{name}-%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    (out / 'provenance.json').write_text(json.dumps(dict(files=sync_runner(runner)), indent=1) + '\n')
    command = ['nice', '-n', '19', str(unity), '-batchmode', '-force-metal', '-disable-audio', '-job-worker-count', '1',
               '-projectPath', str(runner), '-runTests', '-testPlatform', 'EditMode', '-testFilter', filter_,
               '-testResults', str(out / 'tests.xml'), '-logFile', str(out / 'unity.log')]
    env = dict(os.environ, ACOUSTIC_TEST_WORK_FRACTION=str(work_fraction), ACOUSTIC_SOLID_PYTHON=str(SOLID_PYTHON))
    resources = Worker(command, env, lambda m: print(m, flush=True)).run()
    (out / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    root = xml.parse(out / 'tests.xml').getroot() if (out / 'tests.xml').exists() else None
    summary = {k: root.get(k) for k in ['result', 'total', 'passed', 'failed', 'skipped']} if root is not None else None
    print(json.dumps(dict(log=str(out), summary=summary), indent=1))
    return summary


# ---------------------------------------------------------------- FDTD subset

def fdtd_select(data, count):
    """Cheapest eligible scenes, balanced across eligible families (frozen rule)."""
    by_family = {}
    for folder in scene_dirs(data):
        scene = json.loads((folder / 'scene.json').read_text())
        if scene['fdtd']['eligible']:
            by_family.setdefault(scene['family'], []).append((scene['fdtd']['gcellUpdates'], folder))
    for rows in by_family.values():
        rows.sort()
    chosen, rank = [], 0
    while len(chosen) < count and any(rank < len(r) for r in by_family.values()):
        for family in sorted(by_family):
            if rank < len(by_family[family]) and len(chosen) < count:
                chosen.append(by_family[family][rank][1])
        rank += 1
    return chosen


def fdtd(data, folders, work_fraction, memory_gib=4.0, spacing=None, tag='fdtd'):
    """Solve each scene's FDTD plan; outputs are written atomically to <scene>/<tag>/."""
    import fdtd_wave as fw
    power = PowerCheck()
    for folder in folders:
        final, partial = folder / tag, folder / f'{tag}.partial'
        if final.exists():
            continue
        wait_for_mains()
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        scene = json.loads((folder / 'scene.json').read_text())
        plan = scene['fdtd']
        h = spacing or plan['spacing']
        started = time.time()

        def progress(n, total, last=[0.0]):
            if time.time() - last[0] > 120:
                last[0] = time.time()
                print(f'{folder.name}: {n}/{total} steps, {time.time() - started:.0f} s', flush=True)
        print(f'{folder.name}: FDTD h={h} duration={plan["duration"]} s', flush=True)
        out = fw.solve(scene, h, plan['frequencies'], plan['duration'], plan['peakHz'], pml=plan['pml'],
                       margin=plan['margin'], work_fraction=work_fraction, memory_limit=memory_gib * GIB,
                       progress=progress, pause=power)
        meta = out['meta']
        times = np.array([.5, .75, 1.0]) * plan['duration']
        curve = np.stack([fw.transfer_curve(out['pressure'], out['source'], meta['dt'], hz, times)
                          for hz in plan['frequencies']], axis=2)          # [T, P, Q]
        wave = out['H'].T                                                    # [P, Q]
        decimate = max(1, int(1 / (4000 * meta['dt'])))
        pressure = out['pressure'][::decimate].astype('<f4')
        np.stack([wave.real, wave.imag], -1).astype('<f4').tofile(partial / 'wave.f32')
        np.stack([curve.real, curve.imag], -1).astype('<f4').tofile(partial / 'truncation.f32')
        pressure.tofile(partial / 'pressure.f32')
        out['source'].astype('<f4').tofile(partial / 'source.f32')
        change = np.linalg.norm(curve[1] - curve[2], axis=0) / np.linalg.norm(curve[2], axis=0)
        run = dict(schema='dlss-fdtd-run-v1', sceneId=scene['sceneId'], sceneSha256=sha256(folder / 'scene.json'),
                   solverSha256=sha256(ROOT / 'tools/fdtd_wave.py'), frequencies=plan['frequencies'],
                   truncationTimes=times.tolist(), truncationChange075=change.tolist(),
                   decimation=decimate, pressureShape=list(pressure.shape), sourceShape=[len(out['source'])],
                   workFraction=work_fraction, elapsedSeconds=time.time() - started, **meta)
        (partial / 'run.json').write_text(json.dumps(run, indent=1) + '\n')
        partial.rename(final)
        t = meta['timing']
        print(f'{folder.name}: done, GPU {t["gpuSeconds"]:.0f} s, wall {t["wallSeconds"]:.0f} s, '
              f'{t["gcellPerSecond"]:.2f} Gcell/s, truncation change {np.round(100 * change, 3)}%', flush=True)
        del out
        fw.torch.mps.empty_cache()
        time.sleep(20)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=DATA)
    sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--count', type=int, default=12, help='scenes per family')
    g.add_argument('--families', nargs='+', default=list(ds.FAMILIES))
    g.add_argument('--start', type=int, default=0, help='first scene index (calibration uses a disjoint range)')
    r = sub.add_parser('render')
    r.add_argument('--batch', type=int, default=6)
    r.add_argument('--scenes', nargs='*')
    r.add_argument('--work-fraction', type=float, default=.2)
    r.add_argument('--variant', choices=['tracer', 'tracer-rigid'], default='tracer')
    r.add_argument('--fdtd-subset', type=int, help='render only the FDTD-selected scenes')
    t = sub.add_parser('test')
    t.add_argument('filter')
    t.add_argument('--work-fraction', type=float, default=.2)
    f = sub.add_parser('fdtd')
    f.add_argument('--select', type=int, default=24)
    f.add_argument('--scenes', nargs='*')
    f.add_argument('--spacing', type=float)
    f.add_argument('--tag', default='fdtd')
    f.add_argument('--work-fraction', type=float, default=.2)
    f.add_argument('--list', action='store_true')
    a = p.parse_args()
    data = a.data.resolve()
    if a.command == 'generate':
        manifest = generate(data, {family: a.count for family in a.families}, a.start)
        print(json.dumps({k: v for k, v in manifest.items() if k != 'scenes'}, indent=1))
    elif a.command == 'render':
        folders = [data / 'scenes' / s for s in a.scenes] if a.scenes else \
            fdtd_select(data, a.fdtd_subset) if a.fdtd_subset else scene_dirs(data)
        render(data, folders, a.batch, a.work_fraction, a.variant)
    elif a.command == 'test':
        run_test('test', a.filter, a.work_fraction)
    elif a.command == 'fdtd':
        folders = [data / 'scenes' / s for s in a.scenes] if a.scenes else fdtd_select(data, a.select)
        if a.list:
            for folder in folders:
                plan = json.loads((folder / 'scene.json').read_text())['fdtd']
                print(folder.name, plan['cells'], round(plan['gcellUpdates']))
            return
        fdtd(data, folders, a.work_fraction, spacing=a.spacing, tag=a.tag)


if __name__ == '__main__':
    main()
