"""Guarded workers shared by the Unity and wave tools: the heavy-job worker, the
isolated Unity runner, scene generation and the FDTD wave subset.

Every heavy job runs at nice 19 with one worker, 50% work pacing, a memory guard
and a pause while the Mac runs on battery (the FDTD study's power hook; Unity
workers are suspended with SIGSTOP and resumed with SIGCONT). Bulk outputs stay
under the data root (default .context/audio-dlss-data), never in Assets/.

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY tools/dlss_jobs.py generate --count 12
    nice -n 19 $PY tools/dlss_jobs.py fdtd --select 24
"""
import argparse
import hashlib
import json
import os
import re
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
import heavy_guard as hg  # noqa: E402

DATA = ROOT / '.context/audio-dlss-data'
# On the PC (WSL) Unity is the Windows editor, and its runner sits on the Windows drive (a \\wsl$ project imports slowly).
RUNNER = hg.WORKTREE_WORK / 'runner' if hg.WSL else ROOT / '.context/gpu-audio/runner'
UNITY = (Path('/mnt/c/Program Files/Unity/Hub/Editor/6000.4.1f1/Editor/Unity.exe') if hg.WSL else
         Path('/Applications/Unity/Hub/Editor/6000.4.1f1/Unity.app/Contents/MacOS/Unity'))
GRAPHICS = '-force-d3d12' if hg.WSL else '-force-metal'
# The path walk's megakernel takes FXC (D3D) or Metal's compiler past Unity's default 10-minute timeout per variant;
# every Unity this process launches inherits the longer one.
os.environ.setdefault('UNITY_SHADER_COMPILER_TASK_TIMEOUT_MINUTES', '60')
SOLID_PYTHON = ROOT / '.context/solid-python/bin/python'
GIB = 1024 ** 3
SYNCED = ['Assets/Scripts', 'Assets/Shaders', 'Assets/Resources/Acoustics', 'Assets/Tests/Editor', 'Assets/Editor', 'tools']
RUNNER_OWN = ['Library', 'Temp', 'Logs', 'UserSettings', 'obj', 'Builds', '.context', '.git', '.vs']   # never mirrored


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# On battery, heavy work runs only above this charge; HEAVY_MIN_BATTERY overrides it for one run.
MIN_BATTERY_PERCENT = int(os.environ.get('HEAVY_MIN_BATTERY', 40))


def on_mains():
    """True when heavy work may run: on AC power, or on battery above MIN_BATTERY_PERCENT."""
    if hg.WSL:
        snap = hg.windows().snapshot()
        return bool(snap['ac']) or snap['charge'] >= MIN_BATTERY_PERCENT
    out = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout
    if 'AC Power' in out:
        return True
    charge = re.search(r'(\d+)%', out)
    return charge is not None and int(charge.group(1)) >= MIN_BATTERY_PERCENT


def wait_for_mains(log=print):
    """Block while the battery is low (below MIN_BATTERY_PERCENT and not on AC)."""
    told = False
    while not on_mains():
        if not told:
            log(f'paused: battery below {MIN_BATTERY_PERCENT}% at {time.strftime("%H:%M:%S")}; plug in to continue')
            told = True
        time.sleep(60)
    if told:
        log(f'resumed at {time.strftime("%H:%M:%S")}')


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
    """Run one low-priority child under the machine-wide heavy-job lock, with a
    physical-footprint guard (GPU memory included), a system memory/swap guard and
    battery suspension."""

    def __init__(self, command, env, log, rss_limit=4 * GIB):
        self.command, self.env, self.log, self.limit = command, env, log, rss_limit
        self.samples, self.paused_seconds = [], 0.0

    def run(self):
        with hg.HeavyLock(self.log):
            return self._run()

    def _run(self):
        watch = hg.SystemWatch()
        command, env = hg.host_command(self.command, dict(self.env or os.environ, **{hg.HELD_ENV: '1'}))
        process = subprocess.Popen(command, env=env, start_new_session=True)
        awake = keep_awake(process)
        started, suspended = time.monotonic(), False
        jobs = hg.job_tree(process, command)
        previous = {s: signal.signal(s, self._interrupt) for s in [signal.SIGINT, signal.SIGTERM]}
        try:
            last_power, polls = 0.0, 0
            while process.poll() is None:
                used = jobs.footprint()
                polls += 1
                if polls % 4 == 0 or used > self.limit:
                    self.samples.append(dict(seconds=round(time.monotonic() - started, 1), treeFootprintBytes=used))
                if used > self.limit:
                    jobs.kill()
                    raise MemoryError(f'Worker footprint {used / GIB:.2f} GiB exceeded the {self.limit / GIB:.1f} GiB guard.')
                if not suspended:   # pause while other apps exhaust memory; resume afterwards
                    self.paused_seconds += watch.wait_out(
                        self.log, pause=lambda: jobs.signal(signal.SIGSTOP),
                        resume=lambda: jobs.signal(signal.SIGCONT))
                if time.time() - last_power > 30:
                    last_power = time.time()
                    mains = on_mains()
                    if not mains and not suspended:
                        jobs.signal(signal.SIGSTOP)
                        awake and awake.terminate()   # Let the Mac sleep while the worker waits for AC power.
                        suspended, paused_at = True, time.monotonic()
                        self.log(f'suspended worker: battery below {MIN_BATTERY_PERCENT}% at {time.strftime("%H:%M:%S")}')
                    elif mains and suspended:
                        jobs.signal(signal.SIGCONT)
                        awake = keep_awake(process)
                        suspended = False
                        self.paused_seconds += time.monotonic() - paused_at
                        self.log(f'resumed worker at {time.strftime("%H:%M:%S")}')
                time.sleep(jobs.POLL_SECONDS)
        finally:
            jobs.kill(grace=30)   # every spawned process, incl. own-group/reparented helpers
            process.wait()
            awake and awake.wait(timeout=10)
            for s, h in previous.items():
                signal.signal(s, h)
        return dict(exitCode=process.returncode, wallSeconds=time.monotonic() - started,
                    pausedSeconds=self.paused_seconds,
                    peakTreeFootprintBytes=max([s['treeFootprintBytes'] for s in self.samples] or [0]),
                    samples=self.samples[::15])

    @staticmethod
    def _interrupt(signum, frame):
        raise KeyboardInterrupt


def keep_awake(process):
    """Keep the Mac from sleeping while `process` runs (caffeinate); the PC does not sleep on AC."""
    return None if hg.WSL else subprocess.Popen(['caffeinate', '-i', '-w', str(process.pid)])


# ---------------------------------------------------------------- Unity runner

def sync_runner(runner):
    """Mirror the project into the isolated runner (everything but Unity's caches, builds and .context; files deleted
    here are deleted there), clear a lockfile no running editor holds, and return the provenance: the hashes of the
    code, shaders, tests and tools the run uses."""
    if runner.resolve() == ROOT:
        raise RuntimeError('Use an isolated Unity runner; never the working editor.')
    lock = runner / 'Temp/UnityLockfile'
    if lock.exists():
        if hg.WSL:
            native = subprocess.run(['wslpath', '-w', str(runner)], capture_output=True, text=True).stdout.strip()
            held = any(row[4] for row in hg.windows().snapshot(native)['p'])
        else:
            held = str(runner) in subprocess.run(['ps', '-A', '-o', 'command='], capture_output=True, text=True).stdout
        if held:
            raise RuntimeError('The runner is open in a running editor.')
        lock.unlink()
    seed = hg.WINDOWS_WORK / 'runner/Library'
    if hg.WSL and runner != seed.parent and not (runner / 'Library').exists() and seed.exists():
        # A new worktree's runner starts from the main runner's caches: its shaders took FXC 36 minutes cold.
        (runner / 'Library').parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cp', '-a', str(seed), str(runner / 'Library')], check=True)
    runner.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--delete', *[f'--exclude=/{d}/' for d in RUNNER_OWN], '--exclude=__pycache__/',
                    f'{ROOT}/', f'{runner}/'], check=True)
    return {str(f.relative_to(ROOT)): sha256(f) for folder in SYNCED for f in sorted((ROOT / folder).rglob('*'))
            if f.is_file() and f.suffix != '.meta' and '__pycache__' not in f.parts}


def run_test(name, filter_, work_fraction, runner=RUNNER, unity=UNITY):
    """Run focused Unity checks in the isolated runner under the same guards."""
    out = DATA / 'logs' / time.strftime(f'{name}-%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    (out / 'provenance.json').write_text(json.dumps(dict(files=sync_runner(runner)), indent=1) + '\n')
    command = ['nice', '-n', '19', str(unity), '-batchmode', GRAPHICS, '-disable-audio', '-job-worker-count', '1',
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
    t = sub.add_parser('test')
    t.add_argument('filter')
    t.add_argument('--work-fraction', type=float, default=hg.WORK_FRACTION)
    f = sub.add_parser('fdtd')
    f.add_argument('--select', type=int, default=24)
    f.add_argument('--scenes', nargs='*')
    f.add_argument('--spacing', type=float)
    f.add_argument('--tag', default='fdtd')
    f.add_argument('--work-fraction', type=float, default=hg.WORK_FRACTION)
    f.add_argument('--list', action='store_true')
    a = p.parse_args()
    data = a.data.resolve()
    if a.command == 'generate':
        manifest = generate(data, {family: a.count for family in a.families}, a.start)
        print(json.dumps({k: v for k, v in manifest.items() if k != 'scenes'}, indent=1))
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
    hg.gpu_gate()
    main()
