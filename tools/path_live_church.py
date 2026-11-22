"""The live showcase: the path method live in the Church and the train station (README.md, "Live showcase";
docs/benchmarks/path-denoise-2026-11-12, parts 26 and 38).

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY tools/path_live_church.py build [--development]
                                                      # guarded Unity build -> Builds/LiveShowcase/LiveShowcase.app
                                                      # (a development player -> Builds/LiveShowcaseDevelopment/: per-stage GPU times)
    nice -n 19 $PY tools/path_live_church.py smoke --windowed [--scenes Church TrainStation] [--shots] [--development]
                                                      # guarded scripted run of the build -> .context/path-live/smoke-*/
    nice -n 19 $PY tools/path_live_church.py stills   # guarded still images of every scene -> .context/polish/stills
    nice -n 19 $PY tools/path_live_church.py author   # guarded ShowcaseAuthoring of the showcase scenes

The build comes from the test runner's copy of the project (tools/dlss_jobs.py RUNNER, synced from this
working tree), so the editor's Library is untouched. The smoke run launches the build with -pathLiveSmoke:
each scene's scripted walk (PathLive, PathLiveScene), one scene after another, loaded as the scene picker loads
them. Per scene the output is recorded to output.wav, frame times, GPU times and audio underruns to smoke.json,
and with --shots a screenshot is saved at each phase of the walk. --windowed launches it hidden and unfocused;
without it, it runs windowless (-batchmode) and draws nothing. Arguments after `smoke --` go to the player
(e.g. -pathBudget 6).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_jobs as dj  # noqa: E402
import method_runs as mr  # noqa: E402

WORK = ROOT / '.context/path-live'
APP = ROOT / 'Builds/LiveShowcase/LiveShowcase.app'
DEVELOPMENT_APP = ROOT / 'Builds/LiveShowcaseDevelopment/LiveShowcase.app'
OLD_APP = ROOT / 'Builds/ChurchLive'   # the Church-only app this replaces; removed by the next build (disk space)
# The scenes and their visual content. dlss_jobs syncs only code, shaders, tests and tools into the runner, so the
# player build and the stills would otherwise use the runner's older copies of these.
CONTENT = ['Assets/Scenes', 'Assets/Materials', 'Assets/Prefabs', 'Assets/External', 'Assets/Audio']


def sync_content():
    """Mirror CONTENT into the runner: copy new or changed files, remove files deleted here."""
    for folder in CONTENT:
        for source in (ROOT / folder).rglob('*'):
            target = dj.RUNNER / source.relative_to(ROOT)
            if source.is_file() and (not target.exists() or source.read_bytes() != target.read_bytes()):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        for target in sorted((dj.RUNNER / folder).rglob('*'), reverse=True):
            if not (ROOT / target.relative_to(dj.RUNNER)).exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()


def launch(name, arguments):
    """A guarded batch editor run in the runner, its scenes and content synced first."""
    mr.LOGS = WORK / 'logs'; mr.LOGS.mkdir(parents=True, exist_ok=True)
    sync_content()
    return mr.launch(name, arguments)


def stills(scenes):
    """Still images of the scenes (SceneStills) for a visual review: .context/polish/stills."""
    out = ROOT / '.context/polish/stills'
    shutil.rmtree(out, ignore_errors=True)
    logs, resources = launch('stills', ['-quit', '-executeMethod', 'SceneStills.Capture', '-stillsOutput', str(out),
                                        *(['-stillsScenes', ','.join(scenes)] if scenes else [])])
    print(f'{len(list(out.glob("*.png")))} stills in {out}' if resources['exitCode'] == 0 else f'stills failed: see {logs}/unity.log')


def author():
    """ShowcaseAuthoring in the runner, then its scenes (and the metas of new scripts) copied back here."""
    logs, resources = launch('author', ['-quit', '-executeMethod', 'ShowcaseAuthoring.Apply'])
    if resources['exitCode'] != 0:
        raise SystemExit(f'authoring failed: see {logs}/unity.log')
    for scene in ['Assets/Scenes/Church.unity', 'Assets/Scenes/TrainStation.unity']:
        shutil.copy2(dj.RUNNER / scene, ROOT / scene)
    for folder in ['Assets/Scripts', 'Assets/Editor']:
        for meta in (dj.RUNNER / folder).rglob('*.meta'):
            target = ROOT / meta.relative_to(dj.RUNNER)
            if not target.exists() and target.with_suffix('').exists():
                shutil.copy2(meta, target)
    print('authored the showcase scenes')


def build(development=False):
    app = DEVELOPMENT_APP if development else APP
    staged = WORK / 'build' / app.name
    shutil.rmtree(WORK / 'build', ignore_errors=True)
    logs, resources = launch('build', ['-quit', '-executeMethod', 'PathLiveBuild.Build', '-pathLiveOutput', str(staged),
                                       *(['-pathLiveDevelopment'] if development else [])])
    if resources['exitCode'] != 0 or not staged.exists():
        raise SystemExit(f'build failed: see {logs}/unity.log')
    if app.exists():
        shutil.rmtree(app)
    app.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staged, app, symlinks=True)
    shutil.rmtree(WORK / 'build')
    shutil.rmtree(OLD_APP, ignore_errors=True)
    print(f'built {app}')


def front_app():
    """The frontmost application's name (lsappinfo: read only, needs no permission)."""
    import subprocess
    asn = subprocess.run(['lsappinfo', 'front'], capture_output=True, text=True).stdout.strip()
    info = subprocess.run(['lsappinfo', 'info', '-only', 'name', asn], capture_output=True, text=True).stdout
    return info.split('=')[-1].strip().strip('"')


def smoke(extra=(), windowed=False, scenes=(), shots=False, development=False):
    app = DEVELOPMENT_APP if development else APP
    out = WORK / time.strftime('smoke-%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    binary = next((app / 'Contents/MacOS').iterdir())
    # -batchmode: no window, so the run never takes the screen's focus; the player still draws the scene every frame
    # (smoke.json reports camera renders per frame) and still plays its audio callbacks (callbacks, underruns).
    command = [str(binary), '-batchmode', '-pathLiveSmoke', str(out), '-screen-width', '1280', '-screen-height', '720',
               '-logFile', str(out / 'player.log'), '-liveTrace', str(out / 'pipeline.csv'),
               *(['-pathLiveScenes', ','.join(scenes)] if scenes else []), *(['-pathLiveShots'] if shots else []), *extra]
    if windowed:
        # A window, so that the scene is drawn and the profiler's GPU timings can be read: launched hidden and not
        # activated (open -g -j). The front application is watched throughout; if the player ever became frontmost it
        # is stopped at once (it must never take the screen's focus). Its footprint is not tracked (open's child).
        import subprocess, threading
        command = ['open', '-g', '-j', '-n', '-W', '-a', str(app), '--args'] + [c for c in command[1:] if c != '-batchmode']
        before, stolen, done = front_app(), [], threading.Event()
        def watch():
            while not done.wait(.2):
                name = front_app()
                if name != before and (app.stem in name or 'AudioRayTrace' in name):
                    stolen.append(name); subprocess.run(['pkill', '-f', str(app)]); return
        threading.Thread(target=watch, daemon=True).start()
    resources = dj.Worker(command, dict(os.environ), lambda m: print(m, flush=True)).run()
    if windowed:
        done.set()
        resources['focusTaken'] = bool(stolen)
        print('the player took the focus and was stopped' if stolen else f'focus stayed with {before}')
    (out / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    print(f'exit {resources["exitCode"]}, {resources["wallSeconds"]:.0f} s, peak footprint {resources["peakTreeFootprintBytes"] / dj.GIB:.2f} GiB')
    reports = sorted(out.glob('*/smoke.json'))
    for report in reports:
        r = json.loads(report.read_text())
        print(f'{r["scene"]}: {r["seconds"]:.0f} s, underruns {r["underrunSamples"]} samples, frames {r["frameMsMedian"]:.2f} / '
              f'{r["frameMsP99"]:.1f} / {r["frameMsMax"]:.1f} ms (median / p99 / max), frame GPU {r["frameGpuMsMedian"]:.1f} ms median, '
              f'{r["cameraRendersPerFrame"]:.2f} camera renders per frame; {report}')
    stalls(out / 'pipeline.csv.stalls')
    if len(reports) < max(1, len(scenes)):
        print(f'{max(1, len(scenes)) - len(reports)} scene(s) without a report; see {out}/player.log')


def stalls(path):
    """The main-thread stalls of the run (-liveTrace FILE.stalls, README part 41), each with its kind: computing (the
    main thread's CPU time is most of it), paging (blocked, with page faults that needed I/O), or blocked (waiting or
    preempted). Scene loads and the end of each take are stalls too; they are not in a walk."""
    import csv
    if not path.exists():
        return
    for row in csv.DictReader(path.open()):
        ms, cpu = float(row['ms']), float(row.get('mainThreadCpuMs') or 'nan')
        kind = ('computing' if cpu > .5 * ms else 'paging' if int(row.get('majorFaults') or 0) > 0 else 'blocked') if cpu == cpu else '?'
        print(f"stall {row['seconds']} s, frame {row['frame']}: {row['system']} {ms:.0f} ms, {kind} (main thread CPU {cpu:.0f} ms, "
              f"{row.get('majorFaults')} page-ins, {row.get('involuntarySwitches')} preemptions)")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['build', 'smoke', 'stills', 'author'])
    p.add_argument('player', nargs='*', help='smoke: extra player arguments (after --); stills: scene names')
    p.add_argument('--windowed', action='store_true', help='smoke: with a window, launched hidden and unfocused (open -g -j)')
    p.add_argument('--scenes', nargs='+', default=[], help='smoke: the scenes to walk, in turn (default: the first)')
    p.add_argument('--shots', action='store_true', help='smoke: a screenshot at each phase of each walk')
    p.add_argument('--development', action='store_true', help='build / smoke: the development player (per-stage GPU times)')
    a = p.parse_args()
    {'build': lambda: build(a.development), 'smoke': lambda: smoke(a.player, a.windowed, a.scenes, a.shots, a.development), 'stills': lambda: stills(a.player),
     'author': author}[a.command]()
