"""The playable Church (README.md, "Playable Church"; docs/benchmarks/path-denoise-2026-11-12, part 26).

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY tools/path_live_church.py build   # guarded Unity build -> Builds/ChurchLive/ChurchLive.app
    nice -n 19 $PY tools/path_live_church.py smoke   # guarded scripted run of the build -> .context/path-live/smoke-*/

The build comes from the test runner's copy of the project (tools/dlss_jobs.py RUNNER, synced from this
working tree), so the editor's Library is untouched. The smoke run launches the build windowed with
-pathLiveSmoke: the scripted walk of PathLiveChurch (stand, walk to the doors, open them, walk out), the
output recorded to output.wav, frame times, GPU times and audio underruns to smoke.json. It runs windowless
(-batchmode) so it never takes the screen's focus; arguments after `smoke --` go to the player (e.g. -pathBudget 6).
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
APP = ROOT / 'Builds/ChurchLive/ChurchLive.app'


def build():
    mr.LOGS = WORK / 'logs'; mr.LOGS.mkdir(parents=True, exist_ok=True)
    staged = WORK / 'build/ChurchLive.app'
    if staged.exists():
        shutil.rmtree(staged)
    logs, resources = mr.launch('build', ['-quit', '-executeMethod', 'PathLiveChurchBuild.Build', '-pathLiveOutput', str(staged)])
    if resources['exitCode'] != 0 or not staged.exists():
        raise SystemExit(f'build failed: see {logs}/unity.log')
    if APP.exists():
        shutil.rmtree(APP)
    APP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staged, APP, symlinks=True)
    print(f'built {APP}')


def front_app():
    """The frontmost application's name (lsappinfo: read only, needs no permission)."""
    import subprocess
    asn = subprocess.run(['lsappinfo', 'front'], capture_output=True, text=True).stdout.strip()
    info = subprocess.run(['lsappinfo', 'info', '-only', 'name', asn], capture_output=True, text=True).stdout
    return info.split('=')[-1].strip().strip('"')


def smoke(extra=(), windowed=False):
    out = WORK / time.strftime('smoke-%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    binary = next((APP / 'Contents/MacOS').iterdir())
    # -batchmode: no window, so the run never takes the screen's focus; the player still draws the scene every frame
    # (smoke.json reports camera renders per frame) and still plays its audio callbacks (callbacks, underruns).
    command = [str(binary), '-batchmode', '-pathLiveSmoke', str(out), '-screen-width', '1280', '-screen-height', '720',
               '-logFile', str(out / 'player.log'), *extra]
    if windowed:
        # A window, so that the scene is drawn and the profiler's GPU timings can be read: launched hidden and not
        # activated (open -g -j). The front application is watched throughout; if the player ever became frontmost it
        # is stopped at once (it must never take the screen's focus). Its footprint is not tracked (open's child).
        import subprocess, threading
        command = ['open', '-g', '-j', '-n', '-W', '-a', str(APP), '--args'] + [c for c in command[1:] if c != '-batchmode']
        before, stolen, done = front_app(), [], threading.Event()
        def watch():
            while not done.wait(.2):
                name = front_app()
                if name != before and ('ChurchLive' in name or 'AudioRayTrace' in name):
                    stolen.append(name); subprocess.run(['pkill', '-f', str(APP)]); return
        threading.Thread(target=watch, daemon=True).start()
    resources = dj.Worker(command, dict(os.environ), lambda m: print(m, flush=True)).run()
    if windowed:
        done.set()
        resources['focusTaken'] = bool(stolen)
        print('the player took the focus and was stopped' if stolen else f'focus stayed with {before}')
    (out / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    report = out / 'smoke.json'
    print(f'exit {resources["exitCode"]}, {resources["wallSeconds"]:.0f} s, peak footprint {resources["peakTreeFootprintBytes"] / dj.GIB:.2f} GiB')
    print(report.read_text() if report.exists() else f'no report; see {out}/player.log')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['build', 'smoke'])
    p.add_argument('player', nargs='*', help='smoke: extra player arguments (after --)')
    p.add_argument('--windowed', action='store_true', help='smoke: with a window, launched hidden and unfocused (open -g -j)')
    a = p.parse_args()
    build() if a.command == 'build' else smoke(a.player, a.windowed)
