"""The playable Church (README.md, "Playable Church"; docs/benchmarks/path-denoise-2026-11-12, part 26).

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY tools/path_live_church.py build   # guarded Unity build -> Builds/ChurchLive/ChurchLive.app
    nice -n 19 $PY tools/path_live_church.py smoke   # guarded scripted run of the build -> .context/path-live/smoke-*/

The build comes from the test runner's copy of the project (tools/dlss_jobs.py RUNNER, synced from this
working tree), so the editor's Library is untouched. The smoke run launches the build windowed with
-pathLiveSmoke: the scripted walk of PathLiveChurch (stand, walk to the doors, open them, walk out), the
output recorded to output.wav, frame times, GPU times and audio underruns to smoke.json.
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


def smoke():
    out = WORK / time.strftime('smoke-%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    binary = next((APP / 'Contents/MacOS').iterdir())
    command = [str(binary), '-pathLiveSmoke', str(out), '-screen-fullscreen', '0', '-screen-width', '1280', '-screen-height', '720',
               '-logFile', str(out / 'player.log')]
    resources = dj.Worker(command, dict(os.environ), lambda m: print(m, flush=True)).run()
    (out / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    report = out / 'smoke.json'
    print(f'exit {resources["exitCode"]}, {resources["wallSeconds"]:.0f} s, peak footprint {resources["peakTreeFootprintBytes"] / dj.GIB:.2f} GiB')
    print(report.read_text() if report.exists() else f'no report; see {out}/player.log')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['build', 'smoke'])
    dict(build=build, smoke=smoke)[p.parse_args().command]()
