"""Guarded Unity jobs for the late-energy benchmark.

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY docs/benchmarks/late-energy-2026-11-04/unity.py segment TAKE OUT START END
    nice -n 19 $PY docs/benchmarks/late-energy-2026-11-04/unity.py render JOB.json OUTPUT
    nice -n 19 $PY docs/benchmarks/late-energy-2026-11-04/unity.py tests FILTER [ENV=VALUE ...]

Reuses tools/dlss_jobs.py: isolated runner sync, nice 19, one job worker,
20% work pacing, 4 GiB process-tree RSS guard and suspension on battery.
Logs go to .context/late-energy/logs.
"""
import json
import os
import subprocess
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as xml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_jobs as dj  # noqa: E402

LOGS = ROOT / '.context/late-energy/logs'


def clear_stale_lock():
    """A batch Unity that aborts can leave its lock file; remove it only if no process owns the runner."""
    lock = dj.RUNNER / 'Temp/UnityLockfile'
    if lock.exists():
        processes = subprocess.run(['ps', '-A', '-o', 'command='], capture_output=True, text=True).stdout
        if str(dj.RUNNER) not in processes:
            lock.unlink()


def prune_runner():
    """sync_runner copies new and changed files; remove runner sources deleted from the repository."""
    removed = []
    for folder in [f for f in dj.SYNCED if f.startswith('Assets/')]:
        for target in (dj.RUNNER / folder).rglob('*'):
            if target.is_file() and not (ROOT / target.relative_to(dj.RUNNER)).exists():
                source = target.with_suffix('') if target.suffix == '.meta' else target
                if not (ROOT / source.relative_to(dj.RUNNER)).exists():
                    target.unlink()
                    removed.append(str(target.relative_to(dj.RUNNER)))
    return removed


def launch(name, arguments, env=None, work_fraction=.2):
    logs = LOGS / time.strftime(f'{name}-%Y%m%d-%H%M%S')
    logs.mkdir(parents=True)
    dj.wait_for_mains()
    clear_stale_lock()
    removed = prune_runner()
    if removed:
        print('removed from runner:', removed, flush=True)
    (logs / 'provenance.json').write_text(json.dumps(dict(files=dj.sync_runner(dj.RUNNER)), indent=1) + '\n')
    command = ['nice', '-n', '19', str(dj.UNITY), '-batchmode', '-force-metal', '-disable-audio', '-job-worker-count', '1',
               '-projectPath', str(dj.RUNNER)] + arguments + ['-logFile', str(logs / 'unity.log')]
    environment = dict(os.environ, ACOUSTIC_TEST_WORK_FRACTION=str(work_fraction),
                       ACOUSTIC_SOLID_PYTHON=str(dj.SOLID_PYTHON), **(env or {}))
    print(f'{logs.name}: {" ".join(arguments)}', flush=True)
    resources = dj.Worker(command, environment, lambda m: print(m, flush=True)).run()
    (logs / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    print(f'{logs.name}: exit {resources["exitCode"]}, {resources["wallSeconds"]:.0f} s, '
          f'peak RSS {resources["peakTreeRssBytes"] / dj.GIB:.2f} GiB', flush=True)
    return logs, resources


def segment(take, output, start, end):
    logs, resources = launch('segment', ['-quit', '-executeMethod', 'LateEnergyMeasurement.DeriveSegment',
                                         '-segmentTake', str(Path(take).resolve()), '-segmentOutput', str(Path(output).resolve()),
                                         '-segmentStart', str(start), '-segmentEnd', str(end)])
    if resources['exitCode'] != 0 or not (Path(output) / 'take.json').exists():
        raise SystemExit(f'Segment failed; see {logs}/unity.log')


def render(job_file, output, work_fraction=.2):
    job = json.loads(Path(job_file).read_text())
    job['workFraction'] = work_fraction
    job['take'] = str((Path(job_file).parent / job['take']).resolve())
    job['output'] = str(Path(output).resolve())
    if Path(job['output']).exists():
        raise SystemExit('Render output must be new.')
    guarded = Path(job['output']).with_suffix('.job.json')
    guarded.parent.mkdir(parents=True, exist_ok=True)
    guarded.write_text(json.dumps(job, indent=2) + '\n')
    logs, resources = launch('render', ['-executeMethod', 'TakeBatch.Render', '-takeJob', str(guarded)],
                             work_fraction=work_fraction)
    (Path(job['output']) / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    if resources['exitCode'] != 0:
        raise SystemExit(f'Render failed; see {logs}/unity.log')


def tests(filter_, env):
    logs = LOGS / 'pending'
    arguments = ['-runTests', '-testPlatform', 'EditMode', '-testFilter', filter_]
    logs, resources = launch('tests', arguments + ['-testResults', str(LOGS / '.results.xml')], env)
    results = LOGS / '.results.xml'
    if results.exists():
        results.rename(logs / 'tests.xml')
        root = xml.parse(logs / 'tests.xml').getroot()
        print(json.dumps(dict(log=str(logs), summary={k: root.get(k) for k in ['result', 'total', 'passed', 'failed', 'skipped']})),
              flush=True)
        for case in root.iter('test-case'):
            if case.get('result') == 'Failed':
                message = case.find('failure/message')
                print('FAILED', case.get('fullname'), (message.text or '')[:600] if message is not None else '', flush=True)
    else:
        print(f'No test results; see {logs}/unity.log', flush=True)


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'segment':
        segment(*sys.argv[2:6])
    elif command == 'render':
        render(sys.argv[2], sys.argv[3])
    elif command == 'tests':
        tests(sys.argv[2], dict(a.split('=', 1) for a in sys.argv[3:]))
    else:
        raise SystemExit(__doc__)
