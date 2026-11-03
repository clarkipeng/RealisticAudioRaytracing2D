"""Run residual-walk diagnostic jobs in the isolated Unity runner.

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY docs/benchmarks/residual-walk-2026-11-02/unity_jobs.py jobs.json

jobs.json is a list of WalkJob objects (see Assets/Tests/Editor/WalkDiagnostics.cs).
Reuses tools/dlss_jobs.py guards: nice 19, one job worker, 20% work pacing,
4 GiB process-tree RSS guard and suspension on battery power.
"""
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as xml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_jobs as dj  # noqa: E402


def run(job_file, work_fraction=.2, filter_='TakeChecks.MeasureResidualWalk'):
    jobs = json.loads(Path(job_file).read_text())
    logs = ROOT / '.context/residual-walk/logs' / time.strftime('walk-%Y%m%d-%H%M%S')
    logs.mkdir(parents=True)
    files = []
    for i, job in enumerate(jobs):
        if Path(job['output']).exists():
            continue
        path = logs / f'job-{i}.json'
        path.write_text(json.dumps(job, indent=1))
        files.append(str(path))
    if not files:
        return
    (logs / 'jobs.txt').write_text('\n'.join(files) + '\n')
    dj.wait_for_mains()
    (logs / 'provenance.json').write_text(json.dumps(dict(files=dj.sync_runner(dj.RUNNER)), indent=1) + '\n')
    command = ['nice', '-n', '19', str(dj.UNITY), '-batchmode', '-force-metal', '-disable-audio', '-job-worker-count', '1',
               '-projectPath', str(dj.RUNNER), '-runTests', '-testPlatform', 'EditMode', '-testFilter', filter_,
               '-testResults', str(logs / 'tests.xml'), '-logFile', str(logs / 'unity.log')]
    env = dict(os.environ, ACOUSTIC_WALK_JOBS=str(logs / 'jobs.txt'),
               ACOUSTIC_DATASET_WORK_FRACTION=str(work_fraction), ACOUSTIC_TEST_WORK_FRACTION=str(work_fraction),
               ACOUSTIC_SOLID_PYTHON=str(dj.SOLID_PYTHON))
    print(f'{logs.name}: {len(files)} jobs', flush=True)
    resources = dj.Worker(command, env, lambda m: print(m, flush=True)).run()
    (logs / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    root = xml.parse(logs / 'tests.xml').getroot() if (logs / 'tests.xml').exists() else None
    print(f'{logs.name}: exit {resources["exitCode"]}, result {root.get("result") if root is not None else None}, '
          f'{resources["wallSeconds"]:.0f} s, peak RSS {resources["peakTreeRssBytes"] / dj.GIB:.2f} GiB', flush=True)
    missing = [j['output'] for j in jobs if not Path(j['output']).exists()]
    if missing:
        raise SystemExit(f'Missing outputs: {missing}; see {logs}/unity.log')


def run_cases(case_file, work_fraction=.2, filter_='TakeChecks.MeasureRuntimeCompleteRoutes'):
    """Wave-field fixtures (ACOUSTIC_COMPLETE_WAVE_CASES), e.g. the 125 Hz canopy BEM map."""
    logs = ROOT / '.context/residual-walk/logs' / time.strftime('wave-%Y%m%d-%H%M%S')
    logs.mkdir(parents=True)
    dj.wait_for_mains()
    (logs / 'provenance.json').write_text(json.dumps(dict(files=dj.sync_runner(dj.RUNNER)), indent=1) + '\n')
    command = ['nice', '-n', '19', str(dj.UNITY), '-batchmode', '-force-metal', '-disable-audio', '-job-worker-count', '1',
               '-projectPath', str(dj.RUNNER), '-runTests', '-testPlatform', 'EditMode', '-testFilter', filter_,
               '-testResults', str(logs / 'tests.xml'), '-logFile', str(logs / 'unity.log')]
    env = dict(os.environ, ACOUSTIC_COMPLETE_WAVE_CASES=str(Path(case_file).resolve()),
               ACOUSTIC_WAVE_WORK_FRACTION=str(work_fraction), ACOUSTIC_TEST_WORK_FRACTION=str(work_fraction),
               ACOUSTIC_SOLID_PYTHON=str(dj.SOLID_PYTHON))
    resources = dj.Worker(command, env, lambda m: print(m, flush=True)).run()
    (logs / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    root = xml.parse(logs / 'tests.xml').getroot() if (logs / 'tests.xml').exists() else None
    print(f'{logs.name}: exit {resources["exitCode"]}, result {root.get("result") if root is not None else None}, '
          f'{resources["wallSeconds"]:.0f} s', flush=True)


def run_tests(filter_, work_fraction=.2):
    """Regression checks in the isolated runner (logs stay under .context/residual-walk)."""
    logs = ROOT / '.context/residual-walk/logs' / time.strftime('tests-%Y%m%d-%H%M%S')
    logs.mkdir(parents=True)
    dj.wait_for_mains()
    (logs / 'provenance.json').write_text(json.dumps(dict(files=dj.sync_runner(dj.RUNNER)), indent=1) + '\n')
    command = ['nice', '-n', '19', str(dj.UNITY), '-batchmode', '-force-metal', '-disable-audio', '-job-worker-count', '1',
               '-projectPath', str(dj.RUNNER), '-runTests', '-testPlatform', 'EditMode', '-testFilter', filter_,
               '-testResults', str(logs / 'tests.xml'), '-logFile', str(logs / 'unity.log')]
    env = dict(os.environ, ACOUSTIC_TEST_WORK_FRACTION=str(work_fraction), ACOUSTIC_SOLID_PYTHON=str(dj.SOLID_PYTHON))
    resources = dj.Worker(command, env, lambda m: print(m, flush=True)).run()
    (logs / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    root = xml.parse(logs / 'tests.xml').getroot() if (logs / 'tests.xml').exists() else None
    print(json.dumps(dict(log=str(logs), summary={k: root.get(k) for k in ['result', 'total', 'passed', 'failed', 'skipped']}
                          if root is not None else None)), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == '--cases':
        run_cases(sys.argv[2])
    elif sys.argv[1] == '--tests':
        run_tests(sys.argv[2])
    else:
        run(sys.argv[1])
