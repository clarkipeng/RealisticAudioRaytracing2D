"""One entry point for Unity EditMode checks, in tiers.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY tools/unity_tests.py quick              # every change: all TakeChecks except category Slow
    $PY tools/unity_tests.py full               # before a commit: everything
    $PY tools/unity_tests.py run "TakeChecks.Late"   # one area (NUnit -testFilter)

Runs in the isolated runner (.context/gpu-audio/runner) under tools/heavy_guard: one heavy job
machine-wide, non-critical pressure and 25% free memory to start, 3 GiB footprint limit, nice 19, one job worker,
50% pacing. Opt-in measurements skip themselves unless their environment variables are set.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as xml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_jobs as dj  # noqa: E402
import heavy_guard as hg  # noqa: E402

LOGS = ROOT / '.context/unity-tests'
TIERS = {'quick': ('TakeChecks', '!Slow'), 'full': ('TakeChecks', None)}


def clear_stale_lock():
    lock = dj.RUNNER / 'Temp/UnityLockfile'
    if lock.exists() and str(dj.RUNNER) not in subprocess.run(['ps', '-A', '-o', 'command='],
                                                              capture_output=True, text=True).stdout:
        lock.unlink()


def prune_runner():
    """Remove runner copies of sources deleted from the repository (sync only copies)."""
    for folder in [f for f in dj.SYNCED if f.startswith('Assets/')]:
        for target in list((dj.RUNNER / folder).rglob('*')):
            source = ROOT / target.relative_to(dj.RUNNER)
            if target.is_file() and not source.exists() and not source.with_suffix('').exists():
                target.unlink()


def run(test_filter, category=None, work_fraction=hg.WORK_FRACTION, env=None):
    logs = LOGS / time.strftime('%Y%m%d-%H%M%S')
    logs.mkdir(parents=True)
    dj.wait_for_mains()
    clear_stale_lock()
    prune_runner()
    dj.sync_runner(dj.RUNNER)
    results = logs / 'tests.xml'
    command = ['nice', '-n', '19', str(dj.UNITY), '-batchmode', '-force-metal', '-disable-audio',
               '-job-worker-count', '1', '-projectPath', str(dj.RUNNER), '-runTests', '-testPlatform', 'EditMode',
               '-testFilter', test_filter, '-testResults', str(results), '-logFile', str(logs / 'unity.log')]
    if category:
        command += ['-testCategory', category]
    environment = dict(os.environ, ACOUSTIC_TEST_WORK_FRACTION=str(work_fraction),
                       ACOUSTIC_SOLID_PYTHON=str(dj.SOLID_PYTHON), **(env or {}))
    print(f'{logs.name}: filter {test_filter}' + (f', category {category}' if category else ''), flush=True)
    resources = dj.Worker(command, environment, lambda m: print(m, flush=True)).run()
    (logs / 'resources.json').write_text(json.dumps(resources, indent=1) + '\n')
    print(f'exit {resources["exitCode"]}, {resources["wallSeconds"]:.0f} s, '
          f'peak footprint {resources["peakTreeFootprintBytes"] / hg.GIB:.2f} GiB', flush=True)
    if not results.exists():
        print(f'No test results; see {logs / "unity.log"}', flush=True)
        return 1
    root = xml.parse(results).getroot()
    print('result {result}: {passed} passed, {failed} failed, {skipped} skipped of {total}'.format(
        **{k: root.get(k) for k in ['result', 'passed', 'failed', 'skipped', 'total']}), flush=True)
    cases = list(root.iter('test-case'))
    for case in cases:
        if case.get('result') == 'Failed':
            message = case.find('failure/message')
            print(f"FAILED {case.get('fullname')}: {(message.text or '').strip()[:600] if message is not None else ''}")
    slowest = sorted(cases, key=lambda c: -float(c.get('duration') or 0))[:5]
    print('slowest: ' + ', '.join(f"{c.get('name')} {float(c.get('duration')):.0f} s" for c in slowest), flush=True)
    return 0 if root.get('failed') == '0' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tier', choices=['quick', 'full', 'run'])
    parser.add_argument('filter', nargs='?', help='NUnit -testFilter for "run"')
    parser.add_argument('--category', help='extra -testCategory for "run"')
    parser.add_argument('--work-fraction', type=float, default=hg.WORK_FRACTION)
    args = parser.parse_args()
    if args.tier == 'run':
        if not args.filter:
            parser.error('"run" needs a test filter')
        test_filter, category = args.filter, args.category
    else:
        test_filter, category = TIERS[args.tier]
    sys.exit(run(test_filter, category, args.work_fraction))


if __name__ == '__main__':
    main()
