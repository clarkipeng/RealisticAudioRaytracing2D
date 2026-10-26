#!/usr/bin/env python3
"""Run the offline Unity sequence at background priority with an owned-process memory guard."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path, help='Isolated Unity project copy')
    parser.add_argument('--output', required=True, type=Path, help='New capture directory')
    parser.add_argument('--unity', default='/Applications/Unity/Hub/Editor/6000.4.1f1/Unity.app/Contents/MacOS/Unity')
    parser.add_argument('--first-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, help='Exclusive frame limit; defaults to the sequence end')
    parser.add_argument('--work-fraction', type=float, default=.2)
    parser.add_argument('--max-resident-mib', type=int, default=3072)
    parser.add_argument('--first-person', action='store_true')
    parser.add_argument('--scene', help='Unity scene asset path')
    parser.add_argument('--route', choices=['walkthrough', 'weather'], default='walkthrough')
    parser.add_argument('--tail-seconds', type=float, default=11)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--take-only', action='store_true')
    parser.add_argument('--render-job', type=Path, help='Render a saved audio job with the same process guard')
    args = parser.parse_args()
    if args.max_resident_mib <= 0 or not 0 < args.work_fraction <= 1 or args.first_frame < 0 or (args.end_frame is not None and args.end_frame <= args.first_frame):
        parser.error('Invalid resource budget or frame interval.')
    output = args.output.resolve()
    if output.exists():
        parser.error('Capture output must be a new directory.')
    output.parent.mkdir(parents=True, exist_ok=True)
    nice = shutil.which('nice')
    if not nice:
        parser.error('This capture launcher requires POSIX nice and ps.')
    command = [nice, '-n', '19', args.unity, '-batchmode', '-quit', '-force-metal', '-disable-audio',
               '-job-worker-count', '1', '-projectPath', str(args.project.resolve()), '-executeMethod',
               'ChurchPresentationCapture.CaptureSequence', '-sequenceOutput', str(output),
               '-sequenceFirstFrame', str(args.first_frame), '-sequenceEndFrame', str(args.end_frame if args.end_frame is not None else 2147483647),
               '-sequenceWorkFraction', str(args.work_fraction), '-logFile', str(output.with_suffix('.log'))]
    if args.render_job:
        job = json.loads(args.render_job.read_text())
        job['workFraction'] = args.work_fraction
        job['take'] = str((args.render_job.parent / job['take']).resolve())
        job['output'] = str(output)
        guarded = output.with_suffix('.job.json')
        guarded.write_text(json.dumps(job, indent=2)+'\n')
        method = command.index('-executeMethod')
        command = command[:method] + ['-executeMethod', 'TakeBatch.Render', '-takeJob', str(guarded),
                                     '-logFile', str(output.with_suffix('.log'))]
    else:
        command += ['-sequenceRoute', args.route, '-sequenceTail', str(args.tail_seconds),
                    '-sequenceHeight', str(args.height)]
        if args.scene:
            command += ['-sequenceScene', args.scene]
        if args.take_only:
            command += ['-sequenceTakeOnly', '1']
    if args.first_person:
        command += ['-sequenceFirstPerson', '1']
    if shutil.which('taskpolicy'):
        command = [shutil.which('taskpolicy'), '-b'] + command
    samples = []
    start = time.monotonic()
    process = subprocess.Popen(command)
    print(f'Background capture PID {process.pid}; frame work fraction {args.work_fraction:.0%}', flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        while process.poll() is None:
            data = subprocess.run(['ps', '-p', str(process.pid), '-o', 'rss=,pcpu=,nice='],
                                  capture_output=True, text=True, check=False).stdout.split()
            if len(data) == 3:
                rss, cpu, priority = int(data[0]) * 1024, float(data[1]), int(data[2])
                samples.append(dict(seconds=round(time.monotonic()-start, 2), rssBytes=rss, cpuPercent=cpu, nice=priority))
                if rss > args.max_resident_mib * 1024**2:
                    raise RuntimeError('Capture exceeded its resident-memory budget; stopping this worker.')
            time.sleep(2)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report = dict(pid=process.pid, exitCode=process.returncode, wallSeconds=time.monotonic()-start,
                      residentLimitMiB=args.max_resident_mib, samples=samples)
        output.with_suffix('.resources.json').write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
