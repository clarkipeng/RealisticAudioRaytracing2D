"""Run one FDTD teacher-validation case and save it under the run directory.

Each case solves one fixture (the 125 Hz canopy or the 250 Hz blocks) at one
grid spacing, absorbing-layer setting and simulated duration. The output keeps
the complex receiver response, its dependence on the truncation time, receiver
impulse responses (decimated) and the listener-plane DFT map.

    nice -n 19 .context/fdtd-teacher/venv/bin/python \
      docs/benchmarks/fdtd-teacher-2026-10-28/study.py CASE \
      --evidence .context/fdtd-teacher/evidence --runs .context/fdtd-teacher/runs
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import fdtd_wave as fw  # noqa: E402

FIXTURES = dict(
    canopy=dict(scene='canopy/reference/scene.json', hz=125.0, peak=150.0, plane=(1, 1.6)),
    blocks=dict(scene='blocks/reference/scene.json', hz=250.0, peak=300.0, plane=(2, 0.2)),
)
# Spacings divide every box-face coordinate (canopy 0.05 m lattice, blocks
# 0.1 m), so walls sit exactly on velocity faces at every level.
SPACINGS = {'h50': .05, 'h25': .025, 'h17': .05 / 3}
CASES = {}


def case(name, fixture, level, duration, free=False, pml=16, margin=.5, reflection=1e-6):
    CASES[name] = dict(fixture=fixture, spacing=SPACINGS[level], duration=duration, free=free,
                       pml=pml, margin=margin, reflection=reflection)


# Canopy: the lossless semi-open room rings for seconds. 1.2 s was fixed from
# canopy-long-h50 (change vs 2 s: 0.17% whole / 0.67% shadow) before any BEM
# comparison.
case('canopy-long-h50', 'canopy', 'h50', 2.0)
for level in SPACINGS:
    case(f'canopy-free-{level}', 'canopy', level, .1, free=True)
    case(f'canopy-{level}', 'canopy', level, 1.2)
    case(f'blocks-free-{level}', 'blocks', level, .06, free=True)
    case(f'blocks-{level}', 'blocks', level, .12)
# Absorbing-layer study at the coarse level: thickness (cells) and distance (m).
for pml, margin in [(8, .5), (32, .5), (16, .25), (16, 1.0), (16, 2.0), (32, 2.0)]:
    case(f'canopy-pml{pml}-m{margin:g}-h50', 'canopy', 'h50', 1.2, pml=pml, margin=margin)
# Fine-grid check of layer thickness (distance is a physical length, covered above).
case('canopy-pml32-m0.5-h25', 'canopy', 'h25', 1.2, pml=32, margin=.5)
case('blocks-pml32-m2-h25', 'blocks', 'h25', .12, pml=32, margin=2.0)


def wait_for_mains():
    """Block while the Mac runs on battery: heavy GPU work only on AC power."""
    told = False
    while 'AC Power' not in subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout:
        if not told:
            print(f'paused on battery power at {time.strftime("%H:%M:%S")}', flush=True)
            told = True
        time.sleep(60)
    if told:
        print(f'resumed on AC power at {time.strftime("%H:%M:%S")}', flush=True)


def power_check():
    """Throttled pause hook: query power state at most every 30 s."""
    if time.time() - power_check.last > 30:
        power_check.last = time.time()
        wait_for_mains()
power_check.last = 0.0


def run(name, evidence, runs, work_fraction):
    spec = CASES[name]
    fixture = FIXTURES[spec['fixture']]
    scene_path = evidence / fixture['scene']
    scene = json.loads(scene_path.read_text())
    started = time.time()

    def progress(n, total):
        if time.time() - progress.last > 60:
            progress.last = time.time()
            print(f'{name}: {n}/{total} steps, {time.time()-started:.0f} s', flush=True)
    progress.last = 0.0
    out = fw.solve(scene, spec['spacing'], fixture['hz'], spec['duration'], fixture['peak'],
                   pml=spec['pml'], margin=spec['margin'], reflection=spec['reflection'],
                   free_field=spec['free'], plane=fixture['plane'], work_fraction=work_fraction,
                   progress=progress, pause=power_check)
    dt = out['meta']['dt']
    pressure, source = out['pressure'], out['source']
    times = np.arange(.05, spec['duration'] + 1e-9, .025)
    curve = fw.transfer_curve(pressure, source, dt, fixture['hz'], times)
    decimate = max(1, int(1 / (6000 * dt)))  # Keep >= 6 kHz; the pulse is negligible above 1 kHz.
    meta = dict(out['meta'], case=name, fixture=spec['fixture'],
                sceneSha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                solverSha256=hashlib.sha256((ROOT / 'tools/fdtd_wave.py').read_bytes()).hexdigest(),
                decimation=decimate, workFraction=work_fraction)
    runs.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(runs / f'{name}.npz', H=out['H'][0], truncationTimes=times, truncationH=curve,
                        pressure=pressure[::decimate].astype(np.float32), source=source,
                        plane=out['plane'][0], meta=json.dumps(meta))
    (runs / f'{name}.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(dict(case=name, timing=meta['timing'], shape=meta['shape']), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('case', nargs='?', choices=sorted(CASES))
    p.add_argument('--evidence', type=Path, default=ROOT / '.context/fdtd-teacher/evidence')
    p.add_argument('--runs', type=Path, default=ROOT / '.context/fdtd-teacher/runs')
    p.add_argument('--work-fraction', type=float, default=.2)
    p.add_argument('--list', action='store_true')
    a = p.parse_args()
    if a.list or not a.case:
        for name, spec in CASES.items():
            print(name, json.dumps(spec))
        return
    wait_for_mains()
    run(a.case, a.evidence, a.runs, a.work_fraction)


if __name__ == '__main__':
    main()
