"""Doppler on reflections, the train station demo (docs/benchmarks/path-denoise-2026-11-12, part 40).

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY tools/doppler_station.py render TAKE   # guarded Unity renders of a station take, with and without
                                                         # Doppler on the reflections -> .context/doppler/station-*/
    $PY tools/doppler_station.py clips                   # listening clips -> .context/energy-audio/listen/

TAKE is a recorded TrainStation take with the horn on the moving train (e.g. a live smoke run's take). Both renders
use the path method as the live showcase runs it (persistent rays under population control, binaural, GPU audio) and
differ only in settings.pathReflectionDoppler: on, every early echo slides at its own delay rate; off, only the
direct sound is Doppler-shifted and the echoes step from update to update.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import method_runs as mr  # noqa: E402

WORK = ROOT / '.context/doppler'
LISTEN = ROOT / '.context/energy-audio/listen'
SETTINGS = dict(responseStepSamples=1024, speedOfSound=343.0, referenceDistance=1.0, minimumDistance=.1, tailSamples=2 * 44100,
                transitionSamples=512, delayFilter=1, listenerOutput=1, sincRadius=16, maximumDelaySeconds=2.0, maximumSignalEvents=4096,
                maximumPendingObservations=16, maximumResponsesPerBatch=4, useBvh=True, transport=1, synthesis=1, audioBackend=1,
                seed=0, diffractionFactor=1.0, airAbsorption=True, fftSize=1024, spectralHop=256, maxBounces=64, method=3,
                methodRays=512, methodRouletteFloor=1e-6, methodEdgeProbability=.5, specularDetectionRadius=2.0,
                pathPersistent=True, pathPopulation=11000, pathIterationCap=16, pathEdgeSliceUpdates=4)
VARIANTS = {'station-doppler': dict(SETTINGS, pathReflectionDoppler=True),
            'station-direct-only': dict(SETTINGS, pathReflectionDoppler=False)}


def render(take):
    mr.LOGS = WORK / 'logs'; mr.LOGS.mkdir(parents=True, exist_ok=True)
    for name, settings in VARIANTS.items():
        output = WORK / name
        shutil.rmtree(output, ignore_errors=True)
        job = dict(take=str(Path(take).resolve()), output=str(output), shader='Assets/Shaders/Transport.compute', settings=settings, workFraction=.5)
        path = WORK / f'{name}.job.json'
        path.write_text(json.dumps(job, indent=1) + '\n')
        logs, resources = mr.launch(name, ['-executeMethod', 'TakeBatch.Render', '-takeJob', str(path)])
        if resources['exitCode'] != 0:
            raise SystemExit(f'{name} failed; see {logs}/unity.log')


def read(path):
    from scipy.io import wavfile
    rate, x = wavfile.read(str(path))
    x = x.astype(np.float64) / (1 if x.dtype.kind == 'f' else 2 ** (8 * x.dtype.itemsize - 1))
    return rate, x.reshape(len(x), -1)


def write(path, rate, x):
    from scipy.io import wavfile
    wavfile.write(str(path), rate, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def clips(start, seconds):
    """The passage of the train, both renders at the same gain, and the two back to back."""
    pieces = []
    for name in VARIANTS:
        wav = next((WORK / name).glob('listener-0*.wav'))
        rate, x = read(wav)
        pieces.append((name, rate, x[int(start * rate):int((start + seconds) * rate)]))
    peak = max(np.abs(p[2]).max() for p in pieces) or 1
    LISTEN.mkdir(parents=True, exist_ok=True)
    for name, rate, x in pieces:
        write(LISTEN / f'{name}.wav', rate, .9 * x / peak)
    rate = pieces[0][1]; gap = np.zeros((rate // 2, pieces[0][2].shape[1]))
    write(LISTEN / 'station-doppler-ab.wav', rate, .9 * np.concatenate([pieces[1][2], gap, pieces[0][2]]) / peak)
    print('\n'.join(str(LISTEN / f) for f in ['station-direct-only.wav', 'station-doppler.wav', 'station-doppler-ab.wav']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['render', 'clips'])
    p.add_argument('take', nargs='?')
    p.add_argument('--start', type=float, default=10.0, help='clips: seconds into the render')
    p.add_argument('--seconds', type=float, default=12.0)
    a = p.parse_args()
    render(a.take) if a.command == 'render' else clips(a.start, a.seconds)
