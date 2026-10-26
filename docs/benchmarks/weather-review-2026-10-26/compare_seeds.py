"""Measure two transport seeds with identical indoor rain inputs.
Usage: python compare_seeds.py <weather-review-directory> <output-json>
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

root, output = map(Path, sys.argv[1:])
runs, signals, rows = [], [], []
for name in ['rain-inside', 'rain-inside-seed1']:
    directory = root / (name + '-audio')
    run = json.loads((directory / 'run.json').read_text())
    assert run['complete']
    path = directory / 'listener-0.wav'
    rate, raw = wavfile.read(path)
    assert rate == 44100 and raw.ndim == 1 and np.isfinite(raw).all()
    pcm = raw[run['dspLatencySamples']:].astype(float)
    window = pcm[2 * rate:round(5.5 * rate)]
    peak = int(np.argmax(abs(pcm)))
    rows.append(dict(seed=run['settings']['seed'], sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     peak=float(abs(pcm[peak])), peakSeconds=peak/rate,
                     overFullScale=int(np.sum(abs(raw) > 1)),
                     windowRmsDbFS=float(20*np.log10(np.sqrt(np.mean(window**2)))),
                     workingNumericBytes=run['workingNumericBytes'], renderWallSeconds=run['wallSeconds']))
    runs.append(run); signals.append(pcm)

identity = ['sceneHash', 'takeStateHash', 'takeEventHash', 'takeSourceHash', 'sampleRate', 'dspLatencySamples',
            'outputSamples', 'provenance', 'audioShaderDependencyHash']
for key in identity:
    assert runs[0][key] == runs[1][key], key
changed = [key for key in runs[0]['settings'] if runs[0]['settings'][key] != runs[1]['settings'][key]]
assert changed == ['seed'], changed
assert len(signals[0]) == len(signals[1])

peaks = []
for i, pcm in enumerate(signals):
    at = int(np.argmax(abs(pcm))); radius = round(.01 * rate)
    other = signals[1-i][max(0, at-radius):at+radius+1]
    peaks.append(dict(seed=rows[i]['seed'], peakSeconds=at/rate,
                      otherSeedPeakWithin10ms=float(np.max(abs(other)))))
report = dict(verifiedSame=identity, changedSettings=changed, windowSeconds=[2, 5.5],
              inputTake='rain-inside-take', renders=rows, largestPeakCrossChecks=peaks,
              fullCorrelation=float(np.corrcoef(*signals)[0, 1]),
              fullDifferenceRms=float(np.sqrt(np.mean((signals[0]-signals[1])**2))),
              interpretation='Two-seed sensitivity diagnostic, not a convergence study or a physical reference.')
output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
