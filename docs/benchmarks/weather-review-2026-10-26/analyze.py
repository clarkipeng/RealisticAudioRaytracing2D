"""Inspect matched stationary weather probes without audio playback.

Usage: python analyze.py <weather-review-directory> <new-output-json>
NumPy/SciPy required. The 2–5.5 s window excludes wind onset and source stop.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

root, output = map(Path, sys.argv[1:])
report = {'windowSeconds': [2, 5.5], 'levels': 'RMS dBFS, not calibrated SPL', 'renders': {}}
for name in ['wind-baseline', 'wind-dense', 'rain-outside', 'rain-inside']:
    directory = root / (name + '-audio')
    run = json.loads((directory / 'run.json').read_text())
    take = json.loads((root / 'probes' / (name + '-take') / 'take.json').read_text())
    assert run['complete']
    for a, b in [('sceneHash', 'sceneHash'), ('takeStateHash', 'stateHash'), ('takeEventHash', 'eventHash')]:
        assert run[a] == take[b]
    listeners = []
    for path in sorted(directory.glob('listener-*.wav')):
        rate, raw = wavfile.read(path)
        assert raw.ndim == 1 and np.isfinite(raw).all()
        pcm = raw[run['dspLatencySamples']:].astype(np.float64)
        x = pcm[2 * rate:round(5.5 * rate)]
        blocks = x.reshape(-1, rate // 4)
        rms = np.sqrt(np.mean(blocks ** 2, axis=1))
        levels = 20 * np.log10(np.maximum(1e-20, rms))
        f, psd = welch(x, rate, nperseg=4096)
        def band(lo, hi):
            selected = (f >= lo) & (f < hi)
            return float(np.trapezoid(psd[selected], f[selected]))
        listeners.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              peak=float(np.max(np.abs(raw))), overFullScale=int(np.sum(np.abs(raw) > 1)),
                              rms=float(np.sqrt(np.mean(x ** 2))), dBFS=float(20*np.log10(np.sqrt(np.mean(x ** 2)))),
                              quarterSecondDb=levels.tolist(), quarterSecondRangeDb=float(np.ptp(levels)),
                              quarterSecondP90MinusP10Db=float(np.quantile(levels, .9)-np.quantile(levels, .1)),
                              lowBandPower80To1000=band(80, 1000), highBandPower1000To10000=band(1000, 10000)))
    resources = json.loads((root / (name + '-audio.resources.json')).read_text())
    report['renders'][name] = dict(sceneHash=take['sceneHash'], stateHash=take['stateHash'], eventHash=take['eventHash'],
                                   geometryHash=take['scene']['geometryHash'], workingNumericBytes=run['workingNumericBytes'],
                                   renderWallSeconds=run['wallSeconds'], launcherWallSeconds=resources['wallSeconds'],
                                   peakResidentMiB=max(s['rssBytes'] for s in resources['samples'])/2**20,
                                   minimumNiceAfterLaunch=min(s['nice'] for s in resources['samples'] if s['seconds']>=2),
                                   listeners=listeners)
output.write_text(json.dumps(report, indent=2)+'\n')
for name, render in report['renders'].items():
    print(name, [(round(x['dBFS'], 2), round(x['quarterSecondP90MinusP10Db'], 2), round(x['peak'], 4)) for x in render['listeners']])
