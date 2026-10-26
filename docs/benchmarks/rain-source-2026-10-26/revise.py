"""Compare the rejected v1 timbre with an authored splash revision, no playback.

Usage: python revise.py <new-output-directory>
The original run.py and its report remain reproducible and unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

from run import ROOT, fade, metrics
import rain_source as rain
from analyze_audio import read_wave


def main():
    os.nice(19)
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    old_bank, bank = rain.make_bank('hard'), rain.make_splash_bank()
    samples, window = 8*rain.RATE, slice(rain.RATE, 7*rain.RATE)
    report = dict(version=bank.description['version'], sampleRate=rain.RATE,
                  feedback='User rejected v1 as still too clicky or artificial.',
                  scope='Dry source listening iteration; not a measured material model or spatial renderer.',
                  bankHash=bank.digest, previousBankHash=old_bank.digest,
                  cases=[], files={}, measuredWindowSeconds=[1, 7])
    selected = {}
    for intensity in [.25, 1, 4]:
        for seed in [23, 47, 91]:
            events = rain.events(bank, seed, 0, 0, samples, intensity)
            # Same times, voices and gains isolate response changes.
            assert np.array_equal(events, rain.events(old_bank, seed, 0, 0, samples, intensity))
            actual = rain.compact(bank, events, samples)
            reference = rain.explicit(bank, events, samples)
            error = float(np.max(np.abs(actual-reference)))
            assert np.isfinite(actual).all() and error < 1e-12
            report['cases'].append(dict(seed=seed, intensity=intensity, events=len(events),
                eventHash=hashlib.sha256(events.tobytes()).hexdigest(),
                maximumAbsoluteError=error, metrics=metrics(actual[window])))
            if intensity == 1:
                selected[seed] = actual
                if seed == 23:
                    old = rain.compact(old_bank, events, samples)
    rate, recording = read_wave(ROOT/'Assets/External/Church/Audio/RainCourtyard.wav')
    assert rate == rain.RATE
    recording = recording[:samples].mean(axis=1)
    previous_report = json.loads((Path(__file__).parent/'report.json').read_text())
    old_gain = previous_report['audition']['syntheticGain']
    rms = lambda x: np.sqrt(np.mean(x[window]**2))
    target = rms(old)*old_gain
    gain, reference_gain = target/rms(selected[23]), target/rms(recording)
    report['audition'] = dict(previousGain=old_gain, revisedGain=float(gain),
        referenceGain=float(reference_gain), gapSeconds=.5,
        levelMatching='Fixed gain per profile, matched to v1 RMS over 1–7 s at seed 23/intensity 1. Same v2 gain for seed 47; no limiter or per-drop normalization.')
    gap = np.zeros(round(.5*rain.RATE))
    clips = {
        'previous-then-splash-revision.wav': np.concatenate([
            fade(old[window]*old_gain), gap, fade(selected[23][window]*gain)]),
        'recorded-then-splash-revision.wav': np.concatenate([
            fade(recording[window]*reference_gain), gap, fade(selected[23][window]*gain)]),
        'splash-revision-seed23.wav': fade(selected[23]*gain),
        'splash-revision-seed47.wav': fade(selected[47]*gain),
    }
    for name, x in clips.items():
        assert np.isfinite(x).all() and np.max(np.abs(x)) < .95
        path = output/name
        wavfile.write(path, rain.RATE, x.astype(np.float32))
        report['files'][name] = dict(seconds=len(x)/rate, peak=float(np.max(np.abs(x))),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    np.savez(output/'splash-bank.npz', pcm=bank.pcm, probability=bank.probability,
             metadata=json.dumps(bank.description))
    wavfile.write(output/'splash-native.wav', rate, selected[23].astype(np.float32))
    report['reference'] = previous_report['reference']
    report['previousMetrics'] = metrics(old[window])
    report['bankBytes'] = bank.pcm.nbytes+bank.probability.nbytes
    report['convolverArrayBytes'] = rain.PulseConvolver(bank).numeric_bytes
    report['memoryScope'] = 'Source arrays only; excludes FFT scratch, bank, event list, output and all propagation.'
    (output/'revision-report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    selected_case = next(c for c in report['cases'] if c['intensity'] == 1 and c['seed'] == 23)
    print(json.dumps(dict(maximumError=max(c['maximumAbsoluteError'] for c in report['cases']),
                         previous=report['previousMetrics'], revised=selected_case['metrics'],
                         audition=report['audition'], files=report['files']), indent=2))


if __name__ == '__main__':
    main()
