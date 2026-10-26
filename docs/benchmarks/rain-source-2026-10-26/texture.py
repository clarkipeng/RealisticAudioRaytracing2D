"""Recorded surface-texture audition; no individual-drop or shelter simulation.

Usage: python texture.py <dense|concrete> <input.wav> <new-output-directory>
Inputs are the documented CC0 recordings; PCM hashes prevent silent substitution.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import wavfile

from run import fade, metrics
import rain_source as rain
import rain_texture


def main():
    os.nice(19)
    profile = sys.argv[1]
    profiles = dict(
        dense=dict(title='Rain and Thunder #1 (0124)',
            page='https://bigsoundbank.com/rain-and-storm-s0124.html',
            pcmHash='899b22a1e3db85d4a6a4e2584c66dde69a2db1faca22ae8eb4b37489a5f6afe9',
            usedFormat='Original 44.1 kHz 16-bit stereo WAV, averaged to mono. Only seconds 1–13 are used, before the later thunder.',
            description='Outdoor heavy rain; surface and microphone distance unspecified. This is an ambience, not a dry impact.',
            starts=np.arange(1, 9, .5), heldOut=[9, 13]),
        concrete=dict(title='Rain on concrete (1289)',
            page='https://bigsoundbank.com/rain-on-concrete-s1289.html',
            pcmHash='fb9d0fc67c04872b4df618bf35aa99f2c097089c147a54e7f6aee953a581ac82',
            usedFormat='MP3 preview decoded/resampled to mono 44.1 kHz float32; not the original 24-bit WAV.',
            description='Big drops on a small concrete slab, outdoors; microphone distance and calibration unspecified. Sparse, peaky recording.',
            starts=np.arange(10, 74), heldOut=[1, 9]))
    preset = profiles[profile]
    source = Path(sys.argv[2])
    rate, recording = wavfile.read(source)
    if np.issubdtype(recording.dtype, np.integer):
        recording = recording.astype(float)/(np.iinfo(recording.dtype).max+1)
    if recording.ndim == 2:
        recording = recording.mean(axis=1)
    recording = recording.astype('<f4')
    digest = hashlib.sha256(recording.tobytes()).hexdigest()
    if rate != rain.RATE or recording.ndim != 1 or digest != preset['pcmHash']:
        raise ValueError('Input PCM does not match the documented recording; preserve provenance.')
    output = Path(sys.argv[3]); output.mkdir(parents=True, exist_ok=False)
    samples, window = 8*rate, slice(rate, 7*rate)
    starts = (preset['starts']*rate).astype(int)
    bank = np.stack([recording[start:start+rate//2] for start in starts])
    bank -= bank.mean(axis=1, keepdims=True)
    reference = recording[preset['heldOut'][0]*rate:preset['heldOut'][1]*rate].astype(float)
    report = dict(version=rain_texture.VERSION, profile=profile,
        feedback='User clarified that the rain itself still sounds weird; splash source and shelter audition are unaccepted.',
        scope='Empirical dense texture audition, not individual-drop synthesis or a calibrated dry source. Recording ambience remains.',
        source=dict(title=preset['title'], author='Joseph SARDIN', license='CC0',
            page=preset['page'], inputFileSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            decodedPcmSha256=digest, usedFormat=preset['usedFormat'], description=preset['description']),
        bank=dict(responses=len(bank), samplesPerResponse=bank.shape[1], bytes=bank.nbytes,
                  startSeconds=(starts/rate).tolist(), durationSeconds=.5, preprocessing='Subtract each passage mean; no per-grain level equalization, denoising or EQ.',
                  sha256=hashlib.sha256(bank.astype('<f4').tobytes()).hexdigest()),
        heldOutReferenceSeconds=preset['heldOut'], sampleRate=rate, durationSeconds=8,
        cases=[], files={})
    selected = {}
    for seed in [23, 47, 91]:
        before = time.perf_counter()
        audio = rain_texture.render(bank, seed, 0, 0, samples)
        elapsed = time.perf_counter()-before
        cuts = [0, 1, 11024, 11025, 23001, 220501, samples]
        partitioned = np.concatenate([rain_texture.render(bank, seed, 0, a, b)
                                      for a, b in zip(cuts, cuts[1:])])
        assert np.array_equal(partitioned, audio) and np.isfinite(audio).all()
        other = rain_texture.render(bank, seed, 1, 0, samples)
        report['cases'].append(dict(seed=seed, renderSeconds=elapsed, partitionBitExact=True,
            metrics=metrics(audio[window]), regionCorrelation=float(np.corrcoef(audio[window], other[window])[0, 1])))
        selected[seed] = audio
    old_bank = rain.make_splash_bank()
    old = rain.compact(old_bank, rain.events(old_bank, 23, 0, 0, samples), samples)
    rms = lambda x: np.sqrt(np.mean(x**2))
    target = 10**(-28/20)
    new_gain, old_gain, ref_gain = target/rms(selected[23][window]), target/rms(old[window]), target/rms(reference)
    maximum = max(max(np.max(abs(x))*new_gain for x in selected.values()),
                  np.max(abs(old))*old_gain, np.max(abs(reference))*ref_gain)
    headroom = min(1., .8/maximum)
    new_gain *= headroom; old_gain *= headroom; ref_gain *= headroom
    report['audition'] = dict(textureGain=float(new_gain), oldGain=float(old_gain), referenceGain=float(ref_gain),
        levelMatching='One fixed gain per model; seed 23 RMS matched over seconds 1–7; reference matched over its whole held-out excerpt. Same texture gain for all seeds. No limiter.',
        gapSeconds=.5, fadeMilliseconds=20)
    gap = np.zeros(round(.5*rate))
    clips = {
        'authored-then-recorded-texture.wav': np.concatenate([fade(old[window]*old_gain), gap, fade(selected[23][window]*new_gain)]),
        'reference-then-recorded-texture.wav': np.concatenate([fade(reference*ref_gain), gap, fade(selected[23][:len(reference)]*new_gain)]),
        'recorded-texture-seed23.wav': fade(selected[23]*new_gain),
        'recorded-texture-seed47.wav': fade(selected[47]*new_gain),
        'held-out-reference.wav': fade(reference*ref_gain),
    }
    for name, audio in clips.items():
        assert np.isfinite(audio).all() and np.max(abs(audio)) < .800001
        path = output/name; wavfile.write(path, rate, audio.astype(np.float32))
        report['files'][name] = dict(seconds=len(audio)/rate, peak=float(np.max(abs(audio))),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    np.savez(output/'texture-bank.npz', pcm=bank, sampleRate=rate, startSeconds=starts/rate)
    wavfile.write(output/'texture-native.wav', rate, selected[23].astype(np.float32))
    report['referenceMetrics'] = metrics(reference)
    report['previousSourceMetrics'] = metrics(old[window])
    (output/'texture-report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if profile == 'dense':
        (output/'index.html').write_text(Path(__file__).with_name('texture.html').read_text())
    print(json.dumps(dict(bankBytes=bank.nbytes, cases=[dict(seed=c['seed'], seconds=c['renderSeconds'],
        regionCorrelation=c['regionCorrelation'], kurtosis=c['metrics']['excessKurtosis']) for c in report['cases']],
        audition=report['audition'], files=report['files']), indent=2))


if __name__ == '__main__':
    main()
