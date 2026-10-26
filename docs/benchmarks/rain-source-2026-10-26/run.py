"""Run the source-only rain comparison without playback or a Unity process.
Usage: python run.py <new-output-directory>
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, welch
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools'))
import rain_source as rain
from analyze_audio import read_wave


def metrics(x):
    x = np.asarray(x, dtype=float)
    n = 441
    frames = x[:len(x)//n*n].reshape(-1, n)
    levels = 20*np.log10(np.maximum(1e-20, np.sqrt(np.mean(frames**2, axis=1))))
    envelopes, band_power = [], []
    for low, high in [(125, 500), (500, 2000), (2000, 8000), (8000, 16000)]:
        y = sosfilt(butter(3, [low, high], btype='bandpass', fs=rain.RATE, output='sos'), x)
        envelopes.append(np.sqrt(np.mean(y[:len(x)//n*n].reshape(-1, n)**2, axis=1)))
        band_power.append(float(np.mean(y*y)))
    rms = np.sqrt(np.mean(x*x))
    f, power = welch(np.sqrt(np.mean(frames**2, axis=1)), 100, nperseg=min(256, len(frames)))
    return dict(rms=float(rms), rmsDbFS=float(20*np.log10(max(1e-20, rms))),
                peak=float(abs(x).max()), crestDb=float(20*np.log10(max(1e-20, abs(x).max()/rms))),
                excessKurtosis=float(kurtosis(x)), levelP90MinusP10Db=float(np.quantile(levels, .9)-np.quantile(levels, .1)),
                bandPowerFraction=(np.array(band_power)/sum(band_power)).tolist(),
                envelopeCorrelation=np.corrcoef(envelopes).tolist(),
                envelopeModulationFractionBelow10Hz=float(power[(f>0)&(f<10)].sum()/power[f>0].sum()))


def fade(x):
    x = x.copy()
    n = round(.02*rain.RATE)
    ramp = np.sin(np.linspace(0, np.pi/2, n))**2
    x[:n] *= ramp; x[-n:] *= ramp[::-1]
    return x


def main():
    os.nice(19)
    output = Path(sys.argv[1]); output.mkdir(parents=True, exist_ok=False)
    samples = 8*rain.RATE
    window = slice(rain.RATE, 7*rain.RATE)
    report = dict(version=rain.VERSION, scope='Dry source only; no distance, occlusion, reflection, roof radiation or SPL calibration.',
                  seconds=8, sampleRate=rain.RATE, blockSize=2048, seeds=[23, 47, 91], intensities=[.25, 1, 4],
                  platform=platform.platform(), numpyVersion=np.__version__,
                  measuredWindowSeconds=[1, 7], cases=[], files={})
    selected, banks = {}, {}
    # Warm the FFT library before timing; there is no import/scene preparation.
    warm = rain.make_bank('hard'); rain.compact(warm, rain.events(warm, 0, 0, 0, 4096), 4096)
    for surface in ['hard', 'water']:
        bank = banks[surface] = rain.make_bank(surface)
        np.savez(output/f'{surface}-bank.npz', pcm=bank.pcm, probability=bank.probability,
                 metadata=json.dumps(bank.description))
        for intensity in report['intensities']:
            for seed in report['seeds']:
                begin = time.perf_counter()
                events = rain.events(bank, seed, 0, 0, samples, intensity)
                schedule_seconds = time.perf_counter()-begin
                begin = time.perf_counter(); explicit = rain.explicit(bank, events, samples)
                explicit_seconds = time.perf_counter()-begin
                begin = time.perf_counter(); compact = rain.compact(bank, events, samples)
                compact_seconds = time.perf_counter()-begin
                difference = explicit-compact
                assert np.isfinite(compact).all() and abs(difference).max() < 1e-12
                report['cases'].append(dict(surface=surface, intensity=intensity, seed=seed, events=len(events),
                    bankHash=bank.digest, eventHash=hashlib.sha256(events.tobytes()).hexdigest(),
                    sourceHash=hashlib.sha256(compact.astype('<f8').tobytes()).hexdigest(),
                    eventGenerationSeconds=schedule_seconds, explicitSeconds=explicit_seconds, compactSeconds=compact_seconds,
                    maximumAbsoluteError=float(abs(difference).max()),
                    relativeRmsError=float(np.linalg.norm(difference)/np.linalg.norm(explicit)),
                    metrics=metrics(compact[window])))
                if intensity == 1 and seed == 23:
                    selected[surface] = compact
                    wavfile.write(output/f'{surface}-native.wav', rain.RATE, compact.astype(np.float32))
                    # Same shape family, fewer/louder events: isolate thinning,
                    # rather than confound it with the old noise kernel's timbre.
                    keep = np.random.default_rng(829).random(len(events)) < 1/64
                    thinned = events[keep].copy(); thinned['gain'] *= 8
                    selected[surface+'-thinned'] = rain.compact(bank, thinned, samples)
        other_events = rain.events(bank, 23, 1, 0, samples)
        other = rain.compact(bank, other_events, samples)
        report[surface] = dict(bankBytes=bank.pcm.nbytes+bank.probability.nbytes,
            explicitArrayWorkingBytes=rain.PulseConvolver(bank).numeric_bytes,
            memoryScope='Arrays in source convolver only; excludes shared bank, output/events, FFT scratch and all propagation/DSP.',
            regionWaveformCorrelation=float(np.corrcoef(selected[surface][window], other[window])[0, 1]),
            thinnedMetrics=metrics(selected[surface+'-thinned'][window]))

    # Existing CC0 outdoor recording is a texture reference, not a dry response
    # or a measured hard/water surface. Keep its recording acoustics explicit.
    ref_path = ROOT/'Assets/External/Church/Audio/RainCourtyard.wav'
    rate, reference = read_wave(ref_path)
    assert rate == rain.RATE and len(reference) >= samples
    reference = reference[:samples].mean(axis=1)
    report['reference'] = dict(file=str(ref_path.relative_to(ROOT)), sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
        url='https://bigsoundbank.com/rain-and-storm-s0124.html', author='Joseph SARDIN', license='CC0',
        conditions='Outdoor heavy rain; surface and microphone position unspecified. Existing mono excerpt; original acoustics retained.',
        metrics=metrics(reference[window]))
    synthetic_peak = max(float(abs(x).max()) for x in selected.values())
    gain = min(10**(-26/20)/np.sqrt(np.mean(selected['hard'][window]**2)), .8/synthetic_peak)
    reference_gain = np.sqrt(np.mean(selected['hard'][window]**2))/np.sqrt(np.mean(reference[window]**2))*gain
    report['audition'] = dict(syntheticGain=float(gain), referenceGain=float(reference_gain), fadeMilliseconds=20,
                             comparisonGapSeconds=.5, levelMatching='One synthetic gain for both surfaces and thinning control. Reference separately RMS-matched to hard candidate over 1–7 s; no limiter.')
    silent = np.zeros(round(.5*rain.RATE))
    clips = {
        'hard-rain-prototype.wav': fade(selected['hard']*gain),
        'water-rain-prototype.wav': fade(selected['water']*gain),
        'real-rain-reference.wav': fade(reference*reference_gain),
        'reference-then-hard-prototype.wav': np.concatenate([fade(reference[window]*reference_gain), silent, fade(selected['hard'][window]*gain)]),
        'dense-then-thinned-control.wav': np.concatenate([fade(selected['hard'][window]*gain), silent, fade(selected['hard-thinned'][window]*gain)]),
    }
    for name, x in clips.items():
        assert np.isfinite(x).all() and abs(x).max() <= .800001
        path = output/name; wavfile.write(path, rain.RATE, x.astype(np.float32))
        report['files'][name] = dict(seconds=len(x)/rain.RATE, peak=float(abs(x).max()),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for name, x, color in [('Hard candidate', selected['hard'][window]*gain, '#167d86'),
                            ('Thinned control', selected['hard-thinned'][window]*gain, '#ad505f'),
                            ('Real recording', reference[window]*reference_gain, '#8464a2')]:
        frames = x.reshape(-1, 441)
        db = 20*np.log10(np.maximum(1e-10, np.sqrt(np.mean(frames*frames, axis=1))))
        axes[0, 0].plot(np.arange(len(db))*.01, db, label=name, color=color, lw=.7, alpha=.8)
        axes[0, 1].hist(db, bins=np.arange(-65, -5, 2), density=True, histtype='step', label=name, color=color)
        f, power = welch(x, rain.RATE, nperseg=4096)
        axes[1, 0].semilogx(f[1:], 10*np.log10(np.maximum(power[1:], 1e-20)), label=name, color=color)
    axes[0, 0].set(title='10 ms envelope: unchanged gain for thinning control', xlabel='Seconds in audition crop', ylabel='RMS dBFS', ylim=(-65, -5))
    axes[0, 1].set(title='Distribution of short-window levels', xlabel='RMS dBFS', ylabel='Density')
    axes[1, 0].set(title='Spectrum (reference RMS-matched for audition only)', xlabel='Hz', ylabel='dBFS/Hz', xlim=(100, 20000), ylim=(-110, -40))
    for surface, color in [('hard', '#167d86'), ('water', '#8464a2')]:
        values = []
        for intensity in report['intensities']:
            cases = [c for c in report['cases'] if c['surface']==surface and c['intensity']==intensity]
            values.append(np.median([c['explicitSeconds']/c['compactSeconds'] for c in cases]))
        axes[1, 1].plot(np.array(report['intensities'])*4000, values, 'o-', label=surface, color=color)
    axes[1, 1].axhline(1, color='#888', ls=':')
    axes[1, 1].set(title='Source mixing speedup: explicit / compact', xlabel='Expected impacts/s per test region', ylabel='Ratio (above 1 = compact faster)', xscale='log')
    for ax in axes.flat: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(output/'source-review.png', dpi=150); plt.close(fig)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for surface in ['hard', 'water']:
        rows = [c for c in report['cases'] if c['surface']==surface and c['intensity']==1]
        print(surface, 'kurtosis:', [round(c['metrics']['excessKurtosis'], 2) for c in rows],
              'bank bytes:', report[surface]['bankBytes'], 'source array working bytes:', report[surface]['explicitArrayWorkingBytes'])
    print('Maximum error:', max(c['maximumAbsoluteError'] for c in report['cases']))
    print('Audition:', report['audition'])


if __name__ == '__main__':
    main()
