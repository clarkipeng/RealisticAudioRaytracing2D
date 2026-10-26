"""Small direct-sound grouping/doorway comparison. No Unity or playback.

Usage: python spatial.py <new-output-directory>
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

from run import fade, metrics
import rain_source as rain
import rain_spatial as spatial


def energy(events, path, bank):
    kernels = np.sum(bank.pcm**2, axis=1)
    return events['gain']**2*path['gain']**2*kernels[events['voice']]


def grouping_error(events, original, grouped, receiver, exact, approx, bank):
    e = energy(events, exact, bank)
    both = exact['visible'] & approx['visible']
    u, v = original-receiver, grouped-receiver
    angle = np.degrees(np.arccos(np.clip(np.sum(u*v, axis=1)/
                                         (exact['distance']*approx['distance']), -1., 1.)))
    weights = e*both
    delay_ms = (approx['delay']-exact['delay'])*1000/rain.RATE
    return dict(visibilityMismatchFraction=float(np.mean(exact['visible'] != approx['visible'])),
        missedReferenceEnergyFraction=float(e[~approx['visible']].sum()/e.sum()),
        isolatedEventEnergyDifferenceDb=float(10*np.log10(energy(events, approx, bank).sum()/e.sum())),
        visibleWeightedDelayRmsMs=float(np.sqrt(np.sum(weights*delay_ms**2)/weights.sum())),
        visibleWeightedDirectionRmsDegrees=float(np.sqrt(np.sum(weights*angle**2)/weights.sum())))


def write_page(output, config, report):
    events = spatial.field(rain.make_splash_bank(), 23, 0, 8*rain.RATE, config)
    points = events['position'][events['surface'] != 0][::80]
    rays = {}
    for name in ['outside', 'inside']:
        mask = spatial.transfer(points, config['receivers'][name], config)['visible']
        rays[name] = points[mask][::max(1, np.count_nonzero(mask)//5)][:5].tolist()
    levels = {c['receiver']: c['metrics']['rmsDbFS'] for c in report['cases']
              if c['seed'] == 23 and c['patchWidthMetres'] == 0}
    data = dict(scene=config, points=points.tolist(), rays=rays,
                insideDifferenceDb=levels['inside']-levels['outside'])
    template = Path(__file__).with_name('shelter.html').read_text()
    (output/'index.html').write_text(template.replace('{{DATA}}', json.dumps(data, allow_nan=False)))


def main():
    os.nice(19)
    output = Path(sys.argv[1]); output.mkdir(parents=True, exist_ok=False)
    config, bank = spatial.scene(), rain.make_splash_bank()
    samples, window = 8*rain.RATE, slice(rain.RATE, 7*rain.RATE)
    report = dict(version=config['version'], seconds=8, sampleRate=rain.RATE,
        scope='Direct sound in an ideal opaque shelter; not a Church render, moving walk or complete room model.',
        sourceFeedback='User said the splash revision is better; selected as the current comparison candidate, not fully realistic.',
        bankHash=bank.digest, seeds=[23, 47, 91], patchWidthsMetres=[2., 1., .5, .25],
        grouping='Same events and first-hit surfaces; replace positions with surface-clipped patch centers only.',
        delayQuantization='Nearest sample; maximum travel-delay rounding error 0.5/44100 s.',
        cases=[], fields=[], files={})
    selected = {}
    for seed in report['seeds']:
        events = spatial.field(bank, seed, 0, samples, config)
        report['fields'].append(dict(seed=seed, eventCount=len(events),
            sha256=hashlib.sha256(events.tobytes()).hexdigest(),
            roofImpacts=int(np.count_nonzero(events['surface']==0))))
        for name, receiver in config['receivers'].items():
            exact_audio, exact = spatial.render(bank, events, events['position'], receiver, config, samples)
            if name == 'inside':
                assert not np.any(exact['visible'][events['surface']==0])
            report['cases'].append(dict(seed=seed, receiver=name, patchWidthMetres=0,
                visibleEvents=int(np.count_nonzero(exact['visible'])), metrics=metrics(exact_audio[window])))
            if seed == 23 and name in ('outside', 'inside'):
                selected[name] = exact_audio
                wavfile.write(output/f'{name}-native.wav', rain.RATE, exact_audio.astype(np.float32))
            for width in report['patchWidthsMetres']:
                positions = spatial.patch_positions(events, config, width)
                audio, path = spatial.render(bank, events, positions, receiver, config, samples)
                assert np.isfinite(audio).all()
                report['cases'].append(dict(seed=seed, receiver=name, patchWidthMetres=width,
                    visibleEvents=int(np.count_nonzero(path['visible'])), metrics=metrics(audio[window]),
                    error=grouping_error(events, events['position'], positions, np.asarray(receiver), exact, path, bank)))
                if seed == 23 and name in ('outside', 'inside') and width in (2., .25):
                    selected[f'{name}-{width:g}m'] = audio
        print('Completed seed', seed, flush=True)
    gain = min(10**(-26/20)/np.sqrt(np.mean(selected['outside'][window]**2)),
               .8/max(np.max(np.abs(x)) for x in selected.values()))
    gap = np.zeros(round(.5*rain.RATE))
    def pair(a, b):
        return np.concatenate([fade(selected[a][window]*gain), gap, fade(selected[b][window]*gain)])
    clips = {'outside-then-inside.wav': pair('outside', 'inside'),
             'outside-individual-then-2m-patches.wav': pair('outside', 'outside-2m'),
             'inside-individual-then-2m-patches.wav': pair('inside', 'inside-2m'),
             'inside-individual-then-025m-patches.wav': pair('inside', 'inside-0.25m'),
             'outside.wav': fade(selected['outside']*gain),
             'inside.wav': fade(selected['inside']*gain)}
    for name, audio in clips.items():
        assert np.isfinite(audio).all() and np.max(abs(audio)) < .81
        path = output/name
        wavfile.write(path, rain.RATE, audio.astype(np.float32))
        report['files'][name] = dict(seconds=len(audio)/rain.RATE, peak=float(np.max(abs(audio))),
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report['audition'] = dict(gain=float(gain), cropSeconds=[1, 7], gapSeconds=.5, fadeMilliseconds=20,
                             levelMatching='One common gain for all receivers and patch sizes; no inside boost, per-clip normalization or limiter.')
    report['memoryScope'] = 'Offline source/geometry arrays only; not a Unity memory or realtime benchmark.'
    for name, value in [('scene.json', config), ('spatial-report.json', report)]:
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    write_page(output, config, report)
    for name in config['receivers']:
        exact = [c['metrics']['rmsDbFS'] for c in report['cases'] if c['receiver']==name and not c['patchWidthMetres']]
        print(name, 'reference RMS dBFS', exact)
        for width in [2., .25]:
            cases = [c for c in report['cases'] if c['receiver']==name and c['patchWidthMetres']==width]
            print(width, 'energy error dB', [round(c['error']['isolatedEventEnergyDifferenceDb'], 3) for c in cases],
                  'visibility mismatch', [round(c['error']['visibilityMismatchFraction'], 4) for c in cases])


if __name__ == '__main__':
    main()
