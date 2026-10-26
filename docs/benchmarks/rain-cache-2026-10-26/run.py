"""Try baked source/listener interpolation in one ideal static shoebox.

python run.py <input-mono-float.wav> <new-output-directory>
No Unity, playback or production renderer changes. Exact mirror geometry is the
cheap reference, not a performance stand-in for tracing arbitrary scenes.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.fft import rfft
from scipy.io import wavfile
from scipy.signal import fftconvolve


RATE, SPEED = 44100, 343.
ROOM = np.array([8., 6., 3.])
ORDER, REFLECTION = 32, .65
SOURCE_BOUNDS = [[1., 2.], [2., 3.]]
RECEIVER_BOUNDS = [[4., 5.], [2.5, 3.5]]


def image_indices(order):
    return np.array([(x, y, z) for x in range(-order, order+1)
                     for y in range(-order+abs(x), order-abs(x)+1)
                     for z in range(-order+abs(x)+abs(y), order-abs(x)-abs(y)+1)
                     if x or y or z], dtype=np.int32)


def paths(source, receiver, indices):
    images = indices*ROOM+np.where(indices % 2 == 0, source, ROOM-source)
    distance = np.linalg.norm(images-receiver, axis=1)
    return distance/SPEED, REFLECTION**np.sum(abs(indices), axis=1)/distance


def impulses(delay, gain):
    """17-tap windowed-sinc delay deposition, same in every comparison."""
    out = np.zeros(RATE+1)
    position = delay*RATE
    base = np.floor(position).astype(int)
    offsets = np.arange(-8, 9)
    weights = np.sinc(offsets[None, :]-(position-base)[:, None])*np.kaiser(17, 7.5)
    weights /= weights.sum(axis=1, keepdims=True)
    at = base[:, None]+offsets
    if np.any(at < 0) or np.any(at >= len(out)):
        raise ValueError('Path outside fixture impulse-response support.')
    np.add.at(out, at.ravel(), (gain[:, None]*weights).ravel())
    return out


def direct(source, receiver):
    distance = np.linalg.norm(source-receiver)
    return impulses(np.array([distance/SPEED]), np.array([1/distance]))


def grid(bounds, step, height):
    axes = [np.linspace(a, b, round((b-a)/step)+1) for a, b in bounds]
    return axes, np.array([[x, y, height] for x in axes[0] for y in axes[1]])


def neighbors(axes, point):
    indices, weights = [], []
    for axis in range(2):
        values = axes[axis]
        if not values[0] <= point[axis] <= values[-1]:
            raise ValueError('Query outside baked domain.')
        lo = min(np.searchsorted(values, point[axis], side='right')-1, len(values)-2)
        fraction = (point[axis]-values[lo])/(values[lo+1]-values[lo])
        indices.append([lo, lo+1]); weights.append([1-fraction, fraction])
    return [(indices[0][a]*len(axes[1])+indices[1][b], weights[0][a]*weights[1][b])
            for a, b in itertools.product(range(2), repeat=2)]


def bake(indices, step, source_bounds, receiver_bounds):
    source_axes, sources = grid(source_bounds, step, .08)
    receiver_axes, receivers = grid(receiver_bounds, step, 1.6)
    delay = np.empty((len(sources), len(receivers), len(indices)), dtype=np.float32)
    gain = np.empty_like(delay)
    ir = np.empty((len(sources), len(receivers), RATE+1), dtype=np.float32)
    begin = time.perf_counter()
    for s, source in enumerate(sources):
        for r, receiver in enumerate(receivers):
            d, g = paths(source, receiver, indices)
            delay[s, r], gain[s, r] = d, g
            ir[s, r] = impulses(d, g)
    return dict(source_axes=source_axes, receiver_axes=receiver_axes, sources=sources,
                receivers=receivers, delay=delay, gain=gain, ir=ir,
                seconds=time.perf_counter()-begin)


def lookup(cache, source, receiver):
    begin = time.perf_counter()
    delay, gain = np.zeros(cache['delay'].shape[-1]), np.zeros(cache['gain'].shape[-1])
    raw = np.zeros(RATE+1)
    for (s, a), (r, b) in itertools.product(neighbors(cache['source_axes'], source),
                                          neighbors(cache['receiver_axes'], receiver)):
        delay += a*b*cache['delay'][s, r]
        gain += a*b*cache['gain'][s, r]
        raw += a*b*cache['ir'][s, r]
    elapsed = time.perf_counter()-begin
    return delay, gain, raw, elapsed


def errors(reference, candidate):
    bands = []
    frequencies = np.fft.rfftfreq(len(reference), 1/RATE)
    a, b = abs(rfft(reference))**2, abs(rfft(candidate))**2
    for lo, hi in [(125, 500), (500, 2000), (2000, 8000), (8000, 16000)]:
        mask = (frequencies >= lo) & (frequencies < hi)
        bands.append(float(10*np.log10(b[mask].sum()/a[mask].sum())))
    return dict(relativeImpulseL2=float(np.linalg.norm(candidate-reference)/np.linalg.norm(reference)),
                energyDifferenceDb=float(10*np.log10(np.sum(candidate**2)/np.sum(reference**2))),
                bandEnergyDifferenceDb=bands)


def checks():
    s, r = np.array([1.8, 2.2, .08]), np.array([4.3, 2.9, 1.6])
    indices = image_indices(1)
    d, g = paths(s, r, indices)
    manual = []
    for axis in range(3):
        for face in [0., ROOM[axis]]:
            image = s.copy(); image[axis] = 2*face-s[axis]
            manual.append(np.linalg.norm(image-r))
    np.testing.assert_allclose(np.sort(d*SPEED), np.sort(manual), rtol=1e-14)
    np.testing.assert_allclose(g*d*SPEED, REFLECTION, rtol=1e-14)
    # Reciprocity compares swapped endpoints, independently of cache lookup.
    indices = image_indices(4)
    forward = impulses(*paths(s, r, indices))
    reverse = impulses(*paths(r, s, indices))
    np.testing.assert_allclose(forward, reverse, atol=2e-13, rtol=1e-10)
    axes, points = grid(SOURCE_BOUNDS, .5, .08)
    for index, point in enumerate(points):
        weights = neighbors(axes, point)
        assert abs(sum(w for _, w in weights)-1) < 1e-14
        assert next(w for i, w in weights if i == index) == 1
    return ['six analytic first-order image distances/gains', 'swapped-endpoint reciprocity',
            'interpolation partition of unity and exact grid-node selection']


def fade(x):
    x = x.copy(); n = round(.02*RATE)
    ramp = np.sin(np.linspace(0, np.pi/2, n))**2
    x[:n] *= ramp; x[-n:] *= ramp[::-1]
    return x


def main():
    os.nice(19)
    verified = checks()
    input_path, output = Path(sys.argv[1]), Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=False)
    rate, dry = wavfile.read(input_path)
    if rate != RATE or dry.ndim != 1 or not np.issubdtype(dry.dtype, np.floating):
        raise ValueError('Supply the same mono float32 44.1 kHz test signal for all methods.')
    if len(dry) < 5*RATE or not np.isfinite(dry).all():
        raise ValueError('Supply at least five seconds of finite source PCM.')
    dry = fade(dry[:5*RATE].astype(float))
    indices = image_indices(ORDER)
    pairs = [(np.array(s), np.array(r)) for s, r in [
        ([1.8, 2.2, .08], [4.3, 2.9, 1.6]),
        ([1.15, 2.8, .08], [4.85, 3.2, 1.6]),
        ([1.42, 2.37, .08], [4.64, 2.68, 1.6]),
        ([1.67, 2.72, .08], [4.18, 3.38, 1.6])]]
    report = dict(scope='Interpolation experiment in an ideal closed shoebox; not Church, a moving walk, dynamic doors or a runtime speedup benchmark.',
        roomMetres=ROOM.tolist(), axes='x/y horizontal, z up', reflectionPressure=REFLECTION,
        maxBounces=ORDER, reflectedPaths=len(indices), sampleRate=RATE, checks=verified,
        direct='Calculated exactly for each query, excluded from the bake.',
        interpolation='Bilinear source × bilinear receiver. Same image index matches reflected paths. Compare raw IR blending with delay/gain interpolation.',
        sourceBounds=SOURCE_BOUNDS, receiverBounds=RECEIVER_BOUNDS,
        inputSha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        inputNote='Existing recording-based dense-rain test signal, unchanged across methods; not a new rain model.',
        cases=[], caches=[], files={})
    selected = {}
    local_bounds = lambda point: [[float(p-.05), float(p+.05)] for p in point[:2]]
    experiments = [(1., SOURCE_BOUNDS, RECEIVER_BOUNDS, pairs),
                   (.5, SOURCE_BOUNDS, RECEIVER_BOUNDS, pairs),
                   (.1, local_bounds(pairs[0][0]), local_bounds(pairs[0][1]), pairs[:1])]
    for step, source_bounds, receiver_bounds, queries in experiments:
        cache = bake(indices, step, source_bounds, receiver_bounds)
        report['caches'].append(dict(spacingMetres=step, sourceSamples=len(cache['sources']),
            listenerSamples=len(cache['receivers']), bakeSeconds=cache['seconds'],
            sourceBounds=source_bounds, receiverBounds=receiver_bounds,
            pathBytes=cache['delay'].nbytes+cache['gain'].nbytes, rawIrBytes=cache['ir'].nbytes))
        np.savez(output/f'cache-{step:g}m.npz', indices=indices, **{k:cache[k] for k in ['sources','receivers','delay','gain','ir']})
        for index, (source, receiver) in enumerate(queries):
            begin = time.perf_counter()
            d, g = paths(source, receiver, indices)
            fresh = direct(source, receiver)+impulses(d, g)
            fresh_seconds = time.perf_counter()-begin
            dc, gc, raw, lookup_seconds = lookup(cache, source, receiver)
            aligned = direct(source, receiver)+impulses(dc, gc)
            raw += direct(source, receiver)
            report['cases'].append(dict(spacingMetres=step, source=source.tolist(), receiver=receiver.tolist(),
                freshResponseSeconds=fresh_seconds, cacheInterpolationSeconds=lookup_seconds,
                timingNote='Lookup timing excludes conversion of interpolated paths into the IR; not comparable to the full fresh-response time.',
                delayWeightedRmsMs=float(1000*np.sqrt(np.sum(g*g*(dc-d)**2)/np.sum(g*g))),
                reflectedGainRelativeRms=float(np.linalg.norm(gc-g)/np.linalg.norm(g)),
                raw=errors(fresh, raw), aligned=errors(fresh, aligned)))
            if index == 0:
                selected['fresh'] = fftconvolve(dry, fresh)
                selected[f'aligned-{step:g}m'] = fftconvolve(dry, aligned)
                selected[f'raw-{step:g}m'] = fftconvolve(dry, raw)
        print('Finished spacing', step, 'bake seconds', round(cache['seconds'], 3), flush=True)
        del cache
    gain = min(10**(-28/20)/np.sqrt(np.mean(selected['fresh'][:5*RATE]**2)),
               .8/max(np.max(abs(x)) for x in selected.values()))
    gap = np.zeros(round(.5*RATE))
    clips = {f'{name}.wav': fade(audio*gain) for name, audio in selected.items()}
    clips['fresh-then-cached.wav'] = np.concatenate([clips['fresh.wav'], gap, clips['aligned-0.1m.wav']])
    clips['fresh-then-raw-blend.wav'] = np.concatenate([clips['fresh.wav'], gap, clips['raw-1m.wav']])
    for name, audio in clips.items():
        assert np.isfinite(audio).all() and np.max(abs(audio)) <= .800001
        path = output/name; wavfile.write(path, RATE, audio.astype(np.float32))
        report['files'][name] = dict(seconds=len(audio)/RATE, peak=float(np.max(abs(audio))),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report['auditionGain'] = float(gain)
    report['audition'] = 'One common gain for all methods; five-second dry signal plus one-second IR tail, no per-method normalization or limiter.'
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'index.html').write_text(Path(__file__).with_name('listen.html').read_text())
    for row in report['cases']:
        print(row['spacingMetres'], 'delay rms ms', round(row['delayWeightedRmsMs'], 3),
              'raw/aligned energy dB', round(row['raw']['energyDifferenceDb'], 3), round(row['aligned']['energyDifferenceDb'], 3))


if __name__ == '__main__':
    main()
