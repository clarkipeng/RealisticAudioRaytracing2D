"""Offline rain excitation experiment; no Unity, playback, room or roof model.

Finite authored response banks plus a seeded marked Poisson process. The compact
renderer bins impulses by response and convolves them in bounded blocks. This is
equivalent to explicitly mixing each response, not thinning/boosting the drops.
Requires NumPy/SciPy. Amplitudes and intensity are uncalibrated authoring units.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.signal import butter, sosfilt


RATE = 44100
EVENT = np.dtype([('sample', '<i8'), ('voice', '<i4'), ('gain', '<f8')])
VERSION = 'rain-source-experiment-v1'


@dataclass
class Bank:
    pcm: np.ndarray
    probability: np.ndarray
    description: dict

    @property
    def digest(self):
        return hashlib.sha256(self.pcm.astype('<f8').tobytes() +
                              self.probability.astype('<f8').tobytes()).hexdigest()


def make_bank(surface, rate=RATE):
    """Authored dry shapes, not measured material or airborne pressure responses.

    Water combines broad impact noise with optional damped bubble-like chirps.
    The oscillator frequency uses Minnaert's water-bubble relation; its airborne
    coupling, decay, occurrence and size relation here are illustrative presets.
    No structural resonator or material-propagation coefficients are used.
    """
    if surface not in ('hard', 'water') or rate < 32000:
        raise ValueError('Use hard/water and a sample rate of at least 32 kHz.')
    duration = .08
    t = np.arange(round(rate * duration)) / rate
    sizes = [.7, 1.1, 1.7, 2.6]
    fractions = [.60, .28, .10, .02]
    variants = 8
    shapes, weights, descriptions = [], [], []
    for size, fraction in zip(sizes, fractions):
        for variant in range(variants):
            rng = np.random.default_rng(np.random.SeedSequence([173, round(size*100), variant]))
            # A noisy splash with size-dependent spectrum and envelope. Shape
            # diversity is baked once; all events refer to the immutable bank.
            cutoff = min(.42*rate, 7800/np.sqrt(size))
            filtered = sosfilt(butter(2, [180, cutoff], fs=rate, btype='bandpass', output='sos'),
                               rng.standard_normal(len(t)))
            attack = .00018 * np.sqrt(size)
            decay = .0025 * size
            envelope = -np.expm1(-t/attack) * np.exp(-t/decay)
            splash_age = np.maximum(0, t-.0015*size)
            splash = -np.expm1(-splash_age/.0006) * np.exp(-splash_age/(.007*size))
            shape = filtered * (envelope + .28*splash)
            frequency = None
            if surface == 'water' and variant < 4:
                radius = rng.uniform(.25, .65) * 1e-3
                frequency = np.sqrt(3*1.4*101325/1000)/(2*np.pi*radius)
                age = np.maximum(0, t-rng.uniform(.002, .008))
                tau = rng.uniform(.002, .006)
                phase = 2*np.pi*frequency*(age + .04*age*age/(2*tau))
                shape += .8*np.sin(phase)*(-np.expm1(-age/.00015))*np.exp(-age/tau)
            # Smooth support and zero integral, without equalizing each drop's
            # RMS/peak. Bigger authored size classes retain different energies.
            fade = np.ones(len(t))
            n = round(.01*rate)
            fade[-n:] = np.cos(np.linspace(0, np.pi/2, n))**2
            shape *= fade
            correction = (envelope+.28*splash)*fade
            shape -= shape.sum()*correction/correction.sum()
            shape *= size**1.5
            shape[0] = shape[-1] = 0
            shapes.append(shape)
            weights.append(fraction/variants)
            descriptions.append(dict(sizeClassMm=size, variant=variant, bubbleHz=frequency))
    return Bank(np.array(shapes), np.array(weights), dict(
        version=VERSION, surface=surface, sampleRate=rate, durationSeconds=duration,
        scope='Authored dry response family; no measured surface, calibrated SPL or structural radiation.',
        sizeDistribution='Illustrative four-class mixture, not a fitted rainfall distribution.',
        responses=descriptions))


def events(bank, seed, region, start, end, intensity=1., rate=RATE):
    """Stable absolute one-second blocks; arbitrary query/batch boundaries agree.

    Intensity 1 means 4,000 events/s in this one-region experiment, not mm/h.
    Equal-onset events are allowed and are accumulated, never overwritten.
    """
    if not (0 <= start <= end) or not np.isfinite(intensity) or intensity < 0:
        raise ValueError('Invalid interval or intensity.')
    chunks = []
    for second in range(start//rate, (end+rate-1)//rate):
        rng = np.random.default_rng(np.random.SeedSequence([seed, region, second]))
        count = rng.poisson(4000*intensity)
        chunk = np.empty(count, dtype=EVENT)
        chunk['sample'] = second*rate + rng.integers(0, rate, count)
        chunk['voice'] = rng.choice(len(bank.pcm), count, p=bank.probability)
        chunk['gain'] = rng.uniform(.7, 1.3, count) * .003
        chunk = chunk[(chunk['sample'] >= start) & (chunk['sample'] < end)]
        chunks.append(chunk[np.argsort(chunk['sample'], kind='stable')])
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=EVENT)


def explicit(bank, event_list, samples):
    """Independent reference: add the complete kernel once for every event."""
    out = np.zeros(samples+bank.pcm.shape[1]-1)
    for event in event_list:
        at = int(event['sample'])
        if at < 0 or at >= samples:
            raise ValueError('Event outside source interval.')
        out[at:at+bank.pcm.shape[1]] += bank.pcm[event['voice']]*event['gain']
    return out


class PulseConvolver:
    """Bounded overlap-add: response count, block size and tail bound storage."""
    def __init__(self, bank, block_size=2048):
        if block_size <= 0:
            raise ValueError('Block size must be positive.')
        self.bank = bank
        self.block_size = block_size
        self.length = next_fast_len(block_size+bank.pcm.shape[1]-1)
        self.filters = rfft(bank.pcm, self.length, axis=1, workers=1)
        self.tail = np.zeros(bank.pcm.shape[1]-1)
        self.sample = 0

    @property
    def numeric_bytes(self):
        # Explicit persistent arrays + upper bound for impulses, FFT/product,
        # spectrum reduction, convolution and overlap buffers. Excludes FFT
        # implementation scratch, event list, output and the shared PCM bank.
        k, n, b = len(self.bank.pcm), self.length, self.block_size
        f = n//2+1
        return int(self.filters.nbytes+self.tail.nbytes+k*b*8+2*k*f*16+f*16+2*n*8)

    def process(self, event_list, count):
        if not 0 < count <= self.block_size:
            raise ValueError('Block exceeds capacity.')
        impulses = np.zeros((len(self.bank.pcm), self.block_size))
        local = event_list['sample']-self.sample
        if np.any((local < 0) | (local >= count)):
            raise ValueError('Event outside supplied block.')
        np.add.at(impulses, (event_list['voice'], local), event_list['gain'])
        spectrum = np.sum(rfft(impulses, self.length, axis=1, workers=1)*self.filters, axis=0)
        result = irfft(spectrum, self.length, workers=1)
        result[:len(self.tail)] += self.tail
        self.tail = result[count:count+len(self.tail)].copy()
        self.sample += count
        return result[:count]


def compact(bank, event_list, samples, block_size=2048):
    """Mix a sorted event list; the final returned samples drain the full tail."""
    renderer = PulseConvolver(bank, block_size)
    out = np.empty(samples+bank.pcm.shape[1]-1)
    for start in range(0, samples, block_size):
        end = min(samples, start+block_size)
        lo, hi = np.searchsorted(event_list['sample'], [start, end])
        out[start:end] = renderer.process(event_list[lo:hi], end-start)
    out[samples:] = renderer.tail
    return out
