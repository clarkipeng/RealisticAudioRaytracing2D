"""Empirical dense rain texture audition, not individually simulated raindrops.

Seeded half-second recorded passages overlap at constant-power windows. Recording
acoustics remain baked in; this is not a calibrated dry material response. No
room processing, playback, transport budget or rain-rate control is provided.
"""
import numpy as np


VERSION = 'rain-recorded-texture-audition-v1'


def render(bank, seed, region, start, end):
    """Absolute sample queries; bounded two-grain mixing without per-drop state.

    Bank shape is (responses, samples), with an even response length. The caller
    owns the bank's sample rate. Random polarity removes coherent ensemble mean
    between independently seeded regions without changing each grain's spectrum.
    It does not guarantee perceptual independence or remove recording ambience.
    """
    if bank.ndim != 2 or not len(bank) or bank.shape[1] < 2 or bank.shape[1] % 2:
        raise ValueError('Expected a nonempty bank with even-length responses.')
    if not 0 <= start <= end:
        raise ValueError('Invalid sample interval.')
    length = bank.shape[1]
    hop = length//2
    window = np.sin(np.pi*np.arange(length)/length)
    out = np.zeros(end-start)
    if start == end:
        return out
    for index in range((start-length)//hop+1, (end-1)//hop+1):
        # The first overlapping index is -1 at t=0. Its contribution initializes
        # a stationary stream; a requested interval does not restart the envelope.
        rng = np.random.default_rng(np.random.SeedSequence([seed, region, index+1]))
        voice, polarity = rng.integers(len(bank)), 2*rng.integers(2)-1
        at = index*hop
        lo, hi = max(start, at), min(end, at+length)
        out[lo-start:hi-start] += bank[voice, lo-at:hi-at]*window[lo-at:hi-at]*polarity
    return out
