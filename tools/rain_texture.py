"""Empirical dense rain texture audition, not individually simulated raindrops.

Seeded recorded passages overlap at constant-power windows. Recording
acoustics remain baked in; this is not a calibrated dry material response. No
room processing, playback, transport budget or rain-rate control is provided.
"""
import numpy as np


VERSION = 'rain-recorded-texture-audition-v1'


def _mix(length, start, end, grain):
    if not 0 <= start <= end:
        raise ValueError('Invalid sample interval.')
    hop = length//2
    window = np.sin(np.pi*np.arange(length)/length)
    out = np.zeros(end-start)
    if start == end:
        return out
    for index in range((start-length)//hop+1, (end-1)//hop+1):
        at = index*hop
        lo, hi = max(start, at), min(end, at+length)
        pcm, polarity = grain(index)
        out[lo-start:hi-start] += pcm[lo-at:hi-at]*window[lo-at:hi-at]*polarity
    return out


def render(bank, seed, region, start, end):
    """Absolute sample queries; bounded two-grain mixing without per-drop state.

    Bank shape is (responses, samples), with an even response length. The caller
    owns the bank's sample rate. Random polarity removes coherent ensemble mean
    between independently seeded regions without changing each grain's spectrum.
    It does not guarantee perceptual independence or remove recording ambience.
    """
    if bank.ndim != 2 or not len(bank) or bank.shape[1] < 2 or bank.shape[1] % 2:
        raise ValueError('Expected a nonempty bank with even-length responses.')
    length = bank.shape[1]
    def grain(index):
        # The first overlapping index is -1 at t=0. Its contribution initializes
        # a stationary stream; a requested interval does not restart the envelope.
        rng = np.random.default_rng(np.random.SeedSequence([seed, region, index+1]))
        voice, polarity = rng.integers(len(bank)), 2*rng.integers(2)-1
        return bank[voice], polarity
    return _mix(length, start, end, grain)


def recording(pcm, grain_samples, seed, region, start, end):
    """Long passages with sample-resolution offsets, without a copied grain bank.

    Absolute time and region identity determine each excerpt. This reduces exact
    excerpt reuse across regions; independence must still be measured. Recording
    acoustics and material character remain part of the empirical source.
    """
    if pcm.ndim != 1 or grain_samples < 2 or grain_samples % 2 or len(pcm) < grain_samples:
        raise ValueError('Expected mono recording and an even grain length fitting it.')
    def grain(index):
        rng = np.random.default_rng(np.random.SeedSequence([seed, region, index+1]))
        offset = rng.integers(len(pcm)-grain_samples+1)
        polarity = 2*rng.integers(2)-1
        return pcm[offset:offset+grain_samples], polarity
    return _mix(grain_samples, start, end, grain)
