"""Offline statistical surface-rain prototype; not the production path tracer.

One deterministic texture per surface patch. Direct shadow rays integrate
incoherent area power to give stereo gains, with a shared centroid delay per
patch. This deliberately omits within-patch phase structure, reflections,
diffraction, transmission, HRTF and structural sound.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from compare_wave_field import shadow_mask
import rain_texture


@dataclass(frozen=True)
class Patch:
    x: tuple
    z: tuple
    height: float
    profile: str

    @property
    def area(self):
        return (self.x[1]-self.x[0])*(self.z[1]-self.z[0])

    @property
    def center(self):
        return np.array([sum(self.x)/2, self.height, sum(self.z)/2])


def patches(surfaces, width):
    if not np.isfinite(width) or width <= 0:
        raise ValueError('Patch width must be finite and positive.')
    result = []
    for s in surfaces:
        for x in np.arange(*s['x'], width):
            for z in np.arange(*s['z'], width):
                result.append(Patch((x, min(x+width, s['x'][1])),
                                    (z, min(z+width, s['z'][1])),
                                    s['height'], s['profile']))
    return result


def sites(patches, side):
    """Stratified midpoint quadrature. Changing side does not change the source."""
    if side < 1:
        raise ValueError('Positive sample-grid side required.')
    u, v = np.meshgrid((np.arange(side)+.5)/side, (np.arange(side)+.5)/side)
    return np.array([np.column_stack((p.x[0]+u.ravel()*(p.x[1]-p.x[0]),
                                     np.full(side*side, p.height),
                                     p.z[0]+v.ravel()*(p.z[1]-p.z[0]))) for p in patches])


def response(patches, points, receiver, walls, right=(0., 0., 1.), speed=343.):
    receiver = np.asarray(receiver)
    delta = points-receiver
    d2 = np.sum(delta*delta, axis=-1)
    if np.any(d2 <= 0):
        raise ValueError('Listener coincides with a source site.')
    visible = ~shadow_mask(dict(source=receiver, walls=walls), points.reshape(-1, 3))
    visible = visible.reshape(d2.shape)
    # Equal-power stereo pan, not head scattering or a binaural HRTF.
    pan = np.clip(np.sum(delta*np.asarray(right), axis=-1)/np.sqrt(d2), -1, 1)
    weights = np.array([p.area for p in patches])[:, None]*visible/d2/points.shape[1]
    power = np.stack((np.sum(weights*(1-pan)/2, axis=1),
                      np.sum(weights*(1+pan)/2, axis=1)), axis=1)
    delay = np.linalg.norm(np.array([p.center for p in patches])-receiver, axis=1)/speed
    return np.sqrt(power), delay


def mix_patch(pcm, seed, region, sample_rate, knots, gain, delay, start, end,
              grain_seconds=4., epoch_seconds=1.):
    """Render an absolute output interval; query partition never restarts a source.

    Global response knots are fixed before batching. Cubic interpolation reads
    the continuously delayed source. This is not bandlimited Doppler rendering.
    """
    samples = np.arange(start, end)
    seconds = samples/sample_rate
    read = (seconds+epoch_seconds-np.interp(seconds, knots, delay))*sample_rate
    if not len(read):
        return np.zeros((0, 2))
    # Cardinal cubic spline prefilter has a decaying boundary transient. Padding
    # keeps it far below float precision in the returned interval.
    lo, hi = int(np.floor(read.min()))-64, int(np.ceil(read.max()))+65
    if lo < 0:
        raise ValueError('Insufficient source pre-roll.')
    source = rain_texture.recording(pcm, round(grain_seconds*sample_rate),
                                    seed, region, lo, hi)
    delayed = map_coordinates(source, [read-lo], order=3, mode='nearest')
    envelope = np.column_stack([np.interp(seconds, knots, gain[:, ear]) for ear in range(2)])
    return delayed[:, None]*envelope
