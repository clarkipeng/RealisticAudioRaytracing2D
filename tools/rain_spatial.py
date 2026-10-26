"""Small offline direct-sound fixture, not the Unity acoustic renderer.

Vertical rain hits the roof first or the exposed ground. Opaque boxes test
visibility; unblocked pressure scales as 1/r with nearest-sample travel delay.
There are no reflections, diffraction, transmission, HRTF or roof vibration.
"""
import numpy as np

import rain_source as rain
from compare_wave_field import shadow_mask


SPEED = 343.
SPATIAL_EVENT = np.dtype(rain.EVENT.descr + [('position', '<f8', (3,)), ('surface', '<i4')])


def scene():
    def box(center, half):
        return dict(center=center, half=half)
    return dict(
        version='rain-direct-shelter-v1', speedOfSound=SPEED,
        domainXZ=[[-10., 10.], [-10., 10.]],
        roofXZ=[[-4.25, .25], [-3.25, 3.25]], roofTop=3.25, sourceOffset=.001,
        # Source regions partition the horizontal catchment exactly. Patch
        # boundaries are clipped to them, never moving ground rain under a roof.
        surfaces=[
            dict(name='roof', x=[-4.25, .25], z=[-3.25, 3.25], height=3.25),
            dict(name='west ground', x=[-10., -4.25], z=[-10., 10.], height=0),
            dict(name='east ground', x=[.25, 10.], z=[-10., 10.], height=0),
            dict(name='south ground', x=[-4.25, .25], z=[-10., -3.25], height=0),
            dict(name='north ground', x=[-4.25, .25], z=[3.25, 10.], height=0)],
        walls=[box([-2., 3.125, 0], [2.25, .125, 3.25]),
               box([-4.125, 1.5, 0], [.125, 1.5, 3.25]),
               box([-2., 1.5, -3.125], [2.25, 1.5, .125]),
               box([-2., 1.5, 3.125], [2.25, 1.5, .125]),
               box([.125, 1.15, -2.], [.125, 1.15, 1.25]),
               box([.125, 1.15, 2.], [.125, 1.15, 1.25]),
               box([.125, 2.65, 0], [.125, .35, 3.25])],
        receivers=dict(outside=[3., 1.6, 0.], inside=[-2., 1.6, 0.],
                       doorway_edge=[-.1, 1.6, .65], roof_edge=[.4, 1.6, 2.9]),
        sourceModel='Same uncalibrated splash response on roof and ground; no structural radiation.',
        propagation='Direct visibility only; opaque slabs, 1/r pressure, nearest-sample delay, mono.')


def field(bank, seed, start, end, config, intensity=1.):
    """Canonical full seconds keep event positions invariant under query splits."""
    chunks = []
    for second in range(start//rain.RATE, (end+rain.RATE-1)//rain.RATE):
        base = rain.events(bank, seed, 0, second*rain.RATE, (second+1)*rain.RATE, intensity)
        chunk = np.zeros(len(base), dtype=SPATIAL_EVENT)
        for name in rain.EVENT.names:
            chunk[name] = base[name]
        rng = np.random.default_rng(np.random.SeedSequence([seed, second, 0x52504f53]))
        xz = rng.uniform(np.array(config['domainXZ'])[:, 0],
                         np.array(config['domainXZ'])[:, 1], (len(base), 2))
        chunk['position'][:, [0, 2]] = xz
        assigned = np.zeros(len(base), dtype=bool)
        for index, surface in enumerate(config['surfaces']):
            mask = ((xz[:, 0] >= surface['x'][0]) & (xz[:, 0] < surface['x'][1]) &
                    (xz[:, 1] >= surface['z'][0]) & (xz[:, 1] < surface['z'][1]))
            if np.any(assigned & mask):
                raise ValueError('Catchment regions overlap.')
            assigned |= mask
            chunk['surface'][mask] = index
            chunk['position'][mask, 1] = surface['height']+config['sourceOffset']
        if not assigned.all():
            raise ValueError('Catchment regions leave a gap.')
        chunks.append(chunk[(chunk['sample'] >= start) & (chunk['sample'] < end)])
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=SPATIAL_EVENT)


def patch_positions(events, config, width):
    """Approximate only position; retain every event identity and its surface."""
    if width <= 0 or not np.isfinite(width):
        raise ValueError('Patch width must be positive and finite.')
    positions = events['position'].copy()
    for index, surface in enumerate(config['surfaces']):
        mask = events['surface'] == index
        for axis, key in [(0, 'x'), (2, 'z')]:
            low, high = surface[key]
            edge = low+np.floor((positions[mask, axis]-low)/width)*width
            positions[mask, axis] = (edge+np.minimum(edge+width, high))/2
    return positions


def transfer(positions, receiver, config):
    distance = np.linalg.norm(positions-np.asarray(receiver), axis=1)
    if np.any(distance <= 0):
        raise ValueError('Receiver coincides with a source.')
    visible = ~shadow_mask(dict(source=receiver, walls=config['walls']), positions)
    return dict(visible=visible, gain=visible/distance,
                delay=np.rint(distance/config['speedOfSound']*rain.RATE).astype(np.int64),
                distance=distance)


def arrivals(events, path):
    visible = path['visible']
    result = np.empty(np.count_nonzero(visible), dtype=rain.EVENT)
    result['sample'] = events['sample'][visible]+path['delay'][visible]
    result['voice'] = events['voice'][visible]
    result['gain'] = events['gain'][visible]*path['gain'][visible]
    return result[np.argsort(result['sample'], kind='stable')]


def render(bank, events, positions, receiver, config, samples):
    path = transfer(positions, receiver, config)
    pulses = arrivals(events, path)
    # Fixed horizon covers every source/receiver within this fixture, independent
    # of which impacts are visible or how finely they are grouped.
    maximum_distance = np.linalg.norm([20., config['roofTop'], 20.])
    horizon = samples+int(np.ceil(maximum_distance/config['speedOfSound']*rain.RATE))
    if len(pulses) and pulses['sample'][-1] >= horizon:
        raise ValueError('Arrival exceeds fixture horizon.')
    return rain.compact(bank, pulses, horizon), path
