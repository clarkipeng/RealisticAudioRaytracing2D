"""Interior volume, surfaces and predicted reverberation of the Church acoustic model.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY docs/benchmarks/energy-audio-2026-11-09/church_room.py TAKE_DIR [MATERIALS_JSON]

Voxelises the take's acoustic triangles at their first recorded pose (doors closed), flood-fills
the air from outside, and takes the enclosed air as the interior. Every triangle whose interior
side touches that air is an interior surface; its area is summed per material. With the take's
materials (or MATERIALS_JSON: {name: {"absorption": [8 bands], ...}}) it predicts Sabine and
Eyring reverberation times per octave band (with air absorption 4mV at 20 C, 50-70% RH).
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]
# Air: energy attenuation m (1/m) = alpha / 4.343 from ISO 9613-1 at 20 C, 50% relative humidity
# (alpha in dB/m: 0.00044 ... 0.105 at 125 Hz ... 8 kHz), the values the renderer now uses
# (Transport.hlsl AirCoefficient). Until 10 November this was Vorlaender's 50-70% table, lower at 4-8 kHz.
AIR_M = [a / 4.343 for a in [0.00044, 0.00131, 0.00273, 0.00466, 0.00989, 0.02967, 0.10529]]


def rotate(q, v):
    return v + 2 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)


def world_triangles(take):
    meta = json.loads((take / 'take.json').read_text())
    scene = meta['scene']
    g = np.frombuffer((take / 'geometry.bin').read_bytes(), dtype=[('v', '<f4', (3, 3)), ('o', '<i4'), ('m', '<i4')])
    frames = np.fromfile(take / 'states.bin', dtype=np.dtype([('sample', '<i8'), ('state', '<f4', (len(scene['objects']), 20))]))
    pose = frames['state'][0].astype(float)
    tri = g['v'].astype(float)
    world = np.empty_like(tri)
    for o in np.unique(g['o']):
        m = g['o'] == o
        world[m] = pose[o, :3] + rotate(pose[o, 3:7], tri[m].reshape(-1, 3)).reshape(-1, 3, 3)
    return scene, world, g['o'], g['m']


def analyse(take, materials=None, cell=.2, exclude=('Meadow',), seal=6, point=(0, 3, -20), assignments=None):
    scene, tri, owner, material = world_triangles(Path(take))
    names = [o['name'] for o in scene['objects']]
    keep = np.array([names[o] not in exclude for o in owner])
    tri, owner, material = tri[keep], owner[keep], material[keep]
    lo, hi = tri.reshape(-1, 3).min(0) - 1, tri.reshape(-1, 3).max(0) + 1
    shape = np.ceil((hi - lo) / cell).astype(int) + 1
    solid = np.zeros(shape, bool)
    for t in tri:  # dense barycentric sampling at a quarter cell
        e1, e2 = t[1] - t[0], t[2] - t[0]
        n = int(np.ceil(max(np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(t[2] - t[1])) / (cell / 4))) + 1
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        ok = u + v <= 1
        p = t[0] + u[ok, None] * e1 + v[ok, None] * e2
        idx = np.floor((p - lo) / cell).astype(int)
        solid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    # Close gaps narrower than ~2 cells (open joints, roof/wall seams) before separating inside from outside;
    # the interior is then grown back by the same amount so its volume is not shrunk.
    closed = ndimage.binary_dilation(solid, iterations=seal)
    air = ~closed
    labels, count = ndimage.label(air)
    outside = labels[0, 0, 0]
    probe = np.floor((np.asarray(point) - lo) / cell).astype(int)
    interior_label = labels[tuple(probe)]
    if interior_label == 0 or interior_label == outside:
        raise ValueError(f'The interior point is not sealed from the outside at seal={seal}.')
    interior = ndimage.binary_dilation(labels == interior_label, iterations=seal) & ~solid
    volume = interior.sum() * cell ** 3
    # interior-facing area per material: a triangle whose centroid +/- 0.6 cell along its normal is interior air
    c = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = .5 * np.linalg.norm(n, axis=1)
    n /= np.maximum(2 * area, 1e-12)[:, None]
    def inside(p):
        idx = np.clip(np.floor((p - lo) / cell).astype(int), 0, shape - 1)
        return interior[idx[:, 0], idx[:, 1], idx[:, 2]]
    sides = inside(c + n * cell * 1.5).astype(int) + inside(c - n * cell * 1.5).astype(int)
    facing = sides > 0
    per = {}
    for m in np.unique(material[facing]):
        per[scene['materials'][m]['name']] = float((area * sides)[facing & (material == m)].sum())  # both faces of thin sheets count
    per_object = {}
    for o in np.unique(owner[facing]):
        per_object[names[o]] = float((area * sides)[facing & (owner == o)].sum())
    S = sum(per.values())
    if assignments:  # predict with other materials: object name -> material name
        per = {}
        for name, s_ in per_object.items():
            key = assignments.get(name, name)
            per[key] = per.get(key, 0) + s_
    mats = materials or {m['name']: m for m in scene['materials']}
    rows = []
    for b, f in enumerate(BANDS):
        A, alpha = 0.0, []
        for name, s in per.items():
            mat = mats[name] if name in mats else next(m for m in scene['materials'] if m['name'] == name)
            if 'frequencies' not in mat:
                a = mat['absorption'][b]
            else:  # take curve: linear in Hz between knots
                a = float(np.interp(f, mat['frequencies'], mat['absorption']))
            A += s * a
        mean = A / S
        air = 4 * AIR_M[b] * volume
        sabine = .161 * volume / (A + air)
        eyring = .161 * volume / (-S * np.log(1 - mean) + air)
        rows.append(dict(band=f, meanAbsorption=mean, sabine=sabine, eyring=eyring))
    return dict(volume=volume, surface=S, perMaterial=per, perObject=per_object, bands=rows, cell=cell)


if __name__ == '__main__':
    materials = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else None
    r = analyse(sys.argv[1], materials)
    print('interior volume', round(r['volume']), 'm3, surface', round(r['surface']), 'm2')
    print(json.dumps({k: v for k, v in r.items() if k != 'perObject'}, indent=1))
    print('largest interior objects (m2):', sorted(r['perObject'].items(), key=lambda kv: -kv[1])[:12])


def openings(take, cell=.2, point=(0, 3, -20), exclude=('Meadow',)):
    """Where the interior air reaches the outside: the smallest dilation that seals the interior, and the
    air cells just beyond that sealed interior which a one-step-smaller dilation still connects outside
    (openings of about 2 * seal * cell metres)."""
    scene, tri, owner, _ = world_triangles(Path(take))
    names = [o['name'] for o in scene['objects']]
    tri = tri[np.array([names[o] not in exclude for o in owner])]
    lo = tri.reshape(-1, 3).min(0) - 1
    shape = np.ceil((tri.reshape(-1, 3).max(0) + 1 - lo) / cell).astype(int) + 1
    solid = np.zeros(shape, bool)
    for t in tri:
        e1, e2 = t[1] - t[0], t[2] - t[0]
        n = int(np.ceil(max(np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(t[2] - t[1])) / (cell / 4))) + 1
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        ok = u + v <= 1
        idx = np.floor((t[0] + u[ok, None] * e1 + v[ok, None] * e2 - lo) / cell).astype(int)
        solid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    probe = tuple(np.floor((np.asarray(point) - lo) / cell).astype(int))
    inner = None
    for seal in range(0, 9):
        lab, _ = ndimage.label(~(ndimage.binary_dilation(solid, iterations=seal) if seal else solid))
        if lab[probe] != lab[0, 0, 0]:
            break
    else:
        return dict(seal=None, points=np.zeros((0, 3)))
    inner = lab == lab[probe]
    if seal == 0:
        return dict(seal=0, points=np.zeros((0, 3)))
    lab2, _ = ndimage.label(~ndimage.binary_dilation(solid, iterations=seal - 1) if seal > 1 else ~solid)
    reach = lab2 == lab2[probe]
    grown = ndimage.binary_dilation(inner, iterations=seal + 2)
    front = reach & ~grown & ndimage.binary_dilation(grown, iterations=1)
    return dict(seal=seal, width=(2 * seal - 1) * cell, points=np.argwhere(front) * cell + lo)


def openings_figure(take, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = openings(take)
    p = r['points']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    if len(p):
        axes[0].scatter(p[:, 0], p[:, 2], s=4, c=p[:, 1], cmap='viridis'); axes[0].set_xlabel('x (m)'); axes[0].set_ylabel('z (m)'); axes[0].set_aspect('equal')
        sc = axes[1].scatter(p[:, 0], p[:, 1], s=4, c=p[:, 2], cmap='viridis'); axes[1].set_xlabel('x (m)'); axes[1].set_ylabel('height y (m)')
        fig.colorbar(sc, ax=axes[1], label='z (m)')
    fig.suptitle(f"Openings from the interior to the outside (sealed by a {r['seal']}-cell dilation, about {r.get('width', 0):.1f} m wide)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return r
