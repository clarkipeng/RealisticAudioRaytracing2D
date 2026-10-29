"""Seeded procedural box scenes for the audio-DLSS denoiser dataset.

One `scene.json` drives both the Unity tracer (TakeChecks.MeasureDatasetScenes,
through the automatic rigid solid bake) and FDTD voxelisation
(tools/fdtd_wave.py). Both read the same `walls` list of axis-aligned boxes
[{center, half}], `source` and flat `points` (metres, y up). Every box face lies
on a 1/16 m lattice. That is exact in float32, so the Unity solid bake sees the
same corners as the JSON (a 5 cm lattice produced sliver triangles), and FDTD
grids of 6.25, 3.125 or 1.5625 cm represent the boxes exactly.

Families, parameter ranges and the split rule are frozen in this module and
versioned by GENERATOR_VERSION. Random numbers come from named streams derived
with SHA-256 (never Python's hash), so a scene id regenerates byte-identically.
Pure NumPy; no Unity, GPU or solver is needed to generate scenes.
"""
import hashlib
import json
import math

import numpy as np

GENERATOR_VERSION = 'dlss-scenes-v2'
SCENE_SCHEMA = 'dlss-scene-v1'
LATTICE = .0625        # Box faces are integer multiples of this (dyadic, exact in float32).
WALL = .25             # Wall, floor and roof thickness (canopy benchmark convention).
RECEIVER_CLEARANCE = .3
SOURCE_CLEARANCE = .5
SOURCE_RECEIVER_MINIMUM = .5
TARGET_RECEIVERS = 72
MAX_RECEIVERS = 80

# Renderer grid: 8 kHz, 64-point FFT, so bins are exact multiples of 125 Hz.
SAMPLE_RATE = 8000
FFT_SIZE = 64
FREQUENCIES = [125.0 * k for k in range(1, FFT_SIZE // 2 + 1)]   # 125 Hz .. 4 kHz
MAX_BOUNCES = 32
MAXIMUM_DELAY = 1.25
SPEED_OF_SOUND = 343.0

# Budgets: the label is the converged tracer, K independent seeds at identical
# settings; inputs are separate runs at fractions of one label run.
LABEL_BUDGET = dict(raysPerBin=512, completeRouteSamples=65536)
LABEL_SEEDS = 8        # v1 used 2; its early-field labels failed acceptance (calibration)
INPUT_FRACTIONS = (256, 64, 16, 4)
INPUT_SEEDS = 4

# FDTD subset: rigid, semi-open families only (a lossless closed room has no
# finite-time single-frequency response), band-limited to 250 Hz.
FDTD_SPACING = .03125  # 44 points per wavelength at 250 Hz
FDTD_FREQUENCIES = [125.0, 250.0]
FDTD_PEAK_HZ = 300.0
FDTD_PML = 16
FDTD_MARGIN = .5
FDTD_DURATION = dict(doorway=.8, corridor=.6, canopy=.4, barriers=.3)
FDTD_MAX_CELLS = 30e6

# Surface materials are the project's existing presets (Assets/Materials/Church
# <name>.asset energy curves); no new coefficients. Floors, walls and roofs share
# one preset per role and scene; clutter picks per box. The FDTD subset is also
# rendered with every surface rigid (variant tracer-rigid), as FDTD has no
# wall impedance yet.
ROLES = ('floor', 'wall', 'roof', 'clutter')
ROLE_PRESETS = dict(floor=('stone', 'wood'), wall=('stone', 'wood'), roof=('stone', 'wood'),
                    clutter=('wood', 'stone', 'glass'))

FAMILIES = ('shoebox', 'doorway', 'corridor', 'canopy', 'barriers', 'furnished')
CLOSED_FAMILIES = ('shoebox', 'furnished')


def split_for(family, parameters):
    """Frozen split rule: whole families and parameter ranges, never per receiver.

    validation holds out the corridor/coupled family and dense furniture
    (8-10 items requested); test holds out the canopy family and doorways wider
    than 1.5 m. The generator draws every third doorway/furnished scene from
    the held-out range.
    """
    if family == 'canopy':
        return 'test'
    if family == 'corridor':
        return 'validation'
    if family == 'doorway' and parameters['openingWidth'] > 1.5:
        return 'test'
    if family == 'furnished' and parameters['items'] >= 8:
        return 'validation'
    return 'train'


def split_group(family, parameters):
    """Split ancestry group recorded in the manifest (one group, one split)."""
    if family == 'doorway':
        return 'doorway-wide' if parameters['openingWidth'] > 1.5 else 'doorway-narrow'
    if family == 'furnished':
        return 'furnished-dense' if parameters['items'] >= 8 else 'furnished-sparse'
    return family


def stable_seed(*parts):
    """31-bit seed from a versioned SHA-256 of the given name parts."""
    digest = hashlib.sha256('/'.join([GENERATOR_VERSION, *map(str, parts)]).encode()).digest()
    return int.from_bytes(digest[:4], 'little') & 0x7fffffff


def stream(*parts):
    return np.random.default_rng(stable_seed(*parts))


def snap(value):
    """Nearest lattice index (integer) for a coordinate in metres."""
    return int(round(value / LATTICE))


class Boxes:
    """Axis-aligned boxes stored as integer lattice bounds, each with a surface role."""

    def __init__(self):
        self.low, self.high, self.roles = [], [], []

    def add(self, low, high, role='wall'):
        if role not in ROLES:
            raise ValueError(f'Unknown surface role {role}')
        lo = [snap(v) for v in low]
        hi = [snap(v) for v in high]
        if any(h <= l for l, h in zip(lo, hi)):
            raise ValueError(f'Degenerate box {low} {high}')
        self.low.append(lo)
        self.high.append(hi)
        self.roles.append(role)

    def walls(self):
        out = []
        for lo, hi in zip(self.low, self.high):
            out.append(dict(center=[round((l + h) * LATTICE / 2, 6) for l, h in zip(lo, hi)],
                            half=[round((h - l) * LATTICE / 2, 6) for l, h in zip(lo, hi)]))
        return out


def box_arrays(walls):
    centers = np.array([w['center'] for w in walls], float).reshape(-1, 3)
    halves = np.array([w['half'] for w in walls], float).reshape(-1, 3)
    return centers - halves, centers + halves


def clearance(points, walls):
    """Euclidean distance from each point to the nearest box (0 inside a box)."""
    points = np.atleast_2d(np.asarray(points, float))
    if not walls:
        return np.full(len(points), np.inf)
    low, high = box_arrays(walls)
    outside = np.maximum(np.maximum(low[None] - points[:, None], points[:, None] - high[None]), 0)
    return np.linalg.norm(outside, axis=2).min(1)


def line_of_sight(source, points, walls):
    """Analytic segment/box test (independent of the shader's classification)."""
    points = np.atleast_2d(np.asarray(points, float))
    source = np.asarray(source, float)
    visible = np.ones(len(points), bool)
    direction = points - source
    for low, high in zip(*box_arrays(walls)):
        enter, leave = np.zeros(len(points)), np.ones(len(points))
        hit = np.ones(len(points), bool)
        for axis in range(3):
            d = direction[:, axis]
            s = source[axis]
            parallel = np.abs(d) < 1e-12
            hit &= ~parallel | ((s > low[axis]) & (s < high[axis]))
            with np.errstate(divide='ignore', invalid='ignore'):
                a = np.where(parallel, -np.inf, (low[axis] - s) / d)
                b = np.where(parallel, np.inf, (high[axis] - s) / d)
            enter = np.maximum(enter, np.minimum(a, b))
            leave = np.minimum(leave, np.maximum(a, b))
        visible &= ~(hit & (enter < leave))
    return visible


def receiver_grid(regions, height, walls, rng):
    """Horizontal receiver grids at `height`, one per rectangular region.

    Each region is (x0, x1, z0, z1). A common spacing is chosen so the total is
    about TARGET_RECEIVERS before clearance filtering; a random sub-cell offset
    keeps receivers off the FDTD lattice. Returns points (n,3) and integer
    (region, ix, iz) grid coordinates used for neighbour baselines.
    """
    area = sum((x1 - x0) * (z1 - z0) for x0, x1, z0, z1 in regions)
    spacing = max(.3, math.sqrt(area / TARGET_RECEIVERS))
    offset = rng.uniform(.2, .8, 2)
    points, grid = [], []
    for r, (x0, x1, z0, z1) in enumerate(regions):
        nx = max(1, int((x1 - x0) / spacing))
        nz = max(1, int((z1 - z0) / spacing))
        sx, sz = (x1 - x0) / nx, (z1 - z0) / nz
        for i in range(nx):
            for k in range(nz):
                points.append([x0 + (i + offset[0]) * sx, height, z0 + (k + offset[1]) * sz])
                grid.append([r, i, k])
    points, grid = np.array(points), np.array(grid, int)
    keep = clearance(points, walls) >= RECEIVER_CLEARANCE
    return points[keep], grid[keep], spacing


def pick_source(rng, region, walls, attempts=2000):
    """Uniform source in a box region (x0,x1,y0,y1,z0,z1) with wall clearance."""
    x0, x1, y0, y1, z0, z1 = region
    for _ in range(attempts):
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(z0, z1)])
        if clearance(p, walls)[0] >= SOURCE_CLEARANCE:
            return p
    raise RuntimeError('No source position satisfies the clearance.')


def lattice_value(rng, low, high):
    """Uniform lattice coordinate in [low, high] metres."""
    return snap(rng.uniform(low, high)) * LATTICE


def room(boxes, x0, x1, z0, z1, height, openings=(), floor=True, roof=True):
    """Interior [x0,x1]x[0,height]x[z0,z1] enclosed by WALL-thick slabs.

    `openings` lists (side, a, b, bottom, top): a hole in wall side
    '-x'|'+x'|'-z'|'+z' spanning [a,b] along the wall and [bottom,top] in y.
    """
    t = WALL
    if floor:
        boxes.add([x0 - t, -t, z0 - t], [x1 + t, 0, z1 + t], 'floor')
    if roof:
        boxes.add([x0 - t, height, z0 - t], [x1 + t, height + t, z1 + t], 'roof')
    sides = {'-x': ('z', x0 - t, x0, z0 - t, z1 + t), '+x': ('z', x1, x1 + t, z0 - t, z1 + t),
             '-z': ('x', z0 - t, z0, x0, x1), '+z': ('x', z1, z1 + t, x0, x1)}
    for side, (along, n0, n1, a0, a1) in sides.items():
        holes = sorted((a, b, lo, hi) for s, a, b, lo, hi in openings if s == side)
        cursor = a0
        segments = []
        for a, b, lo, hi in holes:
            segments.append((cursor, a, 0, height))
            if lo > 0:
                segments.append((a, b, 0, lo))
            if hi < height:
                segments.append((a, b, hi, height))
            cursor = b
        segments.append((cursor, a1, 0, height))
        for a, b, lo, hi in segments:
            if b - a < LATTICE / 2 or hi - lo < LATTICE / 2:
                continue
            if along == 'z':
                boxes.add([n0, lo, a], [n1, hi, b])
            else:
                boxes.add([a, lo, n0], [b, hi, n1])


def shoebox(rng, index):
    lx, lz = lattice_value(rng, 3, 8), lattice_value(rng, 3, 7)
    h = lattice_value(rng, 2.5, 4)
    b = Boxes()
    room(b, 0, lx, 0, lz, h)
    p = dict(length=lx, depth=lz, height=h)
    return b, p, [(0, lx, 0, lz)], (0, lx, 1, min(2.2, h - .5), 0, lz)


def doorway(rng, index):
    # Every third scene draws from the held-out wide-opening range.
    width = lattice_value(rng, 1.5625, 2.0) if index % 3 == 2 else lattice_value(rng, .8, 1.5)
    lx, lz = lattice_value(rng, 3, 7), lattice_value(rng, max(3, width + 1.5), 6)
    h = lattice_value(rng, 2.6, 3.6)
    window = rng.uniform() < .3
    center = lattice_value(rng, .6 + width / 2, lz - .6 - width / 2)
    bottom, top = (lattice_value(rng, .8, 1.0), lattice_value(rng, 1.9, min(2.4, h - .3))) if window \
        else (0.0, lattice_value(rng, 1.9, min(2.4, h - .3)))
    b = Boxes()
    room(b, 0, lx, 0, lz, h, openings=[('+x', center - width / 2, center + width / 2, bottom, top)])
    outside = (lx + WALL, lx + WALL + 3.5, -1.0, lz + 1.0)
    p = dict(length=lx, depth=lz, height=h, openingWidth=width, openingBottom=bottom, openingTop=top,
             window=bool(window), sourceInside=bool(rng.uniform() < .5))
    region = (.5, lx - .5, 1, 2, .5, lz - .5) if p['sourceInside'] else \
        (lx + WALL + .8, lx + WALL + 3.2, 1, 2, -.5, lz + .5)
    return b, p, [(0, lx, 0, lz), outside], region


def corridor(rng, index):
    kind = ['straight', 'L', 'coupled'][int(rng.integers(3))]
    h = lattice_value(rng, 2.4, 3.2)
    b = Boxes()
    if kind == 'straight':
        w, length = lattice_value(rng, 1.2, 2.5), lattice_value(rng, 6, 10)
        # Open ends: no end walls; floor and roof span the corridor only.
        b.add([0, -WALL, -WALL], [length, 0, w + WALL], 'floor')
        b.add([0, h, -WALL], [length, h + WALL, w + WALL], 'roof')
        b.add([0, 0, -WALL], [length, h, 0])
        b.add([0, 0, w], [length, h, w + WALL])
        regions = [(-1.5, length + 1.5, 0, w)]
        source = (.3, length - .3, 1, 2, .3, w - .3)
        p = dict(kind=kind, width=w, length=length, height=h)
    elif kind == 'L':
        w = lattice_value(rng, 1.2, 2.2)
        a, c = lattice_value(rng, 4, 7), lattice_value(rng, 4, 7)
        # Leg A along +x from x=0 (open end) to x=a+w; leg B along +z from the corner (open end at z=c+w).
        b.add([0, -WALL, -WALL], [a + w + WALL, 0, w + WALL], 'floor')
        b.add([a, -WALL, w + WALL], [a + w + WALL, 0, c + w], 'floor')
        b.add([0, h, -WALL], [a + w + WALL, h + WALL, w + WALL], 'roof')
        b.add([a, h, w + WALL], [a + w + WALL, h + WALL, c + w], 'roof')
        b.add([0, 0, -WALL], [a + w + WALL, h, 0])           # outer wall of leg A
        b.add([a + w, 0, 0], [a + w + WALL, h, c + w])       # outer wall of leg B
        b.add([0, 0, w], [a, h, w + WALL])                   # inner wall of leg A
        b.add([a - WALL, 0, w + WALL], [a, h, c + w])        # inner wall of leg B
        regions = [(-1.5, a + w, 0, w), (a, a + w, w, c + w + 1.5)]
        source = (.3, a + w - .3, 1, 2, .3, w - .3)
        p = dict(kind=kind, width=w, legA=a, legB=c, height=h)
    else:
        l1, l2 = lattice_value(rng, 3, 5), lattice_value(rng, 3, 5)
        d = lattice_value(rng, 3, 5)
        door = lattice_value(rng, .8, 1.6)
        inner = lattice_value(rng, .5 + door / 2, d - .5 - door / 2)
        outer = lattice_value(rng, .5 + 1.0 / 2, d - .5 - 1.0 / 2)
        top = lattice_value(rng, 2.0, min(2.3, h - .2))
        room(b, 0, l1, 0, d, h, openings=[('+x', inner - door / 2, inner + door / 2, 0, top),
                                          ('-x', outer - .5, outer + .5, 0, top)])
        x2 = l1 + WALL
        room(b, x2, x2 + l2, 0, d, h, openings=[('-x', inner - door / 2, inner + door / 2, 0, top)])
        regions = [(-2.5 - WALL, -WALL, 0, d), (0, l1, 0, d), (x2, x2 + l2, 0, d)]
        source = (x2 + .5, x2 + l2 - .5, 1, 2, .5, d - .5)
        p = dict(kind=kind, room1=l1, room2=l2, depth=d, height=h, doorWidth=door)
    return b, p, regions, source


def canopy(rng, index):
    kind = ['posts', 'overhang', 'porch'][int(rng.integers(3))]
    h = lattice_value(rng, 2.4, 4.0)
    sx, sz = lattice_value(rng, 3, 6), lattice_value(rng, 3, 5.5)
    b = Boxes()
    b.add([0, h, 0], [sx, h + WALL, sz], 'roof')                 # roof, no floor
    if kind == 'posts':
        for x, z in [(0, 0), (sx - WALL, 0), (0, sz - WALL), (sx - WALL, sz - WALL)]:
            b.add([x, 0, z], [x + WALL, h, z + WALL])
        sides = 0
    else:
        b.add([0, 0, 0], [sx, h, WALL])                          # back wall
        sides = 0 if kind == 'overhang' else int(rng.integers(1, 3))
        for s in range(sides):
            x = 0 if s == 0 else sx - WALL
            depth = lattice_value(rng, sz / 2, sz)
            b.add([x, 0, WALL], [x + WALL, h, depth])
    regions = [(-2, sx + 2, -1.5 if kind == 'posts' else WALL, sz + 2.5)]
    inside = rng.uniform() < .4
    source = (.5, sx - .5, 1, 2, .6, sz - .5) if inside else (-1.5, sx + 1.5, 1, 2, sz + .8, sz + 2.2)
    p = dict(kind=kind, height=h, width=sx, depth=sz, sideWalls=sides, sourceUnder=bool(inside))
    return b, p, regions, source


def barriers(rng, index):
    size = lattice_value(rng, 6, 9)
    count = int(rng.integers(2, 7))
    b = Boxes()
    placed = []
    for _ in range(200):
        if len(placed) == count:
            break
        if rng.uniform() < .5:   # wall segment
            length, thick = lattice_value(rng, 1.5, 4), lattice_value(rng, .2, .4)
            hx, hz = (length, thick) if rng.uniform() < .5 else (thick, length)
            height = lattice_value(rng, 1.5, 3.5)
        else:                    # pillar or block
            hx = hz = lattice_value(rng, .3, 1.0)
            height = lattice_value(rng, 2, 4)
        x, z = lattice_value(rng, 1, size - 1 - hx), lattice_value(rng, 1, size - 1 - hz)
        low, high = np.array([x, 0, z]), np.array([x + hx, height, z + hz])
        if any(np.all(low[[0, 2]] < h[[0, 2]] + .6) and np.all(high[[0, 2]] > l[[0, 2]] - .6) for l, h in placed):
            continue
        placed.append((low, high))
        b.add(low, high)
    regions = [(0, size, 0, size)]
    side = int(rng.integers(4))
    edge = [(-1.5, -.5, 1, 2, 0, size), (size + .5, size + 1.5, 1, 2, 0, size),
            (0, size, 1, 2, -1.5, -.5), (0, size, 1, 2, size + .5, size + 1.5)][side]
    p = dict(size=size, obstacles=len(placed), sourceSide=side)
    return b, p, regions, edge


def furnished(rng, index):
    lx, lz = lattice_value(rng, 5, 8), lattice_value(rng, 4.5, 7)
    h = lattice_value(rng, 2.5, 3.5)
    # Every third scene draws from the held-out dense range (8-10 items).
    items = int(rng.integers(8, 11)) if index % 3 == 2 else int(rng.integers(2, 8))
    b = Boxes()
    room(b, 0, lx, 0, lz, h)
    placed = 0
    for _ in range(300):
        if placed == items:
            break
        kind = rng.uniform()
        if kind < .4:     # table-height block
            wx, wz, hy = lattice_value(rng, .8, 2), lattice_value(rng, .6, 1.2), lattice_value(rng, .7, .8)
        elif kind < .8:   # cabinet / shelf
            wx, wz, hy = lattice_value(rng, .4, 2), lattice_value(rng, .3, .6), lattice_value(rng, 1, 2.2)
        else:             # free-standing partition
            wx, wz, hy = lattice_value(rng, 1, 2.5), .125, lattice_value(rng, 1.4, min(2.2, h - .3))
        if rng.uniform() < .5:
            wx, wz = wz, wx
        x, z = lattice_value(rng, .1, lx - .1 - wx), lattice_value(rng, .1, lz - .1 - wz)
        low, high = [x, 0, z], [x + wx, hy, z + wz]
        overlap = False
        for lo, hi in zip(b.low[6:], b.high[6:]):
            lo, hi = np.array(lo) * LATTICE, np.array(hi) * LATTICE
            if x < hi[0] + .4 and x + wx > lo[0] - .4 and z < hi[2] + .4 and z + wz > lo[2] - .4:
                overlap = True
        if overlap:
            continue
        b.add(low, high, 'clutter')
        placed += 1
    p = dict(length=lx, depth=lz, height=h, items=items, placed=placed)
    return b, p, [(0, lx, 0, lz)], (.5, lx - .5, 1, min(2.2, h - .5), .5, lz - .5)


BUILDERS = dict(shoebox=shoebox, doorway=doorway, corridor=corridor, canopy=canopy,
                barriers=barriers, furnished=furnished)


def surface_materials(roles, rng):
    """One preset per role and scene; clutter boxes choose individually."""
    shared = {role: str(rng.choice(ROLE_PRESETS[role])) for role in ROLES if role != 'clutter'}
    return [shared[r] if r in shared else str(rng.choice(ROLE_PRESETS[r])) for r in roles]


def budgets(scene_id):
    """Resolved render recipe: label seeds and separate input budgets/seeds."""
    out = [dict(name='label', seeds=[stable_seed(scene_id, 'label', k) for k in range(LABEL_SEEDS)], **LABEL_BUDGET)]
    for fraction in INPUT_FRACTIONS:
        out.append(dict(name=f'input-1-{fraction}',
                        raysPerBin=max(1, LABEL_BUDGET['raysPerBin'] // fraction),
                        completeRouteSamples=max(1, LABEL_BUDGET['completeRouteSamples'] // fraction),
                        seeds=[stable_seed(scene_id, 'input', fraction, k) for k in range(INPUT_SEEDS)]))
    return out


def fdtd_plan(scene, family):
    """FDTD grid and cost estimate; eligibility excludes closed lossless rooms."""
    from fdtd_wave import make_grid, scene_bounds
    low, high = scene_bounds(scene, FDTD_MARGIN)
    grid = make_grid(low, high, FDTD_SPACING, FDTD_PML)
    duration = FDTD_DURATION.get(family)
    dt = .99 / math.sqrt(3) * FDTD_SPACING / SPEED_OF_SOUND
    steps = int(math.ceil(duration / dt)) if duration else 0
    eligible = family not in CLOSED_FAMILIES and grid.cells <= FDTD_MAX_CELLS
    return dict(eligible=bool(eligible), spacing=FDTD_SPACING, frequencies=FDTD_FREQUENCIES, peakHz=FDTD_PEAK_HZ,
                pml=FDTD_PML, margin=FDTD_MARGIN, duration=duration, cells=grid.cells, steps=steps,
                gcellUpdates=grid.cells * steps / 1e9,
                reason=None if eligible else ('closed lossless room' if family in CLOSED_FAMILIES else 'domain too large'))


def generate(family, index):
    """Build one scene description; deterministic in (family, index)."""
    if family not in BUILDERS:
        raise ValueError(f'Unknown family {family}')
    scene_id = f'{family}-{index:04d}'
    rng = stream(scene_id, 'geometry')
    boxes, parameters, regions, source_region = BUILDERS[family](rng, index)
    walls = boxes.walls()
    materials = surface_materials(boxes.roles, stream(scene_id, 'materials'))
    endpoints = stream(scene_id, 'endpoints')
    source = pick_source(endpoints, source_region, walls)
    height = round(float(endpoints.uniform(1.2, 1.8)), 3)
    points, grid, spacing = receiver_grid(regions, height, walls, endpoints)
    keep = np.linalg.norm(points - source, axis=1) >= SOURCE_RECEIVER_MINIMUM
    points, grid = points[keep], grid[keep]
    if len(points) > MAX_RECEIVERS:
        chosen = np.sort(endpoints.choice(len(points), MAX_RECEIVERS, replace=False))
        points, grid = points[chosen], grid[chosen]
    if len(points) < 16:
        raise RuntimeError(f'{scene_id}: only {len(points)} receivers survive clearance filtering.')
    points = np.round(points, 6)
    source = np.round(source, 6)
    scene = dict(
        schema=SCENE_SCHEMA, generator=GENERATOR_VERSION, sceneId=scene_id, family=family, index=index,
        parameters=parameters, split=split_for(family, parameters), splitGroup=split_group(family, parameters),
        closed=family in CLOSED_FAMILIES, lattice=LATTICE,
        source=source.tolist(), walls=walls, surfaceRoles=boxes.roles, surfaceMaterials=materials,
        points=points.ravel().tolist(),
        receiverGrid=dict(region=grid[:, 0].tolist(), ix=grid[:, 1].tolist(), iz=grid[:, 2].tolist(),
                          spacing=round(spacing, 6), height=height),
        lineOfSight=line_of_sight(source, points, walls).astype(int).tolist(),
        frequencies=FREQUENCIES, sampleRate=SAMPLE_RATE, fftSize=FFT_SIZE, speedOfSound=SPEED_OF_SOUND,
        maxBounces=MAX_BOUNCES, maximumDelaySeconds=MAXIMUM_DELAY, bakeSolids=True,
        render=dict(budgets=budgets(scene_id)),
        referenceConvention='exp(+i*w*t)')
    scene['fdtd'] = fdtd_plan(scene, family)
    validate(scene)
    return scene


def validate(scene):
    """Geometry/endpoint invariants shared by both solvers."""
    walls = scene['walls']
    low, high = box_arrays(walls)
    for value in np.concatenate([low.ravel(), high.ravel()]):
        if abs(value / LATTICE - round(value / LATTICE)) > 1e-6:
            raise ValueError(f'{scene["sceneId"]}: face {value} is off the {LATTICE} m lattice.')
    points = np.array(scene['points']).reshape(-1, 3)
    if clearance(scene['source'], walls)[0] < SOURCE_CLEARANCE - 1e-9:
        raise ValueError('Source clearance violated.')
    if np.any(clearance(points, walls) < RECEIVER_CLEARANCE - 1e-9):
        raise ValueError('Receiver clearance violated.')
    if np.any(np.linalg.norm(points - scene['source'], axis=1) < SOURCE_RECEIVER_MINIMUM - 1e-9):
        raise ValueError('Receiver too close to the source.')
    for hz in scene['frequencies']:
        if (hz * scene['fftSize'] / scene['sampleRate']) % 1:
            raise ValueError('Frequency is not an exact FFT bin.')


def dumps(scene):
    """Canonical bytes: the scene hash is SHA-256 of exactly these bytes."""
    return (json.dumps(scene, indent=1, sort_keys=True, allow_nan=False) + '\n').encode()


def neighbours(scene):
    """Indices of 4-connected grid neighbours of every receiver (same region)."""
    g = scene['receiverGrid']
    keys = {(r, i, k): n for n, (r, i, k) in enumerate(zip(g['region'], g['ix'], g['iz']))}
    out = []
    for r, i, k in zip(g['region'], g['ix'], g['iz']):
        out.append([keys[c] for c in [(r, i - 1, k), (r, i + 1, k), (r, i, k - 1), (r, i, k + 1)] if c in keys])
    return out
