"""The Church's room modes (README part A): voxelise the take's acoustic triangles (doors closed), bake
the interior's modes with tools/room_modes.py, measure the Schroeder frequency from them.

Walls: every interior face takes the material of the triangles in the solid cell beyond it, with the
energy that leaves the interior through it, absorption + transmission, as its random-incidence
absorption (converted to a real admittance by Paris's formula, as the FDTD ground truth does). Faces
onto the unglazed openings (gables, belfry) see no solid: sound leaves through them as through an open
window, absorption 1 (Sabine's convention, and what the path method's escaping rays do). An opening is not a
locally reacting wall, so it enters the first-order damping through the admittance whose diffuse-field
loss is that absorption: averaged over the modes, delta_n = (c/2) sum beta S_nm becomes Sabine's
c sum (8 beta) A / (8 V), so beta = alpha / 8 = 1/8 (beta = 1 at first order would lose 8 times too much).
Which side of a triangle faces the nave is read from the nave's mask at 0.2 m (church_room.py's
analysis), finer than the mode grid, so thin sheets and slabs are not counted from their far side.

    PY=.context/fdtd-teacher/venv/bin/python
    python3 tools/heavy_guard.py run --limit-gib 3.3 -- $PY $B/church_modes.py bake [HMAX]
    $PY $B/church_modes.py consistency [HMAX]      # modes against diffuse-field theory in the crossover octave
    python3 tools/heavy_guard.py run --limit-gib 3.3 -- $PY $B/church_modes.py fdtd [HMAX]
                                                   # the same voxel church solved by the FDTD: a check of the bake
"""
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'docs/benchmarks/energy-audio-2026-11-09'))
import fdtd_wave as fw  # noqa: E402
import room_modes as rm  # noqa: E402
import church_room as cr  # noqa: E402

TAKE = ROOT / '.context/energy-audio/church/speech-walk/take'
WORK = ROOT / '.context/bass-and-diffraction'
INSIDE = (0, 3, -20)                 # a point in the nave (church_room.py)
OPEN = 'opening'


def band_curve(material, key):
    return [float(np.interp(f, material['frequencies'], material[key])) for f in rm.mc.BANDS]


def voxels(h):
    """Rectilinear (uniform) edges, solid ids (-1 interior air) and material names at spacing h."""
    scene, tri, owner, mat = cr.world_triangles(TAKE)
    names = [o['name'] for o in scene['objects']]
    keep = np.array([names[o] != 'Meadow' for o in owner])
    tri, mat = tri[keep], mat[keep]
    lo = tri.reshape(-1, 3).min(0) - 2 * h
    hi = tri.reshape(-1, 3).max(0) + 2 * h
    shape = np.ceil((hi - lo) / h).astype(int)
    surface = -np.ones(shape, np.int16)
    for t, m in zip(tri, mat):                        # surface sampling at a quarter cell
        e1, e2 = t[1] - t[0], t[2] - t[0]
        n = int(np.ceil(max(np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(t[2] - t[1])) / (h / 4))) + 1
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        ok = u + v <= 1
        idx = np.floor((t[0] + u[ok, None] * e1 + v[ok, None] * e2 - lo) / h).astype(int)
        surface[idx[:, 0], idx[:, 1], idx[:, 2]] = m
    solid = surface >= 0
    probe = tuple(np.floor((np.asarray(INSIDE) - lo) / h).astype(int))
    for seal in range(0, 6):                          # the smallest dilation that separates the nave from outside
        lab, _ = ndimage.label(~(ndimage.binary_dilation(solid, iterations=seal) if seal else solid))
        if lab[probe] != lab[0, 0, 0]:
            break
    # The seal separates the nave's air from the outside's; every other air cell joins whichever of the two it is
    # nearer to (the dilation's leftovers in corners are nave air; the cells in an opening's throat are split
    # at its middle), and the faces between the nave and outside air are the openings.
    nave = lab == lab[probe]
    outside = (lab > 0) & ~nave
    d_nave = ndimage.distance_transform_edt(~nave)
    d_out = ndimage.distance_transform_edt(~outside)
    near = ~solid & (d_nave <= d_out)
    parts, _ = ndimage.label(near)
    interior = parts == parts[probe]                   # pockets cut off from the nave (inside the organ case,
    pockets = near & ~interior                         # the tower's voids) never meet its air: no faces
    mats = [m['name'] for m in scene['materials']] + [OPEN]
    stone = next(i for i, m in enumerate(scene['materials']) if m['name'] == 'Church stone')
    out = np.where(interior, -1, np.where(solid, surface, np.where(pockets, stone, len(mats) - 1))).astype(np.int16)
    edges = [lo[a] + h * np.arange(shape[a] + 1) for a in range(3)]
    return edges, out, mats, scene, seal


def nave_mask(tri, cell=.2, seal=6):
    """The nave's air at 0.2 m (church_room.analyse): surface voxels, sealed by 6 dilations, flood-filled
    from the nave, grown back."""
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
    labels, _ = ndimage.label(~ndimage.binary_dilation(solid, iterations=seal))
    probe = tuple(np.floor((np.asarray(INSIDE) - lo) / cell).astype(int))
    return ndimage.binary_dilation(labels == labels[probe], iterations=seal) & ~solid, lo, cell


def surface_integrals(bank, h, mats):
    """S_nm over the mesh's own interior-facing triangles instead of the voxel faces (a staircase has up to
    sqrt3 times the area of a slanted wall): psi^2 probed on points spread over each triangle at h / 2,
    just inside the air, times the area each stands for. Openings keep their voxel faces."""
    scene, tri, owner, mat = cr.world_triangles(TAKE)
    names = [o['name'] for o in scene['objects']]
    keep = np.array([names[o] != 'Meadow' for o in owner])
    tri, mat = tri[keep], mat[keep]
    edges = [bank['edges0'], bank['edges1'], bank['edges2']]
    solid = bank['solid']
    lo = np.array([e[0] for e in edges])

    def air(p):
        idx = np.clip(np.floor((p - lo) / h).astype(int), 0, np.array(solid.shape) - 1)
        return solid[idx[:, 0], idx[:, 1], idx[:, 2]] < 0

    fine, flo, cell = nave_mask(tri)

    def nave(p):
        idx = np.clip(np.floor((p - flo) / cell).astype(int), 0, np.array(fine.shape) - 1)
        return fine[idx[:, 0], idx[:, 1], idx[:, 2]]

    points, weights, materials = [], [], []
    for t, m in zip(tri, mat):
        e1, e2 = t[1] - t[0], t[2] - t[0]
        nrm = np.cross(e1, e2)
        area = 0.5 * np.linalg.norm(nrm)
        if area <= 0:
            continue
        nrm /= 2 * area
        n = int(np.ceil(max(np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(t[2] - t[1])) / (h / 2)))
        u, v = np.meshgrid((np.arange(n) + 1 / 3) / n, (np.arange(n) + 1 / 3) / n)
        ok = u + v < 1
        p = t[0] + u[ok, None] * e1 + v[ok, None] * e2
        for side in (1, -1):                          # each side of the triangle that faces the nave: the first air
            facing = nave(p + side * nrm[None] * 1.5 * cell)  # cell along its normal (each point: big triangles
            if not facing.any():                              # reach outside the nave)
                continue
            q = p.copy()
            inside = ~facing
            for step in np.arange(0.25, 4, 0.5):        # (up to four cells: a thick shell of voxels)
                trial = p + side * nrm * step * h
                new = ~inside & air(trial)
                q[new] = trial[new]
                inside |= new
            inside &= facing
            if inside.any():
                points.append(q[inside])
                weights.append(np.full(inside.sum(), area / ok.sum()))
                materials.append(np.full(inside.sum(), m))
    points, weights, materials = np.concatenate(points), np.concatenate(weights), np.concatenate(materials)
    S = np.zeros_like(bank['surface'])
    for a in range(0, len(points), 4096):
        psi = rm.probe(bank, points[a:a + 4096])
        for m in np.unique(materials[a:a + 4096]):
            sel = materials[a:a + 4096] == m
            S[:, m] += (weights[a:a + 4096][sel, None] * psi[sel] ** 2).sum(0)
    opening = mats.index(OPEN)
    S[:, opening] = bank['surface'][:, opening]
    return S, {mats[m]: float(weights[materials == m].sum()) for m in np.unique(materials)}


def bake(h):
    edges, solid, mats, scene, seal = voxels(h)
    absorption, admittance = [], []
    for name in mats:
        if name == OPEN:
            absorption.append([1.0] * 8)
            admittance.append([1.0 / 8] * 8)
            continue
        m = next(x for x in scene['materials'] if x['name'] == name)
        loss = [min(a + t, fw.STATISTICAL_ABSORPTION_MAX) for a, t in zip(band_curve(m, 'absorption'), band_curve(m, 'transmission'))]
        absorption.append(loss)
        admittance.append([fw.admittance_from_absorption(a) for a in loss])
    print(f'church at {h} m: seal {seal} cells, {int((solid < 0).sum())} interior cells', flush=True)
    out = WORK / f'bank-church-{h:g}'
    bank = rm.load_bank(out) if (out / 'bank.npz').exists() else \
        rm.bake_grid(edges, solid, mats, h, absorption=absorption, admittance=admittance, out=out)
    S, mesh_area = surface_integrals(bank, h, mats)
    np.save(out / 'surface-mesh.npy', S)
    k = bank['k']
    delta = rm.damping(k, S, admittance)
    fs = rm.schroeder_frequency(k, delta)
    report = dict(h=h, seal=seal, volume=float(bank['volume']), area=float(bank['area']),
                  voxelAreaPerMaterial={m: float(a) for m, a in zip(mats, bank['surface'][int(np.argmin(k))] * bank['volume'])},
                  meshAreaPerMaterial=mesh_area, modes=int(len(k)),
                  topHz=float(k.max() * rm.C / (2 * math.pi)), schroederHz=fs,
                  crossoverHz=[fs / math.sqrt(2), fs * math.sqrt(2)] if fs else None,
                  bakeSeconds=float(bank['seconds']),
                  modalT60AtSchroeder=float(3 * math.log(10) / np.mean(delta[np.abs(k * rm.C / (2 * math.pi) / fs - 1) < .2])) if fs else None,
                  shapeBytesFloat16=int(2 * len(k) * len(bank['volumes'])), admittance=dict(zip(mats, admittance)))
    (WORK / f'church-{h:g}.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({kk: v for kk, v in report.items() if kk != 'admittance'}, indent=1), flush=True)


def consistency(h, listeners=256, seed=0, source='organ'):
    """Energy consistency at the crossover: a source (the organ, or the nave's centre) heard at random nave
    points 1.2-1.8 m above the floor,
    band energy (the octave band centred on f_S, the suite's band shape) from the modes, against diffuse-field
    theory with the same surfaces and absorption, 1/r^2 + 16 pi / A (A = sum 8 beta_m S_m V over materials,
    the constant mode's wall integrals), and against it with Waterhouse's interior correction
    1 / (1 + S lambda / (8 V)). The path method estimates the uncorrected diffuse field."""
    report = json.loads((WORK / f'church-{h:g}.json').read_text())
    bank = rm.load_bank(WORK / f'bank-church-{h:g}')
    S = np.load(WORK / f'bank-church-{h:g}' / 'surface-mesh.npy')
    mats = [str(m) for m in bank['materials']]
    adm = [report['admittance'][m] for m in mats]
    k = bank['k']
    delta = rm.damping(k, S, adm)
    fs = report['schroederHz']
    meta = json.loads((TAKE / 'take.json').read_text())
    frames = np.fromfile(TAKE / 'states.bin', dtype=np.dtype([('sample', '<i8'), ('state', '<f4', (len(meta['scene']['objects']), 20))]))
    clear = ndimage.distance_transform_edt(bank['solid'] < 0) * h
    lam_s = rm.C / fs
    edges_ = [bank['edges0'], bank['edges1'], bank['edges2']]
    lo_ = np.array([e[0] for e in edges_])
    if source == 'organ':
        organ = frames['state'][0][1, :3].astype(float)
    else:                                              # 'free': the nave's point farthest from every surface
        organ = lo_ + (np.array(np.unravel_index(np.argmax(clear), clear.shape)) + 0.5) * h
    solid = bank['solid']
    edges = [bank['edges0'], bank['edges1'], bank['edges2']]
    lo = np.array([e[0] for e in edges])
    rng = np.random.default_rng(seed)
    cells = bank['cells']
    floor = {}                                         # lowest air cell per column
    for c in cells:
        floor[(c[0], c[2])] = min(floor.get((c[0], c[2]), 10 ** 9), c[1])
    pts = []
    keys = list(floor)
    free = np.argwhere(clear >= lam_s / 4)             # 'free': listeners a quarter wavelength from every surface
    while len(pts) < listeners:
        if source == 'free':
            p = lo + (free[rng.integers(len(free))] + rng.uniform(size=3)) * h
        else:
            i, j = keys[rng.integers(len(keys))]
            y = lo[1] + floor[(i, j)] * h + rng.uniform(1.2, 1.8)
            p = np.array([lo[0] + (i + rng.uniform()) * h, y, lo[2] + (j + rng.uniform()) * h])
        idx = np.floor((p - lo) / h).astype(int)
        if np.all(idx >= 0) and np.all(idx < solid.shape) and solid[tuple(idx)] < 0 and np.linalg.norm(p - organ) > 3:
            pts.append(p)
    pts = np.array(pts)
    a = rm.probe(bank, np.vstack([organ, pts]))
    top = 2 * fs
    freqs = np.arange(0, top + rm.DF, rm.DF)
    keep = k * rm.C / (2 * math.pi) <= top * 1.2
    H = rm.transfer(k[keep], delta[keep], a[0, keep], a[1:, keep], freqs)
    H[:, 0] = 0
    E, _ = rm.mc.energy_from_spectrum(H, rm.DF, bands=np.array([fs]))
    modal = E[:, 0].sum(-1)
    r = np.linalg.norm(pts - organ, axis=1)
    f_band = fs
    beta_f = rm.admittance_at(adm, np.array([f_band]))[0]
    A = float(np.sum(8 * beta_f * S[int(np.argmin(k))] * bank['volume']))
    diffuse = 1 / r ** 2 + 16 * math.pi / A
    lam = rm.C / f_band
    area = float(np.sum(S[int(np.argmin(k))] * bank['volume']))
    waterhouse = 1 / r ** 2 + 16 * math.pi / A / (1 + area * lam / (8 * bank['volume']))
    far = np.array([clear[tuple(np.floor((q - lo) / h).astype(int))] for q in pts]) >= lam_s / 4
    src_clear = float(clear[tuple(np.floor((organ - lo) / h).astype(int))])
    out = dict(source=source, sourceClearanceM=src_clear, quarterWavelengthM=lam_s / 4,
               interiorListeners=int(far.sum()),
               interiorModalOverWaterhouseDb=float(10 * np.log10(modal[far].mean() / waterhouse[far].mean())) if far.any() else None,
               schroederHz=fs, listeners=listeners, equivalentAreaM2=A, surfaceM2=area,
               modalOverDiffuseDb=dict(mean=float(10 * np.log10(modal.mean() / diffuse.mean())),
                                       receiverSdDb=float(np.std(10 * np.log10(modal / diffuse)))),
               modalOverWaterhouseDb=float(10 * np.log10(modal.mean() / waterhouse.mean())),
               modesBelowTop=int(keep.sum()))
    (WORK / f'church-consistency-{source}-{h:g}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


def fdtd(h, receivers=64, seed=1):
    """The bake's numerical check in the Church: the FDTD ground-truth solver on the very voxel model the
    modes were baked on (same cells, same walls as admittances held from 125 Hz down, openings beta = 1/8),
    one run, against the bank's modes with the voxel faces' wall integrals, in the octave bands centred on
    f_S and on the crossover's top. Two cases: source and listeners clear of every surface by a quarter
    wavelength at f_S, and the organ with listeners at head height."""
    import method_compare_suite as mcs
    sys.path.insert(0, str(HERE))
    import modes as md
    report = json.loads((WORK / f'church-{h:g}.json').read_text())
    bank = rm.load_bank(WORK / f'bank-church-{h:g}')
    mats = [str(m) for m in bank['materials']]
    adm = [report['admittance'][m] for m in mats]
    solid = bank['solid']
    ids = np.where(solid < 0, 0, solid + 1).astype(np.uint8)
    edges = [bank['edges0'], bank['edges1'], bank['edges2']]
    grid = fw.Grid(float(h), tuple(float(e[0]) for e in edges), tuple(int(n) for n in solid.shape), 0)
    table = {m + 1: float(adm[m][0]) for m in range(len(mats))}   # band 0's admittance (held below 125 Hz)
    k, S = bank['k'], bank['surface']
    delta = rm.damping(k, S, adm)
    fs = rm.schroeder_frequency(k, delta)
    bands = np.array([fs, math.sqrt(2) * fs])
    clear = ndimage.distance_transform_edt(solid < 0) * h
    lo = np.array([e[0] for e in edges])
    rng = np.random.default_rng(seed)
    meta = json.loads((TAKE / 'take.json').read_text())
    frames = np.fromfile(TAKE / 'states.bin', dtype=np.dtype([('sample', '<i8'), ('state', '<f4', (len(meta['scene']['objects']), 20))]))
    cells = bank['cells']
    out = {}
    for case in ('free', 'organ'):
        if case == 'free':
            src = lo + (np.array(np.unravel_index(np.argmax(clear), clear.shape)) + 0.5) * h
            pool = np.argwhere(clear >= rm.C / fs / 4)
            pts = lo + (pool[rng.choice(len(pool), receivers, replace=False)] + rng.uniform(size=(receivers, 3))) * h
        else:
            src = frames['state'][0][1, :3].astype(float)
            pool = cells[(lo[1] + cells[:, 1] * h > 1.0) & (lo[1] + cells[:, 1] * h < 2.5)]
            pts = lo + (pool[rng.choice(len(pool), receivers, replace=False)] + 0.5) * h
        sim = fw.Simulation(grid, ids, admittance=table, memory_limit=2 * fw.GIB)
        res = sim.run(src, pts, 2.05, 1.2 * fs)
        H, df = mcs.fdtd_transfer_spectrum(res['pressure'], res['source'], res['dt'])
        Eg, _ = rm.mc.energy_from_spectrum(H, df, bands)
        a = rm.probe(bank, np.vstack([src, pts]))
        top = 2 * bands.max()
        freqs = np.arange(0, top + rm.DF, rm.DF)
        Hm = rm.transfer(k, delta, a[0], a[1:], freqs)
        Hm[:, 0] = 0
        Em, _ = rm.mc.energy_from_spectrum(Hm, rm.DF, bands)
        vb = int(2.0 / rm.mc.BIN_SECONDS)
        rows = {}
        for b, fc in enumerate(bands):
            lev, edt, t30 = [], [], []
            for r in range(receivers):
                g, m = Eg[r, b].copy(), Em[r, b].copy()
                g[vb:] = 0
                m[vb:] = 0
                t0 = rm.mc.onset_bin(g[:vb])
                mg, mm = rm.mc.receiver_metrics(g, vb, t0), rm.mc.receiver_metrics(m, vb, t0)
                lev.append(mm['level'] - mg['level'])
                for lst, key in ((edt, 'EDT'), (t30, 'T30')):
                    if np.isfinite(mm[key]) and np.isfinite(mg[key]):
                        lst.append(100 * (mm[key] - mg[key]) / mg[key])
            rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else None
            rows[f'{fc:.1f}'] = dict(level=rms(lev), bias=float(np.mean(lev)), EDT=rms(edt), T30=rms(t30), T30receivers=len(t30))
        out[case] = dict(source=src.tolist(), bands=rows, fdtdSeconds=res['timing'].get('gpuSeconds'))
        print(case, json.dumps(rows), flush=True)
    (WORK / f'church-fdtd-{h:g}.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fdtd':
        fdtd(float(sys.argv[2]) if len(sys.argv) > 2 else 0.75)
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == 'consistency':
        consistency(float(sys.argv[2]) if len(sys.argv) > 2 else 0.75, source=sys.argv[3] if len(sys.argv) > 3 else 'organ')
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == 'bake':
        bake(float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
    else:
        raise SystemExit(__doc__)
