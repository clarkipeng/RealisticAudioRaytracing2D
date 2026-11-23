"""Room modes below the Schroeder frequency, and their crossover with the path method.

Geometric acoustics is the high-frequency limit of the wave equation; below the Schroeder frequency,
where fewer than three modes share a modal bandwidth, a closed room's response is its modes. This
tool bakes them per scene and couples them to a source and a listener:

* Rigid-room modes are the eigenpairs of the Neumann Laplacian, discretised by finite volumes on a
  rectilinear grid whose lines include every wall face (the same face-conforming cells as the FDTD
  ground truth, tools/fdtd_wave.py). K psi = k^2 M psi, M the cell volumes; psi normalised so that
  sum(M psi^2) = 1.
* Locally reacting walls of real normalised admittance beta (rho c / Z) damp mode n, to first order in
  beta, at delta_n = (c / 2) sum_m beta_m(f_n) S_nm, S_nm = integral of psi_n^2 over the faces of
  material m (Morse and Ingard). The bank stores S_nm, so materials (and their admittance at each
  mode's own frequency) can change without another solve.
* A monopole at s heard at r: H(w) = sum_n 4 pi c^2 psi_n(s) psi_n(r) / (w_n^2 - w^2 + 2 i delta_n w),
  in the suite's convention (exp(+iwt), free field exp(-ikr)/r).
* The crossover is the Schroeder frequency measured from the bank itself: the lowest f at which the
  modal overlap M(f) = n(f) * mean(delta_n / pi) reaches 3 (n the modes per Hz, both over the octave
  around f). Modes carry the response below it, the path method above it, blended over one octave
  centred on it by the renderer's own band shape, cos^2 in log2 frequency. Coherent parts (the direct
  sound, exact in both) are blended in amplitude, W_m + (1 - W_m) = 1; the path method's incoherent
  rest in power, W_m^2 + W_g^2 = 1. So nothing is counted twice and every part keeps its energy.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY tools/room_modes.py suite [--hmax 0.125]   # closed suite rooms: bake (cached), combine with the path
                                                    # method's runs, write methods path-modal and modal
    $PY tools/room_modes.py export BANK_DIR OUT.bytes   # the bank for the renderer (RoomModes.cs)

Heavy (the eigensolves): run under tools/heavy_guard.py.
"""
import argparse
import json
import math
import resource
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import method_compare as mc  # noqa: E402

C = 343.0
SCHROEDER_OVERLAP = 3.0          # Schroeder's criterion: three modes per modal bandwidth
BANK_SCHEMA = 'room-modes-v1'


# --------------------------------------------------------------------------- geometry

def boxes_of(scene):
    """Wall boxes (lo, hi) and their material indices, from a method-compare scene."""
    names = list(scene['materials'])
    boxes = []
    for w, m in zip(scene['walls'], scene['surfaceMaterials']):
        c, h = np.asarray(w['center'], float), np.asarray(w['half'], float)
        boxes.append((c - h, c + h, names.index(m)))
    return boxes, names


def rectilinear_edges(boxes, hmax, snap=1e-9):
    """Cell edges per axis: every box face inside the domain, intervals split to at most hmax."""
    lo = np.min([b[0] for b in boxes], 0)
    hi = np.max([b[1] for b in boxes], 0)
    edges = []
    for a in range(3):
        cuts = sorted({round(float(v), 9) for b in boxes for v in (b[0][a], b[1][a]) if lo[a] - snap <= v <= hi[a] + snap})
        out = [cuts[0]]
        for x0, x1 in zip(cuts[:-1], cuts[1:]):
            n = max(1, int(math.ceil((x1 - x0) / hmax - 1e-9)))
            out.extend(np.linspace(x0, x1, n + 1)[1:].tolist())
        edges.append(np.asarray(out))
    return edges


def voxelise(edges, boxes):
    """Material per cell: -1 for air, else the material index of the (first) box containing the centre."""
    centres = [0.5 * (e[1:] + e[:-1]) for e in edges]
    solid = -np.ones([len(c) for c in centres], np.int16)
    for lo, hi, m in boxes:
        sel = [np.nonzero((c > lo[a]) & (c < hi[a]))[0] for a, c in enumerate(centres)]
        if all(len(s) for s in sel):
            block = solid[np.ix_(*sel)]
            block[block < 0] = m
            solid[np.ix_(*sel)] = block
    return solid


# --------------------------------------------------------------------------- operator

def operator(edges, solid):
    """Finite-volume Neumann Laplacian on the air cells.

    Returns its parts along each axis [K_x, K_y, K_z] (sparse, symmetric, n x n; K = their sum), cell
    volumes m [n], the air cells' grid index [n, 3], and the wall faces as (air cell, face area,
    material); faces on the grid's outer boundary count as rigid (material -1) and are reported, since a
    closed room should not reach the boundary. -K_a psi / m is the second difference along axis a with
    the walls' zero normal gradient built in.
    """
    widths = [np.diff(e) for e in edges]
    air = solid < 0
    index = -np.ones(solid.shape, np.int64)
    cells = np.argwhere(air)
    index[air] = np.arange(len(cells))
    n = len(cells)
    vol = widths[0][cells[:, 0]] * widths[1][cells[:, 1]] * widths[2][cells[:, 2]]
    parts = []
    face_cell, face_area, face_mat = [], [], []
    for a in range(3):
        rows, cols, vals = [], [], []
        b, c = [x for x in range(3) if x != a]
        area_of = lambda ijk: widths[b][ijk[:, b]] * widths[c][ijk[:, c]]
        # interior faces between cell i and i + e_a
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[a], hi[a] = slice(0, -1), slice(1, None)
        A_, B_ = index[tuple(lo)], index[tuple(hi)]
        sl, sh = solid[tuple(lo)], solid[tuple(hi)]
        both = (A_ >= 0) & (B_ >= 0)
        ia, ib = A_[both], B_[both]
        ijk = cells[ia]
        dist = 0.5 * (widths[a][ijk[:, a]] + widths[a][ijk[:, a] + 1])
        g = area_of(ijk) / dist
        rows += [ia, ib, ia, ib]
        cols += [ia, ib, ib, ia]
        vals += [g, g, -g, -g]
        for mask, cell_side, mat in (((A_ >= 0) & (sh >= 0), A_, sh), ((B_ >= 0) & (sl >= 0), B_, sl)):
            ic = cell_side[mask]
            face_cell.append(ic)
            face_area.append(area_of(cells[ic]))
            face_mat.append(mat[mask].astype(np.int64))
        # outer boundary faces
        for end in (0, -1):
            s = [slice(None)] * 3
            s[a] = end
            ic = index[tuple(s)]
            ic = ic[ic >= 0]
            if len(ic):
                face_cell.append(ic)
                face_area.append(area_of(cells[ic]))
                face_mat.append(-np.ones(len(ic), np.int64))
        parts.append(sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)))
    faces = (np.concatenate(face_cell), np.concatenate(face_area), np.concatenate(face_mat))
    return parts, vol, cells, faces


def weyl_count(k, volume, surface):
    """Expected number of Neumann modes with wavenumber below k (Weyl, with the surface term)."""
    return volume * k ** 3 / (6 * math.pi ** 2) + surface * k ** 2 / (16 * math.pi)


def eigenmodes(parts, vol, widths, k0, k1, volume, surface, sink, per_slice=160, log=print):
    """All eigenpairs with k0 < k <= k1 of K psi = k^2 M psi, by shift-invert Lanczos in slices; each
    slice's (k^2, psi [n, m] float32) goes to sink, so the modes are never all held in memory.

    Each slice [l0, l1] of k^2 asks for the eigenvalues nearest its centre and is accepted only when
    the returned set reaches past both ends (so none inside was missed); otherwise it asks for more.

    The second-order stencil underestimates k^2 by (h^2 / 12) sum_a (d_a^2 psi)^2 (its truncation error,
    -(h^2/12) d^4/dx^4 per axis); the returned k^2 adds that term, measured on each eigenvector with the
    stencil's own second differences (first-order perturbation theory), which makes it fourth-order
    accurate. Slices partition the stencil's k^2; a mode the correction lifts past k1 is dropped (and one
    lifted past k0 from below is kept by the slice that found it). widths[a][n]: air cells' widths.
    """
    K = parts[0] + parts[1] + parts[2]
    s = 1 / np.sqrt(vol)
    B = sp.diags(s) @ K @ sp.diags(s)
    B = ((B + B.T) * 0.5).tocsc()
    n0, n1 = weyl_count(k0, volume, surface), weyl_count(k1, volume, surface)
    slices = max(1, int(math.ceil((n1 - n0) / per_slice)))
    ks = [k0]                                       # slice ends equally spaced in the Weyl count
    for j in range(1, slices + 1):
        target = n0 + (n1 - n0) * j / slices
        lo, hi = k0, k1
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if weyl_count(mid, volume, surface) < target else (lo, mid)
        ks.append(hi)
    ks[-1] = k1
    found = 0
    for j in range(slices):
        l0, l1 = ks[j] ** 2, ks[j + 1] ** 2
        sigma = 0.5 * (l0 + l1)
        want = int(1.4 * (weyl_count(ks[j + 1], volume, surface) - weyl_count(ks[j], volume, surface))) + 24
        t = time.time()
        while True:
            want = min(want, B.shape[0] - 2)
            w, v = sla.eigsh(B, k=want, sigma=sigma, which='LM', ncv=min(B.shape[0] - 1, 2 * want + 20))
            reach = np.abs(w - sigma).max()
            if (reach > sigma - l0 and reach > l1 - sigma) or want >= B.shape[0] - 2:
                break
            want = int(want * 1.6)
        floor = l0 if l0 > 0 else -np.inf           # the constant mode sits at k^2 = 0 (to rounding)
        keep = (w > floor) & (w <= l1)
        psi = v[:, keep] * s[:, None]               # back to psi, sum(M psi^2) = 1
        del v
        corr = sum(((widths[a] ** 2 / 12)[:, None] * (parts[a] @ psi) ** 2 / vol[:, None]).sum(0) for a in range(3))
        lam = np.maximum(w[keep], 0.0) + corr
        inside = lam <= l1 if j == slices - 1 else np.ones(len(lam), bool)
        sink(lam[inside], psi[:, inside].astype(np.float32))
        found += int(inside.sum())
        log(f'  slice {j + 1}/{slices}: k^2 [{l0:.1f}, {l1:.1f}] {inside.sum()} modes ({want} asked, {time.time() - t:.1f} s, '
            f'peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.2f} GiB)')
    return found


# --------------------------------------------------------------------------- bank

def schroeder_estimate(volume, face_area, face_material, absorption):
    """Sabine's Schroeder frequency 2000 sqrt(T / V), T from each material's largest absorption-free decay
    (its smallest absorption over the bands); used only to size a bake, never as the crossover."""
    A = sum(face_area[face_material == m].sum() * min(absorption[m]) for m in range(len(absorption)))
    return 2000 * math.sqrt(0.161 * volume / max(A, 1e-9) / volume)


def bake_boxes(boxes, materials, hmax, fmax=None, absorption=None, admittance=None, out=None,
               log=lambda m: print(m, flush=True)):
    """Bake the modes of a room given as wall boxes into the directory `out` (bank.npz + psi.f32).

    Without fmax the bake first reaches sqrt2 times Sabine's Schroeder estimate, then measures the
    Schroeder frequency from its own modes (with the materials' admittance) and extends itself in
    further slices until it holds every mode below the crossover's top, sqrt2 times that frequency."""
    edges = rectilinear_edges(boxes, hmax)
    return bake_grid(edges, voxelise(edges, boxes), materials, hmax, fmax, absorption, admittance, out, log)


def bake_grid(edges, solid, materials, hmax, fmax=None, absorption=None, admittance=None, out=None,
              log=lambda m: print(m, flush=True)):
    """bake_boxes on a given rectilinear grid: solid[i, j, k] = -1 for air, else the wall's material."""
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parts, vol, cells, (fc, fa, fm) = operator(edges, solid)
    volume = float(vol.sum())
    surface = float(fa.sum())
    if np.any(fm < 0):
        log(f'  warning: {int((fm < 0).sum())} air faces on the grid boundary are treated as rigid (room not closed)')
    widths = [np.diff(edges[a])[cells[:, a]] for a in range(3)]
    lams, surf = [], []
    psi_file = open(out / 'psi.f32', 'wb')         # [modes, cells], one row per mode

    def sink(lam, psi):
        lams.append(lam)
        S = np.zeros((psi.shape[1], len(materials)))
        for m in range(len(materials)):
            sel = fm == m
            S[:, m] = (fa[sel, None] * psi[fc[sel]].astype(np.float64) ** 2).sum(0)
        surf.append(S)
        psi_file.write(np.ascontiguousarray(psi.T).tobytes())

    top = 0.0
    target = fmax or math.sqrt(2) * schroeder_estimate(volume, fa, fm, absorption)
    while True:
        log(f'bake: grid {solid.shape}, {len(vol)} air cells, V {volume:.2f} m3, S {surface:.2f} m2, '
            f'modes {top:.0f}-{target:.0f} Hz (about {weyl_count(2 * math.pi * target / C, volume, surface) - weyl_count(2 * math.pi * top / C, volume, surface):.0f})')
        eigenmodes(parts, vol, widths, 2 * math.pi * top / C, 2 * math.pi * target / C, volume, surface, sink, log=log)
        top = target
        if fmax:
            break
        k, S = np.sqrt(np.concatenate(lams)), np.concatenate(surf)
        fs = schroeder_frequency(k, damping(k, S, admittance))
        need = math.sqrt(2) * fs if fs else 1.5 * top
        if need <= top:
            break
        target = need
    psi_file.close()
    k = np.sqrt(np.concatenate(lams))
    np.savez(out / 'bank.npz', schema=np.array(BANK_SCHEMA), edges0=edges[0], edges1=edges[1], edges2=edges[2],
             cells=cells.astype(np.int32), solid=solid, volumes=vol, k=k, surface=np.concatenate(surf),
             materials=np.array(materials), volume=np.array(volume), area=np.array(surface), hmax=np.array(hmax),
             fmax=np.array(top), seconds=np.array(time.time() - t0))
    log(f'bake: {len(k)} modes to {top:.0f} Hz in {time.time() - t0:.0f} s')
    return load_bank(out)


def load_bank(directory):
    """The bank as a dict; psi [cells, modes] is a read-only view of the memory-mapped psi.f32."""
    directory = Path(directory)
    bank = dict(np.load(directory / 'bank.npz'))
    shape = (len(bank['k']), len(bank['volumes']))
    bank['psi'] = np.memmap(directory / 'psi.f32', np.float32, 'r', shape=shape).T
    return bank


def box_modes(L, fmax, materials_of_faces, n_materials):
    """Analytic rigid-box modes (the test oracle): k, a shape evaluator, and S_nm.

    materials_of_faces: material index of the faces x=0, x=L0, y=0, y=L1, z=0, z=L2.
    """
    L = np.asarray(L, float)
    nmax = np.floor(2 * fmax * L / C).astype(int)
    idx = np.stack(np.meshgrid(*[np.arange(n + 1) for n in nmax], indexing='ij'), -1).reshape(-1, 3)
    k = np.pi * np.sqrt(((idx / L) ** 2).sum(1))
    keep = k <= 2 * math.pi * fmax / C
    idx, k = idx[keep], k[keep]
    eps = np.where(idx == 0, 1.0, 0.5)
    norm = np.sqrt(L.prod() * eps.prod(1))          # psi = prod cos(...) / norm, sum(M psi^2) = 1
    S = np.zeros((len(k), n_materials))
    for a in range(3):
        face = np.prod([L[b] * eps[:, b] for b in range(3) if b != a], 0) / norm ** 2
        for side in range(2):
            S[:, materials_of_faces[2 * a + side]] += face

    def shape(points):
        points = np.atleast_2d(points)
        return np.prod(np.cos(np.pi * idx[None] * points[:, None, :] / L), -1) / norm
    return k, shape, S


def probe(bank, points):
    """psi_n at points [P, 3] -> [P, N], fourth-order accurate.

    Trilinear interpolation over the eight cell centres around the point, minus its leading error,
    sum_a t_a (1 - t_a) h_a^2 / 2 times the interpolated second difference along a (the stencil's own,
    walls included). Across a wall the neighbour is the cell's mirror image (the zero normal gradient
    makes psi even about the wall), so a point between the last centre and the wall needs no other data.
    """
    edges = [bank['edges0'], bank['edges1'], bank['edges2']]
    solid = bank['solid']
    parts, vol, cells, _ = operator(edges, solid)
    centres = [0.5 * (e[1:] + e[:-1]) for e in edges]
    widths = [np.diff(e) for e in edges]
    index = -np.ones(solid.shape, np.int64)
    index[cells[:, 0], cells[:, 1], cells[:, 2]] = np.arange(len(cells))
    psi = bank['psi']
    plans = []
    for x in np.atleast_2d(points):
        c0 = [int(np.clip(np.searchsorted(edges[a], x[a]) - 1, 0, len(centres[a]) - 1)) for a in range(3)]
        if index[tuple(c0)] < 0:
            raise ValueError(f'point {x} lies in a solid cell')
        step, t, span = [], [], []
        for a in range(3):
            d = x[a] - centres[a][c0[a]]
            st = 1 if d >= 0 else -1
            nb = list(c0)
            nb[a] += st
            if 0 <= nb[a] < solid.shape[a] and index[tuple(nb)] >= 0:
                sp_ = abs(centres[a][nb[a]] - centres[a][c0[a]])
                step.append(st)
            else:                                   # mirror image across the wall: the same cell
                sp_ = widths[a][c0[a]]
                step.append(0)
            t.append(abs(d) / sp_)
            span.append(sp_)
        corners = []
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    delta = [dx, dy, dz]
                    w = np.prod([t[a] if delta[a] else 1 - t[a] for a in range(3)])
                    target = [c0[a] + delta[a] * step[a] for a in range(3)]
                    for a in range(3):              # a solid diagonal corner: step back along an axis
                        if index[tuple(target)] >= 0:
                            break
                        target[a] = c0[a]
                    corners.append((int(index[tuple(target)]), w))
        plans.append((corners, t, span))
    rows = np.unique([j for corners, _, _ in plans for j, _ in corners])
    where = {j: i for i, j in enumerate(rows)}
    second = []
    for a in range(3):                              # only the rows' neighbours of psi are read
        sub = parts[a][rows]                        # CSR: its indices are the neighbour columns
        cols = np.unique(sub.indices)
        second.append(-(sub[:, cols] @ np.asarray(psi[cols], np.float64)) / vol[rows, None])
    out = np.zeros((len(plans), psi.shape[1]))
    for p, (corners, t, span) in enumerate(plans):
        for j, w in corners:
            if w == 0:
                continue
            i = where[j]
            out[p] += w * (np.asarray(psi[j], np.float64) - sum(t[a] * (1 - t[a]) * span[a] ** 2 / 2 * second[a][i] for a in range(3)))
    return out


# --------------------------------------------------------------------------- response

def admittance_at(admittance_bands, f):
    """Each material's admittance at frequencies f: linear in log2 f between band centres, held outside."""
    x = np.log2(np.maximum(np.asarray(f, float), 1e-3) / mc.BANDS[0])
    return np.stack([np.interp(x, np.arange(len(mc.BANDS)), np.asarray(a, float)) for a in admittance_bands], 1)


def damping(k, S, admittance_bands):
    """delta_n = (c / 2) sum_m beta_m(f_n) S_nm."""
    f = k * C / (2 * math.pi)
    return 0.5 * C * (admittance_at(admittance_bands, f) * S).sum(1)


def schroeder_frequency(k, delta, f_grid=None):
    """Lowest frequency at which the modal overlap reaches three: modes per Hz times their mean half-power
    bandwidth delta / pi, both over the window f +- f (sqrt2 - 1/sqrt2) / 2 (an octave's width, centred in
    linear frequency, so a density growing as f^2 is overestimated by only 4%)."""
    f = k * C / (2 * math.pi)
    half = 0.5 * (math.sqrt(2) - 1 / math.sqrt(2))
    if f_grid is None:                              # from the lowest oscillating mode up (the constant mode is the
        f_grid = np.geomspace(f[k * C > delta].min(), f.max() / (1 + half), 400)   # one overdamped mode)
    for fc in f_grid:
        sel = (f >= fc * (1 - half)) & (f < fc * (1 + half))
        if sel.sum() < 2:
            continue
        overlap = sel.sum() / (2 * half * fc) * np.mean(delta[sel] / math.pi)
        if overlap >= SCHROEDER_OVERLAP:
            return float(fc)
    return None


def crossover(f, fs):
    """Modal weight W_m (amplitude): 1 below fs/sqrt2, 0 above fs*sqrt2, cos^2(pi/2 x) power between."""
    x = np.clip(np.log2(np.maximum(f, 1e-30) / fs) + 0.5, 0, 1)
    return np.cos(0.5 * math.pi * x)


def path_share(fs, bands=mc.BANDS):
    """g_b: the share of band b's filter energy left to the path method's incoherent part, W_g^2 = 1 - W_m^2."""
    out = []
    for fc in bands:
        x = np.linspace(-1, 1, 20001)
        f = fc * 2.0 ** x
        g2 = np.cos(0.5 * math.pi * x) ** 2 * f       # |G_b|^2 df, df = f ln2 dx
        out.append(float(np.trapezoid(g2 * (1 - crossover(f, fs) ** 2), x) / np.trapezoid(g2, x)))
    return np.array(out)


def wall_matrices(bank, directory=None):
    """G[m, n, l] = integral over material m's faces of psi_n psi_l (the diagonal is S_nm), cached as walls.npy."""
    path = Path(directory) / 'walls.npy' if directory else None
    if path is not None and path.exists():
        return np.load(path)
    edges = [bank['edges0'], bank['edges1'], bank['edges2']]
    _, _, _, (fc, fa, fm) = operator(edges, bank['solid'])
    psi = np.asarray(bank['psi'][fc], np.float64)
    M = len(bank['materials'])
    G = np.zeros((M, psi.shape[1], psi.shape[1]), np.float32)
    for m in range(M):
        sel = fm == m
        if sel.any():
            G[m] = (psi[sel].T * fa[sel]) @ psi[sel]
    if path is not None:
        np.save(path, G)
    return G


def coupled(k, G, admittance_bands):
    """The walls' admittance projected on the rigid modes in full, solved exactly in that basis.

    The impedance condition dp/dn = -(beta/c) dp/dt, projected on the rigid modes (Galerkin), gives
    q'' + c C q' + Omega^2 q = forcing with C_nl = sum_m beta_m G_mnl; its diagonal is first-order damping
    (delta_n = c C_nn / 2), its off-diagonal terms couple modes that overlap, which first-order theory
    misses where modes overlap or a wall's admittance is not small. beta_m is taken at the pair's geometric
    mean frequency (only near pairs couple; exact for a frequency-independent wall). The state-space form
    x' = A x, x = (q, q'), A = [[0, I], [-Omega^2, -c C]], is diagonalised once: returns its eigenvalues
    (2N, complex), V's upper half U (N x 2N) and V^-1's right half W (2N x N).
    """
    f = k * C / (2 * math.pi)
    N = len(k)
    Cm = np.zeros((N, N))
    fm = np.sqrt(np.outer(np.maximum(f, 1e-3), np.maximum(f, 1e-3)))
    beta = admittance_at(admittance_bands, fm.ravel()).reshape(N, N, -1)
    for m in range(G.shape[0]):
        Cm += beta[:, :, m] * G[m]
    A = np.zeros((2 * N, 2 * N))
    A[:N, N:] = np.eye(N)
    A[N:, :N] = -np.diag((k * C) ** 2)
    A[N:, N:] = -C * Cm
    lam, V = np.linalg.eig(A)
    W = np.linalg.inv(V)
    return lam, V[:N], W[:, N:]


def transfer_coupled(system, a_src, a_rcv, freqs, chunk=1024):
    """H(w) = 4 pi c^2 sum_j (psi(r) . U_j) (W_j . psi(s)) / (i w - lambda_j), the coupled modes' transfer."""
    lam, U, W = system
    u = a_rcv @ U                                   # [R, 2N]
    v = W @ a_src                                   # [2N]
    A = 4 * math.pi * C ** 2 * u * v[None, :]
    w = 2 * math.pi * np.asarray(freqs, float)
    H = np.zeros((A.shape[0], len(w)), complex)
    for a in range(0, len(w), chunk):
        H[:, a:a + chunk] = A @ (1.0 / (1j * w[None, a:a + chunk] - lam[:, None]))
    return H


def transfer(k, delta, a_src, a_rcv, freqs, chunk=1024):
    """Modal transfer functions [R, F]: sum_n 4 pi c^2 psi_n(s) psi_n(r) / (w_n^2 - w^2 + 2 i delta_n w)."""
    A = 4 * math.pi * C ** 2 * a_src[None, :] * a_rcv          # [R, N]
    wn2 = (k * C) ** 2
    w = 2 * math.pi * np.maximum(np.asarray(freqs, float), 1e-6)
    H = np.zeros((A.shape[0], len(w)), complex)
    for a in range(0, len(w), chunk):
        ww = w[a:a + chunk]
        H[:, a:a + chunk] = A @ (1.0 / (wn2[:, None] - ww[None, :] ** 2 + 2j * delta[:, None] * ww[None, :]))
    return H


# --------------------------------------------------------------------------- suite

CLOSED = ['shoebox-0020-presets', 'shoebox-0010-presets', 'furnished-0005-presets']
DF = 1 / 16.0           # spectral grid: a 16 s period, past 200 dB of the slowest decay in the suite


def scene_modes(scene, bank=None, fmax=None, analytic=False):
    """(k, S, psi at the source [N], psi at the receivers [R, N]) from a bank or the analytic box."""
    pts = np.asarray(scene['points'], float).reshape(-1, 3)
    src = np.asarray(scene['source'], float)
    if analytic:
        boxes, names = boxes_of(scene)
        p = scene['parameters']
        L = [p['length'], p['height'], p['depth']]
        faces = []
        for a, (lo_face, hi_face) in enumerate([(0.0, L[0]), (0.0, L[1]), (0.0, L[2])]):
            for v in (lo_face, hi_face):   # the wall box touching this face of the interior
                hit = [m for lo, hi, m in boxes if abs((hi[a] if v == 0.0 else lo[a]) - v) < 1e-6]
                faces.append(hit[0])
        k, shape, S = box_modes(L, fmax, faces, len(names))
        return k, S, shape(src)[0], shape(pts)
    a = probe(bank, np.vstack([src, pts]))
    return bank['k'], bank['surface'], a[0], a[1:]


def direct_energy(scene, bins=mc.TIME_BINS):
    """The path method's exact direct sound as it sits in its histogram: 1/r^2 in the arrival's 1 ms bin."""
    pts = np.asarray(scene['points'], float).reshape(-1, 3)
    r = np.linalg.norm(pts - np.asarray(scene['source'], float), axis=1)
    vis = np.asarray(scene.get('lineOfSight', np.ones(len(r))), float)
    E = np.zeros((len(r), bins))
    k = np.floor(r / C / mc.BIN_SECONDS).astype(int)
    E[np.arange(len(r)), k] = vis / r ** 2
    return E, r, vis


def combine(scene, k, S, a_src, a_rcv, path_energy, fs=None, walls=None):
    """Modes below the Schroeder frequency + the path method above it, as band energies [R, 8, T]
    (band-limited: the path part is smoothed by each band's kernel as the scorer would). walls: the full
    wall matrices G (wall_matrices) for coupled damping; without them, first-order (diagonal) damping."""
    adm = [scene['materials'][m]['admittance'] for m in scene['materials']]
    delta = damping(k, S, adm)
    system = coupled(k, walls, adm) if walls is not None else None
    if system is not None:                          # the coupled modes' own frequencies and decay rates
        up = system[0].imag > 0
        fs = fs or schroeder_frequency(system[0].imag[up] / C, -system[0].real[up])
    fs = fs or schroeder_frequency(k, delta)
    if fs is None:
        raise ValueError('the bank does not reach the Schroeder frequency; bake to a higher fmax')
    touched = [b for b, fc in enumerate(mc.BANDS) if fc / 2 < fs * math.sqrt(2)]
    top = 2 * mc.BANDS[max(touched)]
    if k.max() * C / (2 * math.pi) < fs * math.sqrt(2) * (1 - 1e-3):
        raise ValueError(f'bank ends at {k.max() * C / (2 * math.pi):.0f} Hz, below the crossover top {fs * math.sqrt(2):.0f} Hz')
    freqs = np.arange(0, top + DF, DF)
    Hm = transfer(k, delta, a_src, a_rcv, freqs) if system is None else transfer_coupled(system, a_src, a_rcv, freqs)
    Edir, r, vis = direct_energy(scene)
    Hd = vis[:, None] * np.exp(-2j * math.pi * freqs[None] * r[:, None] / C) / r[:, None]
    W = crossover(freqs, fs)
    Hc = W[None] * Hm + (1 - W[None]) * Hd
    Hc[:, 0] = 0
    Ec, _ = mc.energy_from_spectrum(Hc, DF, bands=mc.BANDS[touched])
    rest = path_energy - Edir[:, None, :]
    out = mc.smooth(path_energy)
    rest_s = mc.smooth(rest)
    g = path_share(fs)
    for j, b in enumerate(touched):
        out[:, b] = Ec[:, j] + g[b] * rest_s[:, b]
    Hm[:, 0] = 0
    Em, _ = mc.energy_from_spectrum(Hm, DF, bands=mc.BANDS[touched])
    modal = np.full_like(out, np.nan)
    for j, b in enumerate(touched):
        modal[:, b] = Em[:, j]
    return out, modal, dict(schroederHz=fs, crossoverHz=[fs / math.sqrt(2), fs * math.sqrt(2)], pathShare=g.tolist(),
                            modes=int(len(k)), touchedBands=[float(mc.BANDS[b]) for b in touched],
                            minRest=float(np.min(rest)), restScale=float(np.max(path_energy)))


def suite(hmax, scenes=CLOSED, root=mc.DEFAULT_ROOT, work=ROOT / '.context/bass-and-diffraction', budget='r02048',
          crossover_hz=None, couple=False):
    """Bake each closed room, combine with every seed of the path method's runs, write the outputs.
    crossover_hz: centre the crossover there instead of at the Schroeder frequency (or at the Schroeder
    frequency if that is higher: the path method is only valid above it); the outputs are then named path-modal-x<Hz>."""
    work.mkdir(parents=True, exist_ok=True)
    report = {}
    tag = ('' if crossover_hz is None else f'-x{crossover_hz:g}') + ('-coupled' if couple else '')
    method = 'path-modal' + tag
    for sid in scenes:
        scene_path = root / sid / 'scene.json'
        scene = json.loads(scene_path.read_text())
        bank_path = work / (f'bank-{sid}-{hmax:g}' + ('' if crossover_hz is None else f'-x{crossover_hz:g}'))
        if (bank_path / 'bank.npz').exists():
            bank = load_bank(bank_path)
        else:
            boxes, names = boxes_of(scene)
            bank = bake_boxes(boxes, names, hmax, fmax=None if crossover_hz is None else
                              max(math.sqrt(2) * crossover_hz, float(load_bank(work / f'bank-{sid}-{hmax:g}')['fmax'])),
                              absorption=[scene['materials'][m]['absorption'] for m in names],
                              admittance=[scene['materials'][m]['admittance'] for m in names], out=bank_path)
        bake_seconds = float(bank['seconds'])
        k, S, a_s, a_r = scene_modes(scene, bank=bank)
        walls = wall_matrices(bank, bank_path) if couple else None
        code = dict(name='tools/room_modes.py', sha256=mc.sha256_file(Path(__file__)))
        base = dict(sceneId=sid, sceneSha256=mc.sha256_file(scene_path), code=code, validSeconds=2.0,
                    bandLimited=True, wallModel='admittance (first order in the modes)')
        for seed_dir in sorted((root / sid / 'path' / budget).glob('[0-9]*')):
            pmeta, pe, _, _ = mc.read_output(seed_dir)
            t = time.time()
            fx = None
            if crossover_hz is not None:     # never below the Schroeder frequency
                fs = schroeder_frequency(k, damping(k, S, [scene['materials'][m]['admittance'] for m in scene['materials']]))
                fx = max(crossover_hz, fs or 0)
            out, modal, info = combine(scene, k, S, a_s, a_r, pe.astype(float), fs=fx, walls=walls)
            seconds = time.time() - t
            mc.write_output(root / sid / method / budget / seed_dir.name,
                            dict(base, method=method, budget=budget, seed=int(seed_dir.name),
                                 budgetValue=pmeta.get('budgetValue'), wallSeconds=pmeta['wallSeconds'] + seconds,
                                 modes=info), out)
        mc.write_output(root / sid / ('modal' + tag) / 'bank' / '0',
                        dict(base, method='modal' + tag, budget='bank', seed=0, budgetValue=1, wallSeconds=bake_seconds), modal)
        report[sid] = dict(info, bakeSeconds=bake_seconds, cells=int(len(bank['volumes'])), hmax=hmax,
                           bankModes=int(len(bank['k'])), bankTopHz=float(bank['k'].max() * C / (2 * math.pi)))
        print(sid, json.dumps({kk: v for kk, v in report[sid].items() if kk != 'pathShare'}), flush=True)
    return report


EXPORT_MAGIC = b'RMB1'


def export(bank, path):
    """Write the bank for the renderer, little-endian:
    'RMB1', int32 version 1; int32 nx, ny, nz; float32 edges x (nx+1), y (ny+1), z (nz+1);
    int16 solid [nx, ny, nz] (C order, -1 air, else material index); int32 M, then M names (int32 byte
    length + UTF-8); int32 N; float32 k [N] (rad/m, fourth-order corrected); float32 S [N, M] (integral of
    psi^2 over each material's faces, sum(M psi^2) = 1); int32 n air cells (C order of solid < 0);
    float16 psi [N, n]."""
    edges = [np.asarray(bank[f'edges{a}'], np.float32) for a in range(3)]
    solid = np.asarray(bank['solid'], np.int16)
    names = [str(x) for x in bank['materials']]
    psi = bank['psi']                                   # [cells, modes]
    with open(path, 'wb') as f:
        f.write(EXPORT_MAGIC)
        f.write(np.array([1, *solid.shape], '<i4').tobytes())
        for e in edges:
            f.write(e.astype('<f4').tobytes())
        f.write(solid.astype('<i2').tobytes())
        f.write(np.array([len(names)], '<i4').tobytes())
        for n in names:
            b = n.encode()
            f.write(np.array([len(b)], '<i4').tobytes() + b)
        f.write(np.array([len(bank['k'])], '<i4').tobytes())
        f.write(np.asarray(bank['k'], '<f4').tobytes())
        f.write(np.asarray(bank['surface'], '<f4').tobytes())
        f.write(np.array([psi.shape[0]], '<i4').tobytes())
        for a in range(0, psi.shape[1], 64):            # mode-major, in blocks
            f.write(np.ascontiguousarray(np.asarray(psi[:, a:a + 64]).T).astype('<f2').tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['suite', 'export'])
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--hmax', type=float, default=0.125)
    parser.add_argument('--scenes', default=','.join(CLOSED))
    parser.add_argument('--crossover', type=float, help='centre the crossover here (Hz, not below the Schroeder frequency)')
    parser.add_argument('--coupled', action='store_true', help='the walls projected on the modes in full (coupled damping)')
    args = parser.parse_args()
    if args.command == 'export':
        export(load_bank(args.paths[0]), args.paths[1])
        return
    report = suite(args.hmax, args.scenes.split(','), crossover_hz=args.crossover, couple=args.coupled)
    out = ROOT / '.context/bass-and-diffraction' / ('suite' + ('' if args.crossover is None else f'-x{args.crossover:g}') +
                                                    ('-coupled' if args.coupled else '') + '.json')
    out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
