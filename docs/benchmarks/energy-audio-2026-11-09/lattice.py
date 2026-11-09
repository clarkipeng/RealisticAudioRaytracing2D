"""Incoherent image lattice of a suite shoebox with (a) the fixed random-incidence absorption the
methods use and (b) the ground truth's locally reacting wall |R(theta)|^2, scored like a method.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY docs/benchmarks/energy-audio-2026-11-09/lattice.py shoebox-0020-presets 1.5

Separates the closed-room decay error into: estimator (energy method vs lattice a),
wall law (lattice a vs b) and wave effects (lattice b vs ground truth).
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import method_compare as mc

DATA = ROOT / '.context/method-compare/data'
sid = sys.argv[1] if len(sys.argv) > 1 else 'shoebox-0020-presets'
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
scene = mc.load_scene(DATA, sid)
c = scene['speedOfSound']
walls = scene['walls']
# room interior: inner faces of the six slabs
inner_lo, inner_hi = np.zeros(3), np.zeros(3)
for a in range(3):
    faces = [w for w in walls if w['half'][a] < .2]
    lows = [w['center'][a] + w['half'][a] for w in faces]
    highs = [w['center'][a] - w['half'][a] for w in faces]
    inner_lo[a], inner_hi[a] = min(lows), max(highs)
L = inner_hi - inner_lo
mat = scene['materials'][scene['surfaceMaterials'][0]]
S = np.array(scene['source']) - inner_lo
P = np.array(scene['points']).reshape(-1, 3) - inner_lo
gt = mc.read_output(DATA / sid / 'gt/ref/0')
gbands = [b for b, f in enumerate(mc.BANDS) if f in gt[0]['bandsProvided']]
print(sid, 'room', L, 'source', S)


def axis_images(xs, La, nmax):
    n = np.arange(-nmax, nmax + 1)
    pos = np.concatenate([2 * n * La + xs, 2 * n * La - xs])
    refl = np.concatenate([np.abs(2 * n), np.abs(2 * n - 1)])
    return pos, refl


def lattice(receiver, alpha, zeta, bins):
    """Two histograms [bins]: fixed alpha and |R(theta)|^2, direct-sound units."""
    R = c * bins * mc.BIN_SECONDS
    imgs = [axis_images(S[a], L[a], int(R / (2 * L[a])) + 2) for a in range(3)]
    ha, hb = np.zeros(bins), np.zeros(bins)
    px, rx = imgs[0]
    py, ry = imgs[1]
    pz, rz = imgs[2]
    dy, dz = py - receiver[1], pz - receiver[2]
    DY, DZ = np.meshgrid(dy, dz, indexing='ij')
    RY, RZ = np.meshgrid(ry, rz, indexing='ij')
    for x, nx in zip(px, rx):
        dx = x - receiver[0]
        d2 = dx * dx + DY * DY + DZ * DZ
        d = np.sqrt(d2)
        k = (d / c / mc.BIN_SECONDS).astype(int)
        keep = k < bins
        if not keep.any():
            continue
        dd, kk = d[keep], k[keep]
        n = nx + RY[keep] + RZ[keep]
        ea = (1 - alpha) ** n / dd ** 2
        def r2(cos):
            r = (zeta * cos - 1) / (zeta * cos + 1)
            return r * r
        eb = r2(abs(dx) / dd) ** nx * r2(np.abs(DY[keep]) / dd) ** RY[keep] * r2(np.abs(DZ[keep]) / dd) ** RZ[keep] / dd ** 2
        ha += np.bincount(kk, ea, bins)
        hb += np.bincount(kk, eb, bins)
    return ha, hb


def paris(zeta):
    return 8 / zeta ** 2 * (zeta + 1 - 1 / (1 + zeta) - 2 * np.log(1 + zeta))


bins = int(seconds / mc.BIN_SECONDS)
pick = list(range(0, len(P), max(1, len(P) // 8)))
E_a = np.zeros((len(P), 8, mc.TIME_BINS)); E_b = np.zeros_like(E_a)
for b in gbands:
    alpha = mat['absorption'][b]
    zeta = 1 / mat['admittance'][b]
    print(f'band {int(mc.BANDS[b])}: alpha {alpha:.4f}, zeta {zeta:.1f}, Paris alpha(zeta) {paris(zeta):.4f}', flush=True)
    for r in pick:
        ha, hb = lattice(P[r], alpha, zeta, bins)
        E_a[r, b, :bins] = ha; E_b[r, b, :bins] = hb
(ROOT / '.context/energy-audio').mkdir(parents=True, exist_ok=True)
np.savez_compressed(ROOT / f'.context/energy-audio/lattice-{sid}.npz', a=E_a, b=E_b, pick=pick)


def sub(o):
    return (o[0], o[1][pick], None, None)


meta = dict(gt[0], validSeconds=min(seconds, gt[0]['validSeconds']))
G = (meta, gt[1][pick], None, None)
A = (dict(meta, bandLimited=True), mc.smooth(E_a[pick]), None, None)
B = (dict(meta, bandLimited=True), mc.smooth(E_b[pick]), None, None)
outs = {'lattice-alpha': A, 'lattice-R(theta)': B}
for s in (0, 1):
    # the comparison root's current energy outputs (re-rendered 9 November)
    o = mc.read_output(DATA / sid / 'energy/r02048' / str(s))
    outs[f'energy s{s}'] = (dict(o[0], validSeconds=meta['validSeconds']), o[1][pick], None, None)
for s in (0,):
    o = mc.read_output(DATA / sid / 'hybrid/r02048' / str(s))
    outs[f'hybrid s{s}'] = (dict(o[0], validSeconds=meta['validSeconds']), o[1][pick], None, None)
for name, o in outs.items():
    for ref_name, ref in (('gt', G), ('lattice-alpha', A)):
        if name == ref_name:
            continue
        err = mc.compare_output(None, ref, o)
        line = f'{name:16s} vs {ref_name:13s}'
        for b in gbands:
            line += f' | {int(mc.BANDS[b])}:'
            for k in ('level', 'EDT', 'T20', 'T30'):
                e = err[k][:, b]
                e = e[np.isfinite(e)]
                line += f' {k} {np.mean(e):+.1f}' if len(e) else f' {k} -'
            line += f' etc {np.nanmean(err["etc"][:, b]):.1f}'
        print(line)
