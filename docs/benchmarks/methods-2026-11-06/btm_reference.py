"""Independent BTM band energies for the method checks (TakeChecks.MethodsMatchTheHalfPlaneReference
and MethodsReachTheFarSideOfASolid): numerical Biot-Tolstoy-Medwin integrals in float64, band
energies by tools/method_compare.py (the comparison's own band filters). No Unity, no renderer.

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY docs/benchmarks/methods-2026-11-06/btm_reference.py   # about 45 s, 3.9 GB peak (band step)

The edge kernel is the one validated against the EDtoolbox second-order cube field
(-0.1324 - 0.0776i at 125 Hz, the reference of TunedRoutesReachTheFarSideOfASolid).
"""
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
import method_compare as mc  # noqa: E402

C = 343.0


def wedge_angle(d, f, n, exterior, radial):
    a = np.arctan2(d @ n, d @ f)
    a = np.where(a < 0, a + 2 * np.pi, a)
    a = np.where(np.minimum(a, 2 * np.pi - a) * radial <= 1e-4, 0, a)
    return np.where(np.abs(a - exterior) * radial <= 1e-4, exterior, a)


def kernel(src, at, rcv, u, f, n, exterior):
    """BTM K of a rigid wedge (CompleteDiffraction.hlsl TunedEdgeKernel); arrays of points [N, 3]."""
    si, so = src - at, rcv - at
    zs, zr = si @ u, so @ u
    ps, pr = si - np.outer(zs, u), so - np.outer(zr, u)
    rs, rr = np.linalg.norm(ps, axis=-1), np.linalg.norm(pr, axis=-1)
    ts, tr = wedge_angle(ps, f, n, exterior, rs), wedge_angle(pr, f, n, exterior, rr)
    nu, m, l = np.pi / exterior, np.linalg.norm(si, axis=-1), np.linalg.norm(so, axis=-1)
    x = np.abs(rs * zr + rr * zs) / np.sqrt(2 * rs * rr * (m * l + rs * rr - zs * zr))
    sh = np.sinh(nu * np.log(x + np.sqrt(1 + x * x)))
    total = 0
    for t in (ts + tr, ts - tr, tr - ts, -ts - tr):
        phi = nu * (np.pi + t)
        h = np.sin(.5 * phi)
        total = total + np.where(np.abs(h) > 1e-6, np.sin(phi) / (2 * (sh * sh + h * h)), 0)
    return np.where((ts > exterior) | (tr > exterior), 0, -nu * total / (4 * np.pi))


def band_energies(delays, gains, horizon=.3):
    keep = delays < horizon
    e = mc.energy_from_arrivals(delays[keep], np.repeat(gains[keep, None], 8, 1), bins=int(round(horizon / mc.BIN_SECONDS)))
    return np.nansum(e, -1)


def half_plane():
    # KnifeEdgeFixture (3D): sheet x = 0, y < 0; top edge along z through the origin;
    # thin sheet: exterior angle 2 pi, face 0 runs down (-y), air-side normal +x.
    S, L = np.array([-2., 1, 0]), np.array([2., -2, 0])
    u, f, n = np.array([0, 0, 1.]), np.array([0, -1., 0]), np.array([1., 0, 0])
    z = np.concatenate([np.linspace(-100, -2, 20001)[:-1], np.linspace(-2, 2, 400001), np.linspace(2, 100, 20001)[1:]])
    w = np.gradient(z)
    P = np.stack([np.zeros_like(z), np.zeros_like(z), z], 1)
    m, l = np.linalg.norm(P - S, axis=1), np.linalg.norm(L - P, axis=1)
    K = kernel(np.broadcast_to(S, P.shape), P, np.broadcast_to(L, P.shape), u, f, n, 2 * np.pi)
    return band_energies((m + l) / C, K * w / (m * l))


def thick_wall(width, extent=12.0, step=.005):
    """DD over the top of a rigid wall `width` thick (top face y = 1, faces x = +-width/2,
    long in z): edge 1 at (-width/2, 1), edge 2 at (+width/2, 1), both along z.
    Integrated row by row (one z1 at a time) to keep memory small."""
    S, L = np.array([-3., 0, 0]), np.array([3., 0, 0])
    u = np.array([0, 0, 1.])
    f1, n1 = np.array([0, -1., 0]), np.array([-1., 0, 0])   # front face down, air side -x
    f2, n2 = np.array([-1., 0, 0]), np.array([0, 1., 0])    # top face toward -x, air side +y
    z = np.arange(-extent, extent + step / 2, step)
    P2 = np.stack([np.full(z.size, width / 2), np.ones(z.size), z], 1)
    l = np.linalg.norm(L - P2, axis=1)
    delays, gains = [], []
    for z1 in z:
        p1 = np.array([-width / 2, 1, z1])
        m, d = np.linalg.norm(p1 - S), np.linalg.norm(P2 - p1, axis=1)
        K1 = kernel(S[None], p1[None], P2, u, f1, n1, 1.5 * np.pi)
        K2 = kernel(np.broadcast_to(p1, P2.shape), P2, np.broadcast_to(L, P2.shape), u, f2, n2, 1.5 * np.pi)
        delays.append((m + d + l) / C)
        gains.append(.5 * K1 * K2 * step * step / (m * d * l))
    return band_energies(np.concatenate(delays), np.concatenate(gains))


if __name__ == '__main__':
    print('half-plane band energies 125 Hz ... 16 kHz:', np.array2string(half_plane(), precision=6, separator=', '))
    for width in (.25, 2.0):
        print(f'{width} m wall DD band energies 125 Hz ... 16 kHz:', np.array2string(thick_wall(width), precision=6, separator=', '))
