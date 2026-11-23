"""Track B, step 1: the normalisation of diffraction as a walk event, checked by Monte Carlo against
UTD's stationary point and BTM on the half-plane fixture of TakeChecks.MethodsMatchTheHalfPlaneReference.

The estimator (README part B.1):
  * a ray of flux Phi whose line passes the edge at offset rho, angle gamma, deposits on the edge the
    line density Phi K1(rho) / sin(gamma) at its closest point (K1 the beam kernel's 1-D marginal);
    summed over rays this is the incident intensity per unit edge length;
  * a diffracted ray leaves that point on Keller's cone (incidence from the chain's origin, exact),
    azimuth uniform over the exterior angle, carrying Phi K1 / sin(gamma) * angle (no D yet);
  * at the listener the beam kernel K(rho_L) of its first segment estimates the diffracted ray tube's
    spreading, s' / (sin^2(beta) s (s + s')), and the score is multiplied by |D|^2 sin^2(beta)
    evaluated at the listener's own azimuth and distance (D pointwise, not smoothed over the beam).
Variants: 'ray' evaluates D at the ray's own azimuth (part 24's prototype), 'open' deposits only the
rays that pass on the open side (part 24's blocked half).

    PY=.context/fdtd-teacher/venv/bin/python
    $PY $B/walk_edge.py halfplane [rays]
    $PY $B/walk_edge.py wall WIDTH [rays]      # double diffraction over a rigid wall: grazing at the second edge
"""
import math
import sys

import numpy as np

C = 343.0
BANDS = 125.0 * 2.0 ** np.arange(8)
BTM_WALL = {2.0: [1.620450e-3, 7.792134e-4, 3.207570e-4, 1.120553e-4, 3.379192e-5],
            0.25: [2.606818e-3, 1.510434e-3, 7.953154e-4, 3.818016e-4, 1.673416e-4]}
BTM_HALFPLANE = {125: 0.009199, 250: 0.006599, 500: 0.004427, 1000: 0.002758, 2000: 0.001590}


def transition(X):
    """UTD transition function, the shaders' rational approximation (MethodEnergy.hlsl UtdTransition)."""
    X = np.maximum(X, 0)
    z = np.sqrt(2 * X / np.pi)
    f = (1 + .926 * z) / (2 + 1.792 * z + 3.104 * z * z)
    g = 1 / (2 + 4.142 * z + 3.492 * z * z + 6.670 * z ** 3)
    s = np.sqrt(2 * np.pi * X)
    return s * f, s * g


def utd_energy(n, phi_in, phi_out, sin_beta, k, distance):
    """|D|^2 of a rigid wedge of exterior angle n pi (Kouyoumjian and Pathak; MethodEnergy.hlsl UtdEnergy)."""
    phi_in, phi_out = np.broadcast_arrays(np.asarray(phi_in, float), np.asarray(phi_out, float))
    re = np.zeros(phi_out.shape)
    im = np.zeros(phi_out.shape)
    for i in range(4):
        sign = 1 if i % 2 == 0 else -1
        x = (phi_out - phi_in) if i < 2 else (phi_out + phi_in)
        arg = (np.pi + sign * x) / (2 * n)
        small = np.abs(np.sin(arg)) < 1e-5
        x = np.where(small, x + sign * 4e-5 * n, x)
        arg = (np.pi + sign * x) / (2 * n)
        N = np.round((x + sign * np.pi) / (2 * np.pi * n))
        c = np.cos(n * np.pi * N - .5 * x)
        F = transition(k * distance * 2 * c * c)
        cot = np.cos(arg) / np.sin(arg)
        re += cot * F[0]
        im += cot * F[1]
    return (re * re + im * im) / (8 * np.pi * k * n * n * sin_beta ** 2)


def azimuth(v, face, normal):
    a = np.arctan2(v @ normal, v @ face)
    return np.where(a < 0, a + 2 * np.pi, a)


def halfplane(rays=1 << 20, seed=1):
    """Screen x = 0, y < 0 (edge on the z axis), source (-2, 1, 0), listener (2, -2, 0)."""
    rng = np.random.default_rng(seed)
    S, R = np.array([-2.0, 1.0, 0.0]), np.array([2.0, -2.0, 0.0])
    u, face, normal, exterior = np.array([0, 0, 1.0]), np.array([0, -1.0, 0]), np.array([-1.0, 0, 0]), 2 * np.pi
    n = exterior / np.pi
    # stationary point (Keller) and UTD
    Q = np.array([0, 0, 0.0])
    sp_, s_ = np.linalg.norm(Q - S), np.linalg.norm(R - Q)
    phi_in_q, phi_out_q = azimuth((S - Q)[None], face, normal)[0], azimuth((R - Q)[None], face, normal)[0]
    print(f'half-plane: incidence {math.degrees(phi_in_q):.1f} deg, listener {math.degrees(phi_out_q):.1f} deg, '
          f'shadow boundary {math.degrees(phi_in_q + math.pi):.1f} deg')
    # rays from the source; flux 4 pi / N (intensity 1 at 1 m)
    z = rng.uniform(-1, 1, rays)
    a = rng.uniform(0, 2 * np.pi, rays)
    d = np.stack([np.sqrt(1 - z * z) * np.cos(a), np.sqrt(1 - z * z) * np.sin(a), z], 1)
    flux = 4 * np.pi / rays
    dxy = d[:, :2]
    sin_g = np.linalg.norm(dxy, axis=1)
    t = -(S[:2] @ dxy.T) / np.maximum(sin_g ** 2, 1e-12)
    closest = S + d * t[:, None]
    e = np.stack([np.zeros(rays), np.zeros(rays), closest[:, 2]], 1)
    rho = np.linalg.norm(closest - e, axis=1)
    ahead = t > 0
    # which side of the screen the ray passes: crosses x = 0 at y < 0 before its closest approach -> blocked side
    tx = -S[0] / np.where(np.abs(d[:, 0]) > 1e-12, d[:, 0], 1e-12)
    blocked = (d[:, 0] > 0) & (S[1] + d[:, 1] * tx < 0) & (tx < t)
    incident = e - S
    sprime = np.linalg.norm(incident, axis=1)
    incident /= sprime[:, None]
    cos_b = incident @ u
    sin_b = np.sqrt(1 - cos_b ** 2)
    phi_in = azimuth(-incident, face, normal)
    phi = rng.uniform(0, exterior, rays)
    out = u[None] * cos_b[:, None] + sin_b[:, None] * (face[None] * np.cos(phi)[:, None] + normal[None] * np.sin(phi)[:, None])
    along = np.einsum('ij,ij->i', R - e, out)
    rhoL2 = np.maximum(np.einsum('ij,ij->i', R - e, R - e) - along ** 2, 0)
    toR = R - e
    phi_R = azimuth(toR, face, normal)
    clearance = 2.0                                   # the listener's distance to the screen
    rows = []
    for fc in BANDS[:5]:
        k = 2 * np.pi * fc / C
        lam = C / fc
        w2_edge = lam * np.maximum(t, 1e-9) / np.pi                  # incident beam width at the edge (L from the source)
        K1 = np.sqrt(2 / (np.pi * w2_edge)) * np.exp(-2 * rho ** 2 / w2_edge)
        dep = flux * K1 / np.maximum(sin_g, 1e-9)
        w2 = lam * np.maximum(along, 1e-9) / np.pi
        K = 2 / (np.pi * w2) * np.exp(-2 * rhoL2 / w2) / (1 - np.exp(-2 * clearance ** 2 / w2))
        K = np.where((along > 0) & (rhoL2 < clearance ** 2), K, 0)
        s_l = np.maximum(along, 1e-9)
        L_R = s_l * sprime / (s_l + sprime) * sin_b ** 2
        D_listener = utd_energy(n, phi_in, phi_R, sin_b, k, L_R) * sin_b ** 2
        D_ray = utd_energy(n, phi_in, phi, sin_b, k, L_R) * sin_b ** 2
        base = dep * exterior * K * ahead
        est = {'pointwise D, both sides': np.sum(base * D_listener),
               'D at the ray azimuth': np.sum(base * D_ray),
               'pointwise D, open side only': np.sum(base * D_listener * ~blocked)}
        sb = 1.0
        L_q = s_ * sp_ / (s_ + sp_)
        stat = (1 / sp_ ** 2) * utd_energy(n, phi_in_q, phi_out_q, sb, k, L_q) * sp_ / (s_ * (s_ + sp_))
        # Monte Carlo standard error of the main estimator
        se = np.sqrt(rays) * np.std(base * D_listener)
        rows.append((fc, stat, est, se))
        print(f'{fc:5.0f} Hz: UTD stationary {stat:.5g} (BTM {BTM_HALFPLANE[int(fc)]:.5g}, '
              f'{10 * math.log10(stat / BTM_HALFPLANE[int(fc)]):+.2f} dB); ' +
              '; '.join(f'{name} {10 * math.log10(v / stat):+.2f} dB' for name, v in est.items()) +
              f'; s.e. {se / stat * 100:.1f}%', flush=True)
    return rows


def cone(u, face, normal, incident, phi):
    cos_b = incident @ u
    sin_b = np.sqrt(np.maximum(1 - cos_b ** 2, 0))
    return u[None] * cos_b[:, None] + sin_b[:, None] * (face[None] * np.cos(phi)[:, None] + normal[None] * np.sin(phi)[:, None]), sin_b


def closest(p, d, a, u):
    """Closest approach of lines p + t d and a + s u: t, s, offset rho, sin of their angle."""
    w0 = p - a
    b = d @ u
    sin2 = np.maximum(1 - b * b, 1e-12)
    dd, ee = np.einsum('ij,ij->i', w0, d), w0 @ u
    t = (b * ee - dd) / sin2
    s_ = (ee - b * dd) / sin2
    gap = p + d * t[:, None] - (a + u * s_[:, None])
    return t, s_, np.linalg.norm(gap, axis=1), np.sqrt(sin2)


def wall(width, rays=1 << 21, seed=2):
    """Rigid wall x in [-w/2, w/2], y <= 1, long in z; source (-3, 0, 0), listener (3, 0, 0) (the fixture of
    MethodsReachTheFarSideOfASolid). Edge 1 (-w/2, 1) and edge 2 (w/2, 1) along z, exterior angle 3 pi / 2.
    Edge 2 is reached along the top face: grazing incidence. The deposit there holds only the rays above the
    face, so it is mirrored (the face's image: the field is even about a rigid face), and UTD's D at edge 2
    takes Kouyoumjian and Pathak's grazing factor 1/2 (the incident field already contains its reflection)."""
    rng = np.random.default_rng(seed)
    S, R = np.array([-3.0, 0, 0]), np.array([3.0, 0, 0])
    u = np.array([0, 0, 1.0])
    a1, a2 = np.array([-width / 2, 1, 0]), np.array([width / 2, 1, 0])
    f1, n1 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])     # edge 1: from the top face, over the top, down the left face
    f2, n2 = np.array([-1.0, 0, 0]), np.array([0, 1.0, 0])    # edge 2: from the top face, over the top, down the right face
    ext = 1.5 * np.pi
    n = ext / np.pi
    # stationary-point UTD (in the z = 0 plane, sin beta = 1)
    s1, s12, s2 = np.linalg.norm(a1 - S), width, np.linalg.norm(R - a2)
    pin1, pout1 = azimuth((S - a1)[None], f1, n1)[0], 0.0
    pin2, pout2 = 0.0, azimuth((R - a2)[None], f2, n2)[0]
    z = rng.uniform(-1, 1, rays)
    az = rng.uniform(0, 2 * np.pi, rays)
    d = np.stack([np.sqrt(1 - z * z) * np.cos(az), np.sqrt(1 - z * z) * np.sin(az), z], 1)
    flux = 4 * np.pi / rays
    # edge 1: every ray's line (those striking the left face too: the face contains the edge)
    t1, z1, rho1, sg1 = closest(np.broadcast_to(S, d.shape), d, a1, u)
    e1 = a1 + u * z1[:, None]
    inc1 = e1 - S
    sp1 = np.linalg.norm(inc1, axis=1)
    inc1 /= sp1[:, None]
    phi1_in = azimuth(-inc1, f1, n1)
    phi1 = rng.uniform(0, ext, rays)
    d1, sb1 = cone(u, f1, n1, inc1, phi1)
    # edge 2 from the diffracted ray of edge 1
    t2, z2, rho2, sg2 = closest(e1, d1, a2, u)
    e2 = a2 + u * z2[:, None]
    above = (d1 @ n1) >= 0
    inc2 = e2 - e1
    s12r = np.linalg.norm(inc2, axis=1)
    inc2 /= s12r[:, None]
    phi1_at2 = azimuth(e2 - e1, f1, n1)                       # pointwise: edge 2 as seen from edge 1 (0: on the face)
    phi2_in = azimuth(-inc2, f2, n2)
    phi2 = rng.uniform(0, ext, rays)
    d2, sb2 = cone(u, f2, n2, inc2, phi2)
    along = np.einsum('ij,ij->i', R - e2, d2)
    rhoL2 = np.maximum(np.einsum('ij,ij->i', R - e2, R - e2) - along ** 2, 0)
    phi2_R = azimuth(R - e2, f2, n2)
    clearance = 2.0 - width / 2                               # the listener's distance to the wall's right face
    for b, fc in enumerate(BANDS[:5]):
        k, lam = 2 * np.pi * fc / C, C / fc
        w2a = lam * np.maximum(t1, 1e-9) / np.pi
        dep1 = flux * np.sqrt(2 / (np.pi * w2a)) * np.exp(-2 * rho1 ** 2 / w2a) / sg1 * (t1 > 0)
        L1 = s12r * sp1 / (s12r + sp1) * sb1 ** 2
        D1 = utd_energy(n, phi1_in, phi1_at2, sb1, k, L1) * sb1 ** 2
        w2b = lam * np.maximum(t2, 1e-9) / np.pi
        dep2 = dep1 * ext * D1 * np.sqrt(2 / (np.pi * w2b)) * np.exp(-2 * rho2 ** 2 / w2b) / sg2 * (t2 > 0) * above
        w2 = lam * np.maximum(along, 1e-9) / np.pi
        K = 2 / (np.pi * w2) * np.exp(-2 * rhoL2 / w2) / (1 - np.exp(-2 * clearance ** 2 / w2))
        K = np.where((along > 0) & (rhoL2 < clearance ** 2), K, 0)
        L2 = along * s12r / (along + s12r) * sb2 ** 2
        D2 = utd_energy(n, phi2_in, phi2_R, sb2, k, L2) * sb2 ** 2
        score = dep2 * 2 * ext * K * D2 / 4                   # mirror x2, grazing (D / 2)^2
        naive = dep2 * ext * K * D2                           # neither: the half deposit and the full D
        Ls1, Ls2 = s12 * s1 / (s12 + s1), s2 * s12 / (s2 + s12)
        stat = (utd_energy(n, pin1, pout1, 1.0, k, Ls1) * utd_energy(n, pin2, pout2, 1.0, k, Ls2) / 4 /
                (s1 * s12 * s2 * (s1 + s12 + s2)))
        se = np.sqrt(rays) * np.std(score)
        btm = BTM_WALL[width][b]
        print(f'{width:g} m wall {fc:5.0f} Hz: UTD stationary (grazing 1/2) {stat:.4g}, {10 * math.log10(stat / btm):+.2f} dB vs BTM; '
              f'walk event {10 * math.log10(score.sum() / stat):+.2f} dB vs UTD ({10 * math.log10(score.sum() / btm):+.2f} vs BTM, '
              f's.e. {se / score.sum() * 100:.1f}%); without the grazing rules {10 * math.log10(naive.sum() / btm):+.2f} dB vs BTM', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'wall':
        wall(float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 21)
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == 'halfplane':
        halfplane(int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 20)
    else:
        raise SystemExit(__doc__)
