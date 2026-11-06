"""Validation of the FDTD absorbing walls used for the method-comparison ground truth.

Studies (tolerances are declared in TOLERANCES before any run and copied into
the report; docs/benchmarks/method-compare/README.md discusses them):

- reflection: point source above a large impedance plane; the measured
  reflection factor Q = (p_wall - p_free) / p_free(image) against the
  Weyl-van der Pol spherical-wave factor and the plane-wave coefficient R(theta),
  for three absorptions, three frequencies and two grids.
- shoebox: (a) an empty shoebox with uniform walls against the analytic image
  lattice with angle-dependent |R(theta)|^2 (incoherent energy sum);
  (b) the same room with furniture against Eyring and Sabine; both in the
  250 Hz band at two grids (refinement) and the 500 Hz band at the suite grid.
- refine: suite scenes re-solved in the 250 Hz band at half the spacing.

Run through tools/method_compare_suite.py validate <study>.
"""
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import wofz

import fdtd_wave as fw
import method_compare as mc
import method_compare_suite as suite

OUT = suite.VALIDATION
TOLERANCES = dict(
    reflection=dict(maxAbsQError=.02, note='|Q_fdtd - Q_theory| at >= 13 ppw, incidence 0-80 deg; error must fall with refinement'),
    shoeboxImages=dict(levelDb=1.0, edtPercent=10.0, t20Percent=10.0, t30Percent=10.0,
                       note='empty shoebox vs analytic image lattice with |R(theta)|^2, receiver mean'),
    shoeboxDiffuse=dict(t30VsEyringPercent=15.0, note='furnished room T30 within 15% of Eyring (random-incidence alpha)'),
    refinement=dict(levelDb=.5, decayPercent=5.0, clarityDb=.5, onsetMs=.5, etcDb=1.0,
                    note='suite grid vs half spacing, RMS over receivers'),
)
C = 343.0


# --------------------------------------------------------------------------- helpers

def pause_hook():
    import dlss_jobs as dj
    return dj.PowerCheck()


def simulate(walls, admit, source, points, low, high, spacing, duration, peak, work_fraction, pml=16):
    """Run one configuration on an explicit domain; returns (pressure, q, dt, timing)."""
    grid = fw.make_grid(low, high, spacing, pml)
    ids, table = fw.material_ids(admit) if walls else ([], {})
    solid, _ = fw.voxelize(grid, walls, ids) if walls else (np.zeros(grid.shape, np.uint8), 0)
    sim = fw.Simulation(grid, solid, admittance=table)
    out = sim.run(source, points, duration, peak, work_fraction=work_fraction, pause=pause_hook())
    del sim
    fw.torch.mps.empty_cache()
    return out['pressure'], out['source'], out['dt'], out['timing'], grid


def weyl_van_der_pol(zeta, theta, k, r2):
    """Spherical-wave reflection factor of a locally reacting plane, exp(+iwt) convention.

    Q = Rp + (1 - Rp) F(w), F = 1 + i sqrt(pi) w exp(-w^2) erfc(-i w),
    w = sqrt(i k r2 / 2) (cos theta + beta), beta = 1/zeta (stated for exp(-iwt)
    and conjugated here; beta is real so only the conjugation changes).
    """
    beta = 1 / zeta
    c = np.cos(theta)
    rp = (c - beta) / (c + beta)
    w = np.sqrt(1j * k * r2 / 2) * (c + beta)
    F = 1 + 1j * np.sqrt(np.pi) * w * wofz(w)
    return np.conj(rp + (1 - rp) * F), rp


# --------------------------------------------------------------------------- reflection

def reflection(work_fraction=.2):
    OUT.mkdir(parents=True, exist_ok=True)
    hs, r2 = .25, 2.5
    thetas = np.radians(np.arange(0, 81, 10))
    source = np.array([0., hs, 0.])
    image = np.array([0., -hs, 0.])
    rx = image + r2 * np.stack([np.sin(thetas), np.cos(thetas), np.zeros_like(thetas)], 1)
    rx_image = rx * np.array([1, -1, 1])
    low, high = np.array([-.6, -2.9, -.6]), np.array([3.1, 2.9, .6])
    slab = [dict(center=[0, -2.5, 0], half=[20, 2.5, 20])]          # y < 0, through the PML
    hz = np.array([250., 500., 1000.])
    rows = []
    cases = [(.05, 1000.), (.025, 1000.), (.0125, 1000.)]
    free = {}
    for spacing, peak in cases:
        p, q, dt, timing, grid = simulate([], [], source, np.vstack([rx, rx_image]), low, high, spacing, .03, peak,
                                          work_fraction)
        free[spacing] = fw.transfer(p, q, dt, hz)                    # [F, 2N]
        for alpha in (.1, .3, .6):
            y = fw.admittance_from_absorption(alpha)
            zeta = 1 / y
            p, q, dt, timing, grid = simulate(slab, [y], source, rx, low, high, spacing, .03, peak, work_fraction)
            H = fw.transfer(p, q, dt, hz)
            N = len(rx)
            for fi, f in enumerate(hz):
                k = 2 * np.pi * f / C
                measured = (H[fi] - free[spacing][fi, :N]) / free[spacing][fi, N:]
                theory, rp = weyl_van_der_pol(zeta, thetas, k, r2)
                for ti, th in enumerate(thetas):
                    rows.append(dict(spacing=spacing, alpha=alpha, zeta=zeta, hz=float(f), ppw=C / f / spacing,
                                     theta=float(np.degrees(th)), measured=[measured[ti].real, measured[ti].imag],
                                     theory=[theory[ti].real, theory[ti].imag], planeWave=float(rp[ti]),
                                     error=float(abs(measured[ti] - theory[ti]))))
            print(f'reflection h={spacing} alpha={alpha}: max |Q error| '
                  f'{max(r["error"] for r in rows if r["spacing"] == spacing and r["alpha"] == alpha):.4f}', flush=True)
    (OUT / 'reflection.json').write_text(json.dumps(dict(tolerance=TOLERANCES['reflection'], sourceHeight=hs,
                                                          imageDistance=r2, rows=rows), indent=1) + '\n')
    return rows


# --------------------------------------------------------------------------- shoebox

ROOM = np.array([5.0, 3.5, 4.0])
FURNITURE = [((.625, 0, .5), (1.875, .75, 1.3125)), ((3.25, 0, 2.625), (4.375, 2.0, 3.125)),
             ((.375, 0, 3.0), (1.0, 1.625, 3.75)), ((2.375, 0, .3125), (2.5, 1.875, 1.875)),
             ((3.625, 0, .625), (4.625, 1.125, 1.375)), ((1.75, 0, 2.25), (2.625, .8125, 3.1875))]
SB_SOURCE = np.array([1.3125, 1.5, 1.9375])
SB_RECEIVERS = np.array([[3.6875, 1.25, 1.8125], [2.3125, 2.0, 2.4375], [4.1875, 1.75, 3.5625], [3.0, 2.6, .8]])


def room_walls(furnished):
    t = .25
    L = ROOM
    walls = [dict(center=[L[0] / 2, -t / 2, L[2] / 2], half=[L[0] / 2 + t, t / 2, L[2] / 2 + t]),
             dict(center=[L[0] / 2, L[1] + t / 2, L[2] / 2], half=[L[0] / 2 + t, t / 2, L[2] / 2 + t]),
             dict(center=[-t / 2, L[1] / 2, L[2] / 2], half=[t / 2, L[1] / 2, L[2] / 2 + t]),
             dict(center=[L[0] + t / 2, L[1] / 2, L[2] / 2], half=[t / 2, L[1] / 2, L[2] / 2 + t]),
             dict(center=[L[0] / 2, L[1] / 2, -t / 2], half=[L[0] / 2, L[1] / 2, t / 2]),
             dict(center=[L[0] / 2, L[1] / 2, L[2] + t / 2], half=[L[0] / 2, L[1] / 2, t / 2])]
    if furnished:
        for lo, hi in FURNITURE:
            lo, hi = np.array(lo), np.array(hi)
            walls.append(dict(center=((lo + hi) / 2).tolist(), half=((hi - lo) / 2).tolist()))
    return walls


def image_energy(alpha_st, receiver, duration, fc, bins, spherical=False):
    """Analytic image-lattice band energy [bins] for the empty uniform shoebox.

    Each image's energy is prod_walls |R(theta_axis)|^(2 n_axis) / d^2 with
    theta_axis the incidence angle on that wall pair (the same for every
    reflection on it), binned at d / c and smoothed with the band's energy
    kernel so it has the band filter's time resolution. With `spherical`, R is
    replaced by the Weyl-van der Pol factor Q(theta, k, d) at the image distance
    (Suh & Nelson's spherical-wave image model), averaged as |Q|^2 over the band
    with the band filter's weights.
    """
    zeta = 1 / fw.admittance_from_absorption(alpha_st)
    if spherical:
        hz = fc * 2.0 ** np.linspace(-1, 1, 9)
        weight = mc.band_power(hz, fc) * hz          # |G|^2 df on a log grid
        weight /= weight.sum()

        def gain(cos, n, d):
            th = np.arccos(np.minimum(cos, 1))
            out = 0
            for f, w in zip(hz, weight):
                if w > 0:
                    q, _ = weyl_van_der_pol(zeta, th, 2 * np.pi * f / C, d)
                    out = out + w * np.abs(q) ** (2 * n)
            return out
    else:
        def gain(cos, n, d):
            return np.abs(fw.reflection_coefficient(zeta, np.arccos(np.minimum(cos, 1)))) ** (2 * n)
    L = ROOM
    s, r = SB_SOURCE, receiver
    nmax = np.ceil(duration * C / (2 * L)).astype(int) + 1
    e = np.zeros(bins)
    axes = [np.arange(-n, n + 1) for n in nmax]
    for ix in axes[0]:
        for px in (0, 1):
            x = 2 * ix * L[0] + (1 - 2 * px) * s[0] - r[0]
            nx = abs(2 * ix - px)
            j, pj = np.meshgrid(axes[1], (0, 1), indexing='ij')
            y = 2 * j * L[1] + (1 - 2 * pj) * s[1] - r[1]
            ny = np.abs(2 * j - pj)
            k, pk = np.meshgrid(axes[2], (0, 1), indexing='ij')
            z = 2 * k * L[2] + (1 - 2 * pk) * s[2] - r[2]
            nz = np.abs(2 * k - pk)
            Y, Z = y.ravel()[:, None], z.ravel()[None, :]
            NY, NZ = ny.ravel()[:, None], nz.ravel()[None, :]
            d = np.sqrt(x * x + Y * Y + Z * Z)
            keep = d < duration * C
            g = gain(abs(x) / d, nx, d) * gain(np.abs(Y) / d, NY, d) * gain(np.abs(Z) / d, NZ, d) / d ** 2
            kbin = np.floor(d[keep] / C / mc.BIN_SECONDS).astype(int)
            e += np.bincount(kbin, weights=g[keep], minlength=bins)[:bins]
    return mc.smooth(e[None, None], [fc])[0, 0]


def modal_edc(alpha, receiver, fc, bins):
    """Schroeder curve (dB) of the empty shoebox from its modes (added after the image check, see README).

    Rigid-room modes cos(n pi x / L) perturbed by a small uniform admittance
    beta decay at 2 delta_n = c beta sum_axis (4 or 2 if n_axis = 0) / L_axis
    (Morse & Ingard): oblique modes at the Sabine rate of alpha = 8 beta,
    tangential and axial modes slower. Modes add incoherently with weight
    |G_b(f_n)|^2 psi_n(s)^2 psi_n(r)^2 / (Lambda_n^2 f_n^2); cross terms are dropped.
    """
    beta = fw.admittance_from_absorption(alpha)
    L = ROOM
    nmax = np.ceil(2 * fc * 2 * L / C).astype(int) + 1
    n = np.stack(np.meshgrid(*[np.arange(k + 1) for k in nmax], indexing='ij'), -1).reshape(-1, 3)
    n = n[n.sum(1) > 0]
    f = C / 2 * np.sqrt(((n / L) ** 2).sum(1))
    g = mc.band_power(f, fc)
    keep = g > 0
    n, f, g = n[keep], f[keep], g[keep]
    psi = lambda p: np.prod(np.cos(np.pi * n * np.asarray(p) / L), 1)
    lam = np.prod(np.where(n > 0, .5, 1.), 1)
    two_delta = C * beta * (np.where(n > 0, 4., 2.) / L).sum(1)
    a2 = g * psi(SB_SOURCE) ** 2 * psi(receiver) ** 2 / lam ** 2 / f ** 2
    t = np.arange(bins) * mc.BIN_SECONDS
    edc = (a2[None] * np.exp(-two_delta[None] * t[:, None]) / two_delta[None]).sum(1)
    return 10 * np.log10(edc / edc[0])


def diffuse_predictions(alpha, furnished):
    L = ROOM
    V = np.prod(L)
    S = 2 * (L[0] * L[1] + L[1] * L[2] + L[0] * L[2])
    if furnished:
        for lo, hi in FURNITURE:
            d = np.array(hi) - np.array(lo)
            V -= np.prod(d)
            # Exposed box area: all faces but the one on the floor, minus the floor area it covers.
            S += 2 * (d[0] * d[1] + d[1] * d[2] + d[0] * d[2]) - 2 * d[0] * d[2]
    sabine = 24 * math.log(10) * V / (C * S * alpha)
    eyring = 24 * math.log(10) * V / (-C * S * math.log(1 - alpha))
    return dict(volume=V, surface=S, sabine=sabine, eyring=eyring)


def shoebox_case(alpha, furnished, fc, spacing, duration, work_fraction):
    walls = room_walls(furnished)
    y = fw.admittance_from_absorption(alpha)
    low, high = np.array([-.25, -.25, -.25]), ROOM + .25
    p, q, dt, timing, grid = simulate(walls, [y] * len(walls), SB_SOURCE, SB_RECEIVERS, low, high, spacing,
                                      duration, 1.2 * fc, work_fraction, pml=0)
    H, df = suite.fdtd_transfer_spectrum(p, q, dt)
    e, _ = mc.energy_from_spectrum(H, df, [fc])
    vb = int((duration - suite.VALID_GUARD) / mc.BIN_SECONDS)
    return e[:, 0], vb, timing


def shoebox(work_fraction=.2):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    duration = 1.25
    cases = [(250., .0625), (250., .03125), (500., .03125)]
    for alpha in (.1, .3):
        for furnished in (False, True):
            for fc, spacing in cases:
                tag = f'shoebox-a{alpha:g}-{"furnished" if furnished else "empty"}-{int(fc)}-h{spacing*1e4:.0f}'
                path = OUT / f'{tag}.npy'
                if path.exists():
                    e = np.load(path)
                    vb = int((duration - suite.VALID_GUARD) / mc.BIN_SECONDS)
                    timing = json.loads((OUT / f'{tag}.json').read_text())
                else:
                    e, vb, timing = shoebox_case(alpha, furnished, fc, spacing, duration, work_fraction)
                    np.save(path, e)
                    (OUT / f'{tag}.json').write_text(json.dumps(timing, default=mc._json))
                row = dict(tag=tag, alpha=alpha, furnished=furnished, band=fc, spacing=spacing,
                           gpuSeconds=timing['gpuSeconds'], receivers=[])
                for r in range(len(SB_RECEIVERS)):
                    t0 = mc.onset_bin(e[r, :vb])
                    m = mc.receiver_metrics(e[r], vb, t0)
                    rec = dict(fdtd=m)
                    if not furnished:
                        ref = image_energy(alpha, SB_RECEIVERS[r], duration, fc, mc.TIME_BINS)
                        rec['images'] = mc.receiver_metrics(ref, vb, mc.onset_bin(ref[:vb]))
                        rec['imageCurve'] = mc.schroeder(ref[:vb])[::10].tolist()
                        sph = image_energy(alpha, SB_RECEIVERS[r], duration, fc, mc.TIME_BINS, spherical=True)
                        rec['imagesSpherical'] = mc.receiver_metrics(sph, vb, mc.onset_bin(sph[:vb]))
                        modal = modal_edc(alpha, SB_RECEIVERS[r], fc, vb)
                        rec['modal'] = {k: mc.decay_time(modal, *mc.DECAY_FITS[k]) for k in mc.DECAY_FITS}
                        rec['modalCurve'] = modal[::10].tolist()
                    rec['curve'] = mc.schroeder(e[r, :vb])[::10].tolist()
                    row['receivers'].append(rec)
                row['diffuse'] = diffuse_predictions(alpha, furnished)
                rows.append(row)
                print(tag, 'T30', [round(x['fdtd']['T30'], 3) for x in row['receivers']],
                      'images' if not furnished else 'eyring',
                      [round(x['images']['T30'], 3) for x in row['receivers']] if not furnished else
                      round(row['diffuse']['eyring'], 3), flush=True)
    (OUT / 'shoebox.json').write_text(json.dumps(dict(tolerance=dict(images=TOLERANCES['shoeboxImages'],
                                                                      diffuse=TOLERANCES['shoeboxDiffuse']),
                                                       room=ROOM.tolist(), furniture=FURNITURE,
                                                       source=SB_SOURCE.tolist(), receivers=SB_RECEIVERS.tolist(),
                                                       rows=rows), indent=1, default=mc._json) + '\n')
    return rows


# --------------------------------------------------------------------------- refinement on suite scenes

REFINE = [('barriers-0020-presets', 250.), ('canopy-0009-presets', 250.), ('furnished-0005-presets', 250.),
          ('doorway-0008-presets', 250.)]


def refine(work_fraction=.2):
    OUT.mkdir(parents=True, exist_ok=True)
    import dlss_jobs as dj
    results = []
    for sid, fc in REFINE:
        scene = json.loads((suite.DATA / sid / 'scene.json').read_text())
        b = int(np.flatnonzero(mc.BANDS == fc)[0])
        s = suite.run_settings(scene, fc)
        fine = s['spacing'] / 2
        path = OUT / f'refine-{sid}-{int(fc)}-h{fine*1e4:.0f}.npy'
        base_info = json.loads((suite.FDTD / sid / f'{int(fc)}.json').read_text())
        base = np.load(suite.FDTD / sid / f'{int(fc)}-energy.npy')
        if not path.exists():
            dj.wait_for_mains()
            out = fw.solve(scene, fine, [fc], s['duration'], s['peak'], pml=s['pml'], margin=s['margin'],
                           work_fraction=work_fraction, pause=pause_hook(),
                           wall_admittance=suite.wall_admittance(scene, b))
            H, df = suite.fdtd_transfer_spectrum(out['pressure'], out['source'], out['meta']['dt'])
            e, _ = mc.energy_from_spectrum(H, df, [fc])
            np.save(path, e[:, 0].astype(np.float32))
            (path.with_suffix('.json')).write_text(json.dumps(out['meta'], default=mc._json))
            fw.torch.mps.empty_cache()
        e = np.load(path)
        vb = int(round(base_info['validSeconds'] / mc.BIN_SECONDS))
        e[:, vb:] = 0
        R = len(e)
        diffs = {k: [] for k in ('level', 'EDT', 'T20', 'T30', 'C50', 'C80', 'onset', 'etc')}
        for r in range(R):
            t0 = mc.onset_bin(base[r, :vb])
            a = mc.receiver_metrics(base[r], vb, t0)
            f_ = mc.receiver_metrics(e[r], vb, t0)
            for k in ('level', 'C50', 'C80'):
                diffs[k].append(a[k] - f_[k])
            for k in ('EDT', 'T20', 'T30'):
                diffs[k].append(100 * (a[k] - f_[k]) / f_[k])
            diffs['onset'].append(a['onset'] - f_['onset'])
            diffs['etc'].append(mc.etc_distance(base[r], e[r], vb))
        rms = {k: float(np.sqrt(np.nanmean(np.square(v)))) if np.isfinite(v).any() else None for k, v in diffs.items()}
        defined = {k: int(np.isfinite(v).sum()) for k, v in diffs.items()}
        results.append(dict(scene=sid, band=fc, spacing=s['spacing'], fine=fine, receivers=R, rms=rms, defined=defined))
        print(sid, fc, {k: (round(v, 3) if v is not None else None) for k, v in rms.items()}, flush=True)
    (OUT / 'refine.json').write_text(json.dumps(dict(tolerance=TOLERANCES['refinement'], rows=results), indent=1) + '\n')
    return results


# --------------------------------------------------------------------------- figures and summary

DOCS = suite.ROOT / 'docs/benchmarks/method-compare'
ALPHA_COLORS = {.1: '#0072B2', .3: '#D55E00', .6: '#009E73'}


def report(work_fraction=None):
    """Validation figures (docs PNGs) and validation.json with pass/fail against TOLERANCES."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = dict(tolerances=TOLERANCES)
    # Reflection factor vs angle.
    ref = json.loads((OUT / 'reflection.json').read_text())
    rows = ref['rows']
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, constrained_layout=True)
    for ax, f in zip(axs, (250., 500., 1000.)):
        for alpha, color in ALPHA_COLORS.items():
            rs = sorted([r for r in rows if r['hz'] == f and r['alpha'] == alpha and r['spacing'] == .025],
                        key=lambda r: r['theta'])
            th = [r['theta'] for r in rs]
            ax.plot(th, [abs(complex(*r['theory'])) for r in rs], '-', color=color, lw=2, label=f'alpha {alpha} theory')
            ax.plot(th, [abs(r['planeWave']) for r in rs], ':', color=color, lw=1.5)
            ax.plot(th, [abs(complex(*r['measured'])) for r in rs], 'o', color=color, ms=6, mfc='white', mew=1.5)
        ax.set_title(f'{f:g} Hz, h = 2.5 cm ({343 / f / .025:.0f} ppw)', fontsize=10)
        ax.set_xlabel('incidence angle (deg)')
        ax.grid(True, color='#e6e6e6', lw=.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel('|Q|')
    axs[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle('Reflection factor of an impedance plane: FDTD (circles), Weyl-van der Pol (solid), '
                 'plane-wave R (dotted)', fontsize=10)
    fig.savefig(DOCS / 'fig-reflection.png', dpi=130)
    plt.close(fig)
    worst = {}
    for h in sorted({r['spacing'] for r in rows}):
        sel = [r for r in rows if r['spacing'] == h and r['ppw'] >= 13]
        worst[f'{h:g}'] = dict(maxError=max(r['error'] for r in sel) if sel else None,
                               maxErrorAll=max(r['error'] for r in rows if r['spacing'] == h))
    tol = TOLERANCES['reflection']['maxAbsQError']
    out['reflection'] = dict(worstByGrid=worst, passed=all(v['maxError'] is None or v['maxError'] <= tol
                                                             for v in worst.values()))
    # Shoebox decay.
    if (OUT / 'shoebox.json').exists():
        sb = json.loads((OUT / 'shoebox.json').read_text())
        table = []
        for row in sb['rows']:
            recs = row['receivers']
            mean = lambda key, src: float(np.nanmean([r[src][key] for r in recs if src in r])) if src in recs[0] else None
            lv = lambda src: 10 * math.log10(np.mean([10 ** (r[src]['level'] / 10) for r in recs])) if src in recs[0] else None
            entry = dict(tag=row['tag'], alpha=row['alpha'], furnished=row['furnished'], band=row['band'],
                         spacing=row['spacing'], gpuSeconds=row['gpuSeconds'],
                         fdtd={k: mean(k, 'fdtd') for k in ('EDT', 'T20', 'T30')}, levelFdtd=lv('fdtd'),
                         eyring=row['diffuse']['eyring'], sabine=row['diffuse']['sabine'])
            if not row['furnished']:
                entry['images'] = {k: mean(k, 'images') for k in ('EDT', 'T20', 'T30')}
                entry['imagesSpherical'] = {k: mean(k, 'imagesSpherical') for k in ('EDT', 'T20', 'T30')}
                entry['modal'] = {k: mean(k, 'modal') for k in ('EDT', 'T20', 'T30')}
                entry['levelImages'] = lv('images')
                entry['perReceiverT30'] = dict(fdtd=[r['fdtd']['T30'] for r in recs],
                                               modal=[r['modal']['T30'] for r in recs],
                                               images=[r['images']['T30'] for r in recs])
            table.append(entry)
        out['shoebox'] = table
        fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True, constrained_layout=True)
        panels = [(a, f, fc) for a in (.1, .3) for f in (False, True) for fc in (500.,)]
        for ax, (alpha, furnished, fc) in zip(axs, panels):
            row = next(r for r in sb['rows'] if r['alpha'] == alpha and r['furnished'] == furnished and r['band'] == fc)
            for k, rec in enumerate(row['receivers']):
                t = np.arange(len(rec['curve'])) * .01
                ax.plot(t, rec['curve'], '-', color='#000000', lw=1.4, label='FDTD' if k == 0 else None)
                if 'modalCurve' in rec:
                    ax.plot(np.arange(len(rec['modalCurve'])) * .01, rec['modalCurve'], '-', color='#009E73', lw=1.2,
                            label='modal sum' if k == 0 else None)
                    ax.plot(np.arange(len(rec['imageCurve'])) * .01, rec['imageCurve'], '--', color='#0072B2', lw=1.2,
                            label='image lattice, plane-wave R' if k == 0 else None)
            for name, T, color in (('Eyring', row['diffuse']['eyring'], '#D55E00'), ('Sabine', row['diffuse']['sabine'], '#CC79A7')):
                ax.plot([0, 1.2], [0, -60 * 1.2 / T], ':', color=color, lw=1.5, label=name)
            ax.set_ylim(-60, 2)
            ax.set_xlim(0, 1.2)
            ax.set_title(f'{"furnished" if furnished else "empty"} 5 x 3.5 x 4 m, alpha {alpha}, {fc:g} Hz', fontsize=9)
            ax.set_xlabel('time after emission (s)')
            ax.grid(True, color='#e6e6e6', lw=.6)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.legend(fontsize=7, frameon=False, loc='lower left')
        axs[0].set_ylabel('Schroeder curve (dB)')
        fig.savefig(DOCS / 'fig-shoebox-decay.png', dpi=130)
        plt.close(fig)
    if (OUT / 'refine.json').exists():
        out['refine'] = json.loads((OUT / 'refine.json').read_text())['rows']
    (DOCS / 'validation.json').write_text(json.dumps(out, indent=1, default=mc._json) + '\n')
    print(json.dumps(out, indent=1, default=mc._json)[:6000])


def main(study, work_fraction=.2):
    dict(reflection=reflection, shoebox=shoebox, refine=refine, report=report)[study](work_fraction)
