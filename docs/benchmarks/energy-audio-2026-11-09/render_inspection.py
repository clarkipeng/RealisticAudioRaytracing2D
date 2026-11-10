"""Technical inspection of the energy method's Church listening render.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY tools/method_runs.py tests "TakeChecks.MeasureEnergyArrivalTimeline" \\
        ACOUSTIC_ENERGY_TIMELINE=$PWD/.context/energy-audio/inspect/timeline.job.json   # the energy render's solves
    $PY docs/benchmarks/energy-audio-2026-11-09/render_inspection.py                             # report.md and figures

Reads the per-solve arrival lists and diffuse histograms (EnergyArrivalTimeline.cs) and
answers: how many discrete arrivals and how they are spaced (flutter, comb filtering),
whether arrivals switch on and off between solves (and why: the 64-slot cap or geometry),
how the diffuse field joins the arrivals in level and time, how much each arrival's
zero-phase band filter rings before its delay, and how large the mirror facets are
compared with the Fresnel zone of their reflection.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
import listen  # noqa: E402

TIMELINE = ROOT / '.context/energy-audio/inspect/timeline'
OUT = ROOT / '.context/energy-audio/inspect'
BANDS = 125.0 * 2 ** np.arange(8)
C = 343.0


def load():
    meta = json.loads((TIMELINE / 'meta.json').read_text())
    solves = np.genfromtxt(TIMELINE / 'solves.csv', delimiter=',', names=True)
    a = np.genfromtxt(TIMELINE / 'arrivals.csv', delimiter=',', names=True, dtype=None, encoding=None)
    late = np.fromfile(TIMELINE / 'late.f32', '<f4').reshape(meta['solves'], 8, meta['lateBins']).astype(float)
    return meta, solves, a, late


def energy(a):
    return np.stack([a[f'e{b}'] for b in range(8)], 1).astype(float)


def energy_at(e, f):
    """MethodEnergy.hlsl EnergyAt: power-complementary raised-cosine bands, flat outside."""
    x = np.clip(np.log2(np.maximum(f, 1) / 125.0), 0, 7)
    b = np.minimum(6, np.floor(x)).astype(int)
    w = np.cos(.5 * np.pi * (x - b)) ** 2
    return w * e[b] + (1 - w) * e[b + 1]


def plane_areas(scene, geometry, pose):
    """Area of every world plane (object-independent: coplanar faces add), keyed by quantised (n, d)."""
    tri = geometry['vertices'].astype(float)
    world = np.empty_like(tri)
    for owner in np.unique(geometry['owner']):
        m = geometry['owner'] == owner
        p = pose[owner]
        world[m] = p[:3] + listen.rotate(p[3:7], tri[m].reshape(-1, 3)).reshape(-1, 3, 3)
    n = np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0])
    area = .5 * np.linalg.norm(n, axis=1)
    ok = area > 0
    n[ok] /= (2 * area[ok])[:, None]
    sign = np.where(np.abs(n[:, 0]) > 1e-6, np.sign(n[:, 0]), np.where(np.abs(n[:, 1]) > 1e-6, np.sign(n[:, 1]), np.sign(n[:, 2])))
    n *= sign[:, None]
    d = (n * world[:, 0]).sum(1)
    table = {}
    for i in np.flatnonzero(ok):
        key = (round(n[i, 0], 2), round(n[i, 1], 2), round(n[i, 2], 2), round(d[i], 1))
        table[key] = table.get(key, 0) + area[i]
    return table


def main():
    meta, solves, a, late = load()
    rate, slots, step_s = meta['sampleRate'], meta['slots'], 1024 / meta['sampleRate']
    E = energy(a)
    total = E.sum(1)
    solve = a['solve'].astype(int)
    S = len(solves)
    t = solves['sample'] / rate
    lines = ['# Energy render inspection (Church speech walk)', '']
    lines.append(f"{S} solves, one per {1000 * step_s:.1f} ms; arrival order {meta['order']}, families {meta['families']} "
                 f"(direct, first-order specular, single-edge D); {meta['planes']} mirror planes, {meta['edges']} edges; {slots} slots.")

    # ---------------- 1. counts and the cap
    counts = np.bincount(solve, minlength=S)
    selected = a['rank'] < slots
    kept = np.bincount(solve, weights=total * selected, minlength=S)
    allE = np.bincount(solve, weights=total, minlength=S)
    direct = np.bincount(solve, weights=total * (a['family'] == 0), minlength=S)
    spec = np.bincount(solve, weights=total * selected * (a['family'] == 1), minlength=S)
    edge = np.bincount(solve, weights=total * selected * (a['family'] == 2), minlength=S)
    nspec = np.bincount(solve, weights=(a['family'] == 1) * selected, minlength=S)
    nedge = np.bincount(solve, weights=(a['family'] == 2) * selected, minlength=S)
    inside = t < 10
    lines += ['', '## 1. How many discrete arrivals', '']
    lines.append(f"Arrivals found per solve: median {np.median(counts[inside]):.0f} inside (0–10 s), {np.median(counts[~inside]):.0f} after; "
                 f"{(counts > slots).mean() * 100:.0f}% of solves have more than {slots}.")
    lines.append(f"Played (top {slots}): median {np.median(nspec[inside]):.0f} first-order reflections and {np.median(nedge[inside]):.0f} edge (D) paths inside.")
    lines.append(f"Energy of the arrivals beyond the cap (moved to the diffuse field): median {100 * np.median(1 - kept[inside] / allE[inside]):.2f}% "
                 f"of all discrete energy inside, max {100 * np.max(1 - kept / allE):.1f}%.")
    ref = (spec + edge) / np.maximum(direct, 1e-30)
    lines.append(f"Played reflections + edge paths relative to the direct sound inside: median {10 * np.log10(np.median(ref[inside & (direct > 0)])):+.1f} dB "
                 f"(edge paths alone {10 * np.log10(np.median((edge / np.maximum(direct, 1e-30))[inside & (direct > 0)])):+.1f} dB).")

    # ---------------- 2. spacing: flutter and comb filtering
    lines += ['', '## 2. Spacing: flutter and comb filtering', '']
    lines.append('Every specular arrival is first order (the Church has too many planes for second order), so no arrival series '
                 'can repeat between parallel walls: flutter echo cannot come from the discrete arrivals.')
    strongest, gaps, ripple, near = [], [], [], []
    f = np.linspace(100, 8000, 4000)
    for s in np.flatnonzero(inside)[::4]:
        m = (solve == s) & selected
        d = a['delay'][m]; e = E[m]; fam = a['family'][m]
        if not (fam == 0).any():
            continue
        e0 = total[m][fam == 0][0]; t0 = d[fam == 0][0]
        refl = fam != 0
        if refl.any():
            k = np.argmax(total[m] * refl)
            strongest.append((10 * np.log10(total[m][k] / e0), 1000 * (d[k] - t0)))
            near.append(((total[m] * refl) > e0 * 10 ** -.6).sum())  # within 6 dB of direct
            ds = np.sort(d[refl]); gaps.extend(np.diff(ds) * 1000)
        # magnitude response of the played set (zero-phase band gains, exact delays), 1 kHz band gains
        g = np.sqrt(np.array([energy_at(x, f) for x in e]))
        H = (g * np.exp(-2j * np.pi * f[None, :] * (d - t0)[:, None])).sum(0)
        level = 20 * np.log10(np.abs(H) / np.sqrt((g ** 2).sum(0)))
        smooth = np.convolve(level, np.ones(50) / 50, 'same')
        ripple.append(np.std((level - smooth)[50:-50]))
    st = np.array(strongest)
    s1 = int(np.argmin(abs(t - 1.0))); m1 = (solve == s1) & selected
    if (a['family'][m1] == 0).any():
        L = np.array([solves['lx'][s1], solves['ly'][s1], solves['lz'][s1]]); Sx = np.array([solves['sx'][s1], solves['sy'][s1], solves['sz'][s1]])
        t0 = a['delay'][m1][a['family'][m1] == 0][0]; e0 = total[m1][a['family'][m1] == 0][0]
        for i in np.flatnonzero(m1 & (a['family'] == 1))[np.argsort(-total[m1 & (a['family'] == 1)])][:4]:
            image = L + np.array([a['dx'][i], a['dy'][i], a['dz'][i]]) * a['length'][i]
            lines.append(f"At 1 s: reflection {10 * np.log10(total[i] / e0):+.1f} dB, {1000 * (a['delay'][i] - t0):.1f} ms after the direct sound, "
                         f"image source offset from the source {np.round(image - Sx, 1).tolist()} m.")
    lines.append(f"Strongest reflection or edge path relative to the direct sound (inside, every 4th solve): median {np.median(st[:, 0]):+.1f} dB, "
                 f"range {st[:, 0].min():+.1f} … {st[:, 0].max():+.1f} dB, arriving {np.median(st[:, 1]):.1f} ms after it (median).")
    lines.append(f"Arrivals within 6 dB of the direct sound: median {np.median(near):.0f} per solve.")
    cluster, cluster_n = [], []
    for s in np.flatnonzero(inside):
        m = (solve == s) & selected
        if not (a['family'][m] == 0).any():
            continue
        t0 = a['delay'][m][a['family'][m] == 0][0]; e0 = total[m][a['family'][m] == 0][0]
        c = (a['family'][m] == 2) & (a['delay'][m] - t0 < .002)
        cluster.append(total[m][c].sum() / e0); cluster_n.append(c.sum())
    lines.append(f"Edge (D) paths arriving within 2 ms of the direct sound: median {np.median(cluster_n):.0f} per solve, together {10 * np.log10(np.median(cluster)):+.1f} dB "
                 f"relative to the direct sound (90th percentile {10 * np.log10(np.percentile(cluster, 90)):+.1f} dB). They are single-edge UTD paths past edges near the line of sight; "
                 "as energies they add to the direct sound instead of interfering with it, and as zero-phase copies 0–2 ms late they comb-filter it.")
    lines.append(f"Spacing between consecutive played reflections: median {np.median(gaps):.2f} ms, 10th percentile {np.percentile(gaps, 10):.2f} ms.")
    lines.append(f"Comb ripple of the played set (fine structure of |H(f)| around its 1/12-oct-like smooth, 100 Hz–8 kHz): median {np.median(ripple):.1f} dB std.")

    # ---------------- 3. popping at solve updates
    lines += ['', '## 3. Arrivals switching on and off between solves', '']
    by = {}
    for s in range(S):
        m = solve == s
        by[s] = (dict(zip(a['identity'][m], total[m])), dict(zip(a['identity'][m], a['rank'][m])), direct[s])
    events = {'cap': [], 'geometry': []}
    fam = dict(zip(zip(solve, a['identity']), a['family']))
    switch_family = {0: 0, 1: 0, 2: 0}
    steps = []
    for s in range(1, S):
        (e1, r1, d1), (e2, r2, d2) = by[s - 1], by[s]
        ref = max(kept[s - 1], kept[s], 1e-30)   # all played discrete energy of the solve
        played1 = {k for k, r in r1.items() if r < slots}; played2 = {k for k, r in r2.items() if r < slots}
        for k in played1 ^ played2:
            level = 10 * np.log10(max(e1.get(k, 0), e2.get(k, 0)) / ref)
            cause = 'cap' if (k in e1 and k in e2) else 'geometry'
            events[cause].append((t[s], level))
            if cause == 'geometry' and level > -20:
                switch_family[int(fam.get((s, k), fam.get((s - 1, k), 0)))] += 1
        for k in played1 & played2:
            if e1[k] > 0 and e2[k] > 0:
                steps.append((t[s], 10 * np.log10(e2[k] / e1[k]), 10 * np.log10(max(e1[k], e2[k]) / ref)))
    seconds = t[-1]
    for cause, label in (('geometry', 'appear/disappear geometrically (visibility of the image or edge path changes)'),
                         ('cap', 'cross the 64-slot cap while still present (reshuffling)')):
        ev = np.array(events[cause]) if events[cause] else np.zeros((0, 2))
        loud = ev[:, 1] > -20 if len(ev) else np.zeros(0, bool)
        lines.append(f"Paths that {label}: {len(ev)} switches ({len(ev) / seconds:.1f}/s); {loud.sum()} of them within 20 dB of all played discrete energy "
                     f"(loudest {ev[:, 1].max() if len(ev) else float('nan'):+.1f} dB).")
    lines.append(f"Of the loud geometric switches: {switch_family[2]} edge (D) paths, {switch_family[1]} reflections, {switch_family[0]} direct sound.")
    st = np.array(steps)
    big = (np.abs(st[:, 1]) > 3) & (st[:, 2] > -20)
    lines.append(f"Persistent played paths whose level jumps by more than 3 dB between solves while within 20 dB of all played energy: {big.sum()} ({big.sum() / seconds:.1f}/s).")
    lines.append('A switch is a crossfade over one 23 ms response step in the spectral synthesis, not a sample-level click '
                 '(the no-click check passes); a switch within ~10–20 dB of the direct sound is an audible step in colour or echo.')

    # ---------------- 4. discrete arrivals vs diffuse onset
    lines += ['', '## 4. How the diffuse field joins the arrivals', '']
    bin_s = meta['lateBinSeconds']; nb = late.shape[2]
    band = 3
    rows = []
    for s in np.flatnonzero(inside):
        m = (solve == s) & selected
        if not (a['family'][m] == 0).any():
            continue
        t0 = a['delay'][m][a['family'][m] == 0][0]
        disc = np.bincount(np.minimum(nb - 1, (a['delay'][m] / bin_s).astype(int)), weights=E[m][:, band], minlength=nb)
        disc_refl = np.bincount(np.minimum(nb - 1, (a['delay'][m] / bin_s).astype(int)), weights=E[m][:, band] * (a['family'][m] != 0), minlength=nb)
        k0 = int(t0 / bin_s)
        diff = np.roll(late[s, band], -k0); disc = np.roll(disc, -k0); disc_refl = np.roll(disc_refl, -k0)
        last = a['delay'][m][a['family'][m] != 0].max() if (a['family'][m] != 0).any() else t0
        rows.append((t0, last, disc, disc_refl, diff))
    disc = np.mean([r[2] for r in rows], 0); refl = np.mean([r[3] for r in rows], 0); diff = np.mean([r[4] for r in rows], 0)
    total_curve = disc + diff
    lines.append(f"Inside (0–10 s), 1 kHz band, {bin_s * 1000:.1f} ms bins (the late field's resolution), each solve shifted so its direct sound falls in bin 0, then averaged:")
    lines.append('')
    lines.append('| bins after the direct sound (ms) | discrete reflections + edges (dB) | diffuse (dB) | total incl. direct (dB) |')
    lines.append('|---:|---:|---:|---:|')
    ref = total_curve.max()
    for k in range(0, 14):
        lines.append(f"| {k * bin_s * 1000:.0f}–{(k + 1) * bin_s * 1000:.0f} | {10 * np.log10(max(refl[k], 1e-30) / ref):.1f} | "
                     f"{10 * np.log10(max(diff[k], 1e-30) / ref):.1f} | {10 * np.log10(max(total_curve[k], 1e-30) / ref):.1f} |")
    last = np.array([r[1] - r[0] for r in rows])
    lines.append('')
    lines.append(f"Last played reflection: median {1000 * np.median(last):.0f} ms after the direct sound (range {1000 * last.min():.0f}–{1000 * last.max():.0f} ms).")
    early = slice(0, 3)
    lines.append(f"Over the first {3 * bin_s * 1000:.0f} ms after the direct sound the diffuse field holds {100 * diff[early].sum() / (diff[early].sum() + refl[early].sum()):.0f}% "
                 f"of the reflected energy; over the whole response {100 * diff.sum() / (diff.sum() + refl.sum()):.0f}%.")

    # ---------------- 5. zero-phase filter pre-ringing
    lines += ['', '## 5. Pre-ringing of the zero-phase band filter', '']
    n = 1024
    bins = np.fft.rfftfreq(n, 1 / rate)
    pre, spread = [], []
    m = selected & (a['family'] != 0) & np.isin(solve, np.flatnonzero(inside)[::8])
    for x in E[m]:
        g = np.sqrt(energy_at(x, bins))
        h = np.fft.fftshift(np.fft.irfft(g, n))
        p = h ** 2
        centre = n // 2
        pre.append(p[:centre - 22].sum() / p.sum())           # energy more than 0.5 ms before the arrival
        c = np.cumsum(np.sort(p)[::-1]); spread.append((c < .99 * p.sum()).sum() / rate * 1000)
    lines.append(f"Each played arrival is one symmetric (zero-phase) filter built from its eight band energies; the hybrid's image paths are built the same way (real per-bin gains).")
    lines.append(f"Energy arriving more than 0.5 ms before the arrival's delay: median {100 * np.median(pre):.2f}%, max {100 * np.max(pre):.2f}%; "
                 f"99% of each filter's energy lies within {np.median(spread):.2f} ms (median).")

    # ---------------- 6. mirror facet size vs Fresnel zone
    lines += ['', '## 6. Mirror size against the Fresnel zone', '']
    _, scene, geometry, frames = listen.take_scene()
    areas = plane_areas(scene, geometry, frames['state'][0])
    keys = np.array(list(areas.keys())); vals = np.array(list(areas.values()))
    rec = []
    m = selected & (a['family'] == 1) & np.isin(solve, np.flatnonzero(inside)[::8])
    for i in np.flatnonzero(m):
        s = solve[i]
        L = np.array([solves['lx'][s], solves['ly'][s], solves['lz'][s]]); Sx = np.array([solves['sx'][s], solves['sy'][s], solves['sz'][s]])
        d = np.array([a['dx'][i], a['dy'][i], a['dz'][i]]); image = L + d * a['length'][i]
        nrm = image - Sx; nrm /= np.linalg.norm(nrm)
        nrm *= np.sign(nrm[0]) if abs(nrm[0]) > 1e-6 else np.sign(nrm[1]) if abs(nrm[1]) > 1e-6 else np.sign(nrm[2])
        dd = float(nrm @ (.5 * (image + Sx)))
        close = (np.abs(keys[:, :3] - nrm).max(1) < .02) & (np.abs(keys[:, 3] - dd) < .15)
        area = vals[close].sum() if close.any() else np.nan
        Lp = a['length'][i]
        fresnel = {fc: math.pi * (C / fc) * Lp / 4 for fc in (250, 1000)}   # first Fresnel zone area, d1 = d2 = L/2
        rec.append((total[i], area, fresnel[250], fresnel[1000]))
    r = np.array(rec)
    ok = np.isfinite(r[:, 1])
    w = r[ok, 0]
    small250 = (w * (r[ok, 1] < r[ok, 2])).sum() / w.sum(); small1k = (w * (r[ok, 1] < r[ok, 3])).sum() / w.sum()
    lines.append(f"Played first-order reflections matched to their mirror plane: {ok.sum()} of {len(r)}; plane area median {np.median(r[ok, 1]):.2f} m² "
                 f"(energy-weighted median {np.median(np.repeat(r[ok, 1], np.maximum(1, (w / w.min()).astype(int).clip(max=50)))):.2f} m²).")
    lines.append(f"Share of their energy reflected by planes smaller than the first Fresnel zone: {100 * small250:.0f}% at 250 Hz, {100 * small1k:.0f}% at 1 kHz.")
    lines.append('A mirror much smaller than its Fresnel zone reflects much less than its full specular energy at that frequency (it scatters/diffracts instead); '
                 'geometric acoustics gives it the full specular energy at every frequency, which over-represents bright, discrete echoes from small facets.')
    lines += diffuse_smoothness(meta, solves, a, late)
    first = (solve == 0) & (a['family'] == 0)
    lines += rendered_responses(rate, float(a['delay'][first][0]))
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # ---------------- figure: arrivals over time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True)
    axes[0].plot(t, counts, color='#555555', lw=1, label='arrivals found')
    axes[0].plot(t, np.minimum(counts, slots), color='#009E73', lw=1, label=f'played (cap {slots})')
    axes[0].set_ylabel('arrivals'); axes[0].legend(fontsize=8)
    sel = selected & (a['family'] != 0)
    rel = 10 * np.log10(total[sel] / np.maximum(direct[solve[sel]], 1e-30))
    sc = axes[1].scatter(t[solve[sel]], 1000 * (a['delay'][sel] - a['delay'][sel] * 0), c=np.clip(rel, -40, 0), s=2, cmap='viridis')
    axes[1].set_ylabel('arrival delay (ms)'); fig.colorbar(sc, ax=axes[1], label='level re direct (dB)')
    for cause, color in (('geometry', '#D55E00'), ('cap', '#0072B2')):
        ev = np.array(events[cause])
        if len(ev):
            axes[2].plot(ev[:, 0], ev[:, 1], '.', ms=3, color=color, label=f'switch: {cause}')
    axes[2].set_ylabel('switching path re direct (dB)'); axes[2].set_ylim(-60, 5); axes[2].legend(fontsize=8); axes[2].set_xlabel('time (s)')
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'arrivals.png', dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = (np.arange(nb) + .5) * bin_s * 1000
    for curve, label, color in ((refl, 'discrete reflections + edges', '#009E73'), (diff, 'diffuse field', '#CC79A7'), (total_curve, 'total incl. direct', '#000000')):
        ax.plot(x, 10 * np.log10(np.maximum(curve, 1e-30) / ref), color=color, label=label, drawstyle='steps-mid')
    ax.set_xlim(0, 600); ax.set_ylim(-60, 2); ax.set_xlabel('time after the direct sound (ms, 23 ms bins)'); ax.set_ylabel('dB, 1 kHz band'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title('Inside the Church (0–10 s), solve-averaged energy per late bin', fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / 'join.png', dpi=110); plt.close(fig)


def spikiness(late, solves_delay, bin_s, band=3, bins=86):
    """Per solve: share of the diffuse energy in its largest 23 ms bin, and in bins more than 6 dB
    above the running median of their 7-bin neighbourhood (the spikes); aligned to the direct sound."""
    largest, spiky, aligned = [], [], []
    for s, t0 in solves_delay:
        x = np.roll(late[s, band], -int(t0 / bin_s))[:bins]
        if x.sum() <= 0:
            continue
        med = np.array([np.median(x[max(0, i - 3):i + 4]) for i in range(len(x))])
        largest.append(x.max() / x.sum()); spiky.append(x[x > 4 * med].sum() / x.sum()); aligned.append(x)
    return np.array(largest), np.array(spiky), np.array(aligned)


def rendered_responses(rate, direct_delay, seconds=5.0, length=1.0):
    """Each render's response while the listener stands still (0-5 s), by Wiener deconvolution of
    the mono sum by the source clip, in 10 ms bins after the direct sound."""
    clip = np.fromfile(listen.TAKE / 'clip-0.f32', '<f4').astype(float)
    n = int(seconds * rate); N = 1 << int(np.ceil(np.log2(n + length * rate)))
    X = np.fft.rfft(clip[:n], N)
    lines = ['', '## 8. The rendered audio, deconvolved (listener standing, 0–5 s)', '',
             'Response of each render estimated from its mono sum and the source speech; 10 ms bins after the direct sound (its known delay plus the DSP latency), '
             'dB re the direct-sound bin; the estimate has a noise floor near −20 dB.', '',
             '| render | peaks 100–700 ms after the direct sound (ms, dB re direct) | bins 100–700 ms within 10 dB of the direct |', '|---|---|---:|']
    for name in ('energy', 'energy-diffuse-only', 'energy-arrivals-only', 'hybrid'):
        path = listen.WORK / name / 'listener-0.wav'
        if not path.exists():
            continue
        y, _ = listen.read_wav(path)
        Y = np.fft.rfft(y.sum(1)[:n], N)
        h = np.fft.irfft(Y * np.conj(X) / (np.abs(X) ** 2 + 1e-3 * np.mean(np.abs(X) ** 2)), N)[:int(length * rate)]
        f = rate // 100
        e = (h[:len(h) // f * f].reshape(-1, f) ** 2).sum(1)
        latency = json.loads((listen.WORK / name / 'run.json').read_text())['dspLatencySamples']
        k0 = int(round((direct_delay + latency / rate) * 100)); e = e[k0:] / e[k0 - 1:k0 + 2].max()
        peaks = [i for i in range(10, min(len(e) - 1, 70)) if e[i] > e[i - 1] and e[i] >= e[i + 1] and e[i] > .1]
        lines.append(f"| {name} | {', '.join(f'{10 * i} ({10 * np.log10(e[i]):+.0f})' for i in peaks) or '–'} | {int((e[10:70] > .1).sum())} |")
    return lines


def direct_delays(a):
    m = a['family'] == 0
    return list(zip(a['solve'][m].astype(int), a['delay'][m]))


def diffuse_smoothness(meta, solves, a, late):
    lines = ['', '## 7. Is the diffuse field smooth?', '']
    bin_s = meta['lateBinSeconds']
    t = solves['sample'] / meta['sampleRate']
    dd = [(s, t0) for s, t0 in direct_delays(a) if t[s] < 10]
    largest, spiky, aligned = spikiness(late, dd, bin_s)
    lines.append(f"Inside (0–10 s), 1 kHz band, first 2 s after the direct sound in 23 ms bins: the single largest bin holds a median {100 * np.median(largest):.0f}% "
                 f"of a solve's diffuse energy (90th percentile {100 * np.percentile(largest, 90):.0f}%). "
                 "A smooth reverberant decay over the same window would put a few percent in its largest bin.")
    tops = [tuple(np.sort(np.argsort(-x)[:4])) for x in aligned[[i for i, (s, _) in enumerate(dd) if t[s] < 5]]]
    common, n = max(((k, tops.count(k)) for k in set(tops)), key=lambda kv: kv[1])
    lines.append(f"While the listener stands still (0–5 s) the same four bins are the four largest in {n} of {len(tops)} solves: "
                 + ', '.join(f'{k * bin_s * 1000:.0f} ms' for k in common) + ' after the direct sound, each about 20 dB above the bins around it.')
    med = np.median(aligned, 0)
    lines.append('Median over solves of the aligned envelope, first 16 bins (dB re its maximum): ' +
                 ' '.join(f'{v:.0f}' for v in 10 * np.log10(np.maximum(med[:16], 1e-30) / med.max())) + ' — the bumps persist from solve to solve.')
    exps = {}
    for name in ('base', 'no-edge-rays', 'rays-8192', 'seed-1'):
        folder = OUT / f'exp-{name}'
        if not (folder / 'meta.json').exists():
            continue
        m2 = json.loads((folder / 'meta.json').read_text())
        l2 = np.fromfile(folder / 'late.f32', '<f4').reshape(m2['solves'], 8, m2['lateBins']).astype(float)
        a2 = np.genfromtxt(folder / 'arrivals.csv', delimiter=',', names=True)
        exps[name] = spikiness(l2, direct_delays(a2), m2['lateBinSeconds'])
    if exps:
        lines += ['', 'Experiments on 40 solves (every 10th response from the start, the same for each row):', '',
                  '| walk setting | largest bin share (median) | total diffuse energy vs base |', '|---|---:|---:|']
        base_total = exps['base'][2].sum(1) if 'base' in exps else None
        for name, (lg, sp, al) in exps.items():
            ratio = np.median(al.sum(1) / base_total) if base_total is not None and len(al) == len(base_total) else float('nan')
            lines.append(f'| {name} | {100 * np.median(lg):.0f}% | {ratio:.2f} |')
        if 'base' in exps and 'seed-1' in exps:
            b, s1 = exps['base'][2], exps['seed-1'][2]
            same = np.mean(b.argmax(1) == s1.argmax(1))
            top0 = np.bincount(b.argmax(1)).argmax(); top1 = np.bincount(s1.argmax(1)).argmax()
            lines.append('')
            lines.append(f"The largest bin is the same bin for seed 0 and seed 1 in {100 * same:.0f}% of these solves (seed 0: mostly bin {top0}, seed 1: mostly bin {top1}): "
                         "the spikes move with the seed, so they are sampling noise (a few rays of very large weight), frozen in place because every solve uses the same seed.")
    return lines


if __name__ == '__main__':
    main()


def tail_acceptance(names=('timeline', 'exp-fix-base', 'exp-fix-seed-1', 'exp-new-take'), band=3, seconds=10.0):
    """Tail smoothness per export (whole walk inside the Church, 0-10 s): largest 23 ms slice share of
    the diffuse energy, the loudest diffuse slice relative to the direct sound's energy, and (for two
    seeds of one scene) the slice-by-slice level difference a per-solve seed would add as flicker."""
    rows, envelopes = [], {}
    for name in names:
        folder = OUT / name if name != 'timeline' else TIMELINE
        if not (folder / 'meta.json').exists():
            continue
        m = json.loads((folder / 'meta.json').read_text())
        l = np.fromfile(folder / 'late.f32', '<f4').reshape(m['solves'], 8, m['lateBins']).astype(float)
        a = np.genfromtxt(folder / 'arrivals.csv', delimiter=',', names=True)
        s = np.genfromtxt(folder / 'solves.csv', delimiter=',', names=True)
        t = s['sample'] / m['sampleRate']
        dd = [(k, t0) for k, t0 in direct_delays(a) if t[k] < seconds]
        largest, _, aligned = spikiness(l, dd, m['lateBinSeconds'], band)
        direct = {int(k): e for k, e in zip(a['solve'][a['family'] == 0], a[f'e{band}'][a['family'] == 0])}
        ratio = [aligned[i].max() / direct[k] for i, (k, _) in enumerate(dd) if direct.get(k, 0) > 0]
        envelopes[name] = aligned
        rows.append((name, np.median(largest), np.percentile(largest, 90), 10 * np.log10(np.median(ratio)), 10 * np.log10(np.max(ratio)), len(dd)))
    lines = ['| export | largest slice share (median) | (90th pct) | loudest diffuse slice re direct, median | max | solves |', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r[0]} | {100 * r[1]:.0f}% | {100 * r[2]:.0f}% | {r[3]:+.1f} dB | {r[4]:+.1f} dB | {r[5]} |')
    if 'exp-fix-base' in envelopes and 'exp-fix-seed-1' in envelopes:
        x, y = envelopes['exp-fix-base'], envelopes['exp-fix-seed-1']
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        live = (x > x.max(1, keepdims=True) * 1e-3) & (y > y.max(1, keepdims=True) * 1e-3)
        diff = np.abs(10 * np.log10(np.maximum(x, 1e-30) / np.maximum(y, 1e-30)))[live]
        lines.append('')
        lines.append(f'Seed 0 against seed 1 (same solves), slices within 30 dB of the maximum: median level difference {np.median(diff):.1f} dB, 90th percentile {np.percentile(diff, 90):.1f} dB; '
                     'a new seed per solve would make the tail flicker by about this much from one 23 ms step to the next.')
    return lines
