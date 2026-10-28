"""Figures for the audio-DLSS training plan.

Data come from committed benchmark reports (cited in README.md) and the FDTD
feasibility run saved as modes.npz by fdtd_bench.py. Needs numpy, scipy, matplotlib.
Run from this directory: python make_figures.py
"""
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from scipy import signal

OUT = Path(__file__).parent
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0', '#fcfcfb'
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'lines.linewidth': 2,
})
C = 343.0


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def scene_sketch():
    evidence = OUT.parent / 'benchmarks' / 'wave-review-2026-10-27' / 'evidence.zip'
    with zipfile.ZipFile(evidence) as archive:
        scene = json.loads(archive.read('canopy/reference/scene.json'))
    fig, (top, side) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [1.25, 1]})
    for ax, (a, b), title, labels in ((top, (0, 2), 'Top view (roof hidden)', ('x (m)', 'z (m)')),
                                      (side, (0, 1), 'Side view (z = 0 slice)', ('x (m)', 'height y (m)'))):
        for w in scene['walls']:
            c, h = np.array(w['center']), np.array(w['half'])
            if ax is top and c[1] - h[1] > 2.4:
                continue  # roof slab and door lintel
            if ax is side and abs(c[2]) > h[2]:
                continue
            ax.add_patch(Rectangle((c[a] - h[a], c[b] - h[b]), 2 * h[a], 2 * h[b],
                                   facecolor='#cfcdc6', edgecolor=INK2, lw=0.8))
        s = scene['source']
        ax.plot(s[a], s[b], 'o', ms=9, color=S2, mec=SURF, mew=2)
        ax.annotate('source (rain patch)', (s[a], s[b]), (8, 8), textcoords='offset points', color=INK2)
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_title(title, loc='left')
        ax.set_aspect('equal'); ax.grid(False)
        ax.set_xlim(-7, 4.5)
    top.set_ylim(-4, 4); side.set_ylim(-0.3, 4)
    top.text(-4.2, 0, 'indoor\n(blocked)', ha='center', va='center', color=INK2)
    top.text(-2.15, 0, '  doorway\n  1.8 m', ha='left', va='center', color=INK2, fontsize=9)
    side.text(-4.2, 1.4, 'indoor', ha='center', color=INK2)
    fig.suptitle('The 125 Hz canopy benchmark: 7 rigid boxes, wavelength 2.7 m', x=0.01, ha='left',
                 fontweight='bold')
    save(fig, 'fig0-canopy-scene.png')


def wavelength():
    f = np.logspace(np.log10(20), np.log10(16000), 300)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.loglog(f, C / f, color=S1, label='wavelength  λ = c / f')
    sizes = [(0.25, 'wall thickness 0.25 m'), (1.8, 'doorway 1.8 m'), (6.5, 'canopy room 6.5 m'),
             (30, 'church nave ~30 m')]
    for size, label in sizes:
        ax.axhline(size, color=MUTED, lw=1, ls='--')
        ax.text(17000, size, label, va='center', ha='left', color=INK2, fontsize=9)
    ax.axvspan(500, 1400, color=S3, alpha=0.12, lw=0)
    ax.text(830, 0.012, 'published wave/ray\ncrossovers 0.5–1.4 kHz', ha='center', color=INK2, fontsize=9)
    for hz in (125, 250):
        ax.axvline(hz, color=S2, lw=1.2)
        ax.text(hz * 1.04, 60, f'{hz} Hz\n(our tests)', color=INK2, fontsize=9, va='top')
    ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('length (m)')
    ax.set_xlim(20, 16000); ax.set_ylim(0.01, 100)
    ax.set_title('Rays are a good model only when objects are much larger than the wavelength', loc='left')
    ax.legend(loc='lower left', frameon=False)
    save(fig, 'fig1-wavelength.png')


def accuracy_history():
    rows = [('2 rays/bin, before first images (10-27)', 55.92, 100.19),
            ('2 rays/bin, first images (10-27)', 50.05, 100.19),
            ('complete routes, pre-fix (10-28)', 43.04, 125.51),
            ('complete routes, height+seam fixes', 40.73, 123.94),
            ('+ solid boundary bake (accepted)', 29.24, 86.92),
            ('+ third order, global allocation', 124.09, 325.96),
            ('+ third order, per-family allocation', 119.40, 321.36)]
    labels = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, col, title in ((axes[0], 1, 'Whole map'), (axes[1], 2, 'Blocked (shadow) receivers')):
        vals = [r[col] for r in rows]
        colors = [S3 if 'accepted' in l else (S2 if 'third' in l else S1) for l in labels]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals, color=colors, height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v + 3, yi, f'{v:.0f}%', va='center', color=INK, fontsize=9)
        ax.set_yticks(y, labels); ax.set_title(title, loc='left'); ax.grid(axis='y', visible=False)
        ax.set_xlabel('complex-pressure error vs wave reference (%)')
        ax.set_xlim(0, max(vals) * 1.18)
    fig.suptitle('Canopy room at 125 Hz: the accepted bake is best; more path orders made it worse',
                 x=0.01, ha='left', fontweight='bold')
    save(fig, 'fig2-accuracy-history.png')


def complex_plane():
    ref = 0.168523 + 0.122584j
    groups = [
        ('separate objects, 1M samples (2 seeds)', S1, 'o', [0.05661 + 0.03473j, 0.07262 + 0.02525j]),
        ('solid bake, 1M samples (2 seeds)', S3, 'o', [0.088668 + 0.064164j, 0.093728 + 0.061011j]),
        ('third order, 1M samples (2 seeds)', S2, 's', [0.132591 - 0.272046j, 0.062552 - 0.045849j]),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    ax.add_patch(plt.Circle((ref.real, ref.imag), 0.05 * abs(ref), color=MUTED, alpha=0.18, lw=0))
    ax.plot(ref.real, ref.imag, '*', ms=17, color=INK, label='wave reference (BEM)')
    for label, color, marker, pts in groups:
        pts = np.array(pts)
        ax.plot(pts.real, pts.imag, marker, ms=9, color=color, mec=SURF, mew=2, ls='none', label=label)
        ax.plot(pts.real, pts.imag, color=color, lw=1, alpha=0.6)
    b = np.mean([0.088668 + 0.064164j, 0.093728 + 0.061011j])
    ax.annotate('', (ref.real, ref.imag), (b.real, b.imag),
                arrowprops=dict(arrowstyle='->', color=INK2, lw=1.2))
    ax.text(0.15, 0.06, 'bias ≈ 47%\n(does not shrink\nwith more samples)', color=INK2, fontsize=9)
    ax.text(0.03, 0.075, 'seed spread\n(noise) is small', color=INK2, fontsize=9, ha='center')
    ax.text(ref.real + 0.012, ref.imag + 0.01, '±5% target', color=MUTED, fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel('real part of pressure'); ax.set_ylabel('imaginary part of pressure')
    ax.set_aspect('equal'); ax.set_xlim(-0.02, 0.24); ax.set_ylim(-0.29, 0.17)
    ax.set_title('Indoor receiver #323, 125 Hz:\nthe error is bias, not noise', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), frameon=False, fontsize=8.5, ncol=2)
    save(fig, 'fig3-complex-plane.png')


def low_budget():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    data = {2: (42.17, 280.06), 128: (27.80, 27.40)}
    for x, (a, b) in data.items():
        ax.plot([x, x], [a, b], color=MUTED, lw=1, zorder=1)
        ax.plot(x, a, 'o', ms=9, color=S1, mec=SURF, mew=2, zorder=2, label='seed 0' if x == 2 else None)
        ax.plot(x, b, 'o', ms=9, color=S2, mec=SURF, mew=2, zorder=2, label='seed 1' if x == 2 else None)
        if x == 2:
            ax.text(x * 1.15, a, f'{a:.0f}%', va='center', color=INK, fontsize=9)
            ax.text(x * 1.15, b, f'{b:.0f}%', va='center', color=INK, fontsize=9)
        else:
            ax.text(x * 1.15, a, f'{a:.0f}% / {b:.0f}% (both seeds)', va='center', color=INK, fontsize=9)
    ax.set_xscale('log', base=2); ax.set_xticks([2, 128], ['2 rays/bin\n(realtime-like)', '128 rays/bin'])
    ax.set_xlim(1.2, 900); ax.set_ylim(0, 310)
    ax.set_ylabel('whole-map complex error (%)')
    ax.set_title('At realtime budgets, one seed can be 7× worse than another', loc='left')
    ax.legend(frameon=False, loc='upper right')
    save(fig, 'fig4-low-budget-variance.png')


def crossover():
    fc, fs = 700.0, 48000
    f = np.logspace(np.log10(20), np.log10(20000), 600)
    bl, al = signal.butter(2, fc, 'low', fs=fs); bh, ah = signal.butter(2, fc, 'high', fs=fs)
    _, hl = signal.freqz(bl, al, worN=f, fs=fs); _, hh = signal.freqz(bh, ah, worN=f, fs=fs)
    lr_low, lr_high = hl ** 2, hh ** 2  # Linkwitz-Riley 4th order = squared Butterworth
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    db = lambda h: 20 * np.log10(np.maximum(np.abs(h), 1e-6))
    ax.semilogx(f, db(lr_low), color=S1, label='wave teacher (FDTD), low-passed')
    ax.semilogx(f, db(lr_high), color=S2, label='converged ray tracer, high-passed')
    ax.semilogx(f, db(lr_low + lr_high), color=INK2, lw=1.2, ls='--', label='sum: flat magnitude')
    ax.axvline(fc, color=MUTED, lw=1)
    ax.text(fc * 1.05, -42, f'crossover\n{fc:.0f} Hz', color=INK2, fontsize=9)
    ax.set_ylim(-48, 6); ax.set_xlim(20, 20000)
    ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('gain (dB)')
    ax.set_title('Hybrid teacher: each band comes from the method that is accurate there', loc='left')
    ax.legend(frameon=False, loc='lower left', fontsize=9)
    save(fig, 'fig5-crossover.png')


def fdtd_modes():
    d = np.load(OUT / 'modes.npz')
    tr, dt, shape, dx = d['trace'], float(d['dt']), d['shape'], float(d['dx'])
    spec = np.abs(np.fft.rfft(tr * np.hanning(len(tr))))
    f = np.fft.rfftfreq(len(tr), dt)
    sel = (f > 30) & (f < 260)
    L = shape * dx
    analytic = sorted({C / 2 * np.sqrt((a / L[0]) ** 2 + (b / L[1]) ** 2 + (c / L[2]) ** 2)
                       for a in range(7) for b in range(7) for c in range(7) if a + b + c})
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    for i, m in enumerate(a for a in analytic if 30 < a < 260):
        ax.axvline(m, color=S2, lw=0.9, alpha=0.7, label='analytic room modes' if i == 0 else None)
    ax.plot(f[sel], 20 * np.log10(spec[sel] / spec[sel].max()), color=S1, lw=1.4,
            label='GPU FDTD spectrum at a corner receiver')
    ax.set_ylim(-70, 3); ax.set_xlim(30, 260)
    ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('level (dB, normalized)')
    ax.set_title(f'FDTD on the M5 GPU reproduces rigid-box modes ({L[0]:.1f}×{L[1]:.1f}×{L[2]:.1f} m): '
                 '25 peaks within 0.23%', loc='left')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    save(fig, 'fig6-fdtd-modes.png')


def fdtd_cost():
    f = np.linspace(250, 2000, 200)
    ppw, T, rate = 10, 0.5, 0.6e9  # measured naive PyTorch-MPS throughput
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for V, color, label in ((100, S1, '100 m³ (small room)'), (900, S2, '900 m³ (12×12×6 m)'),
                            (9000, S3, '9,000 m³ (courtyard / semi-open)')):
        updates = V * T * np.sqrt(3) * (f * ppw) ** 4 / C ** 3
        ax.semilogy(f, updates / rate, color=color, label=label)
    for secs, lab in ((60, '1 minute'), (3600, '1 hour')):
        ax.axhline(secs, color=MUTED, lw=1, ls='--'); ax.text(1990, secs * 1.15, lab, ha='right', color=INK2, fontsize=9)
    ax.set_xlabel('highest frequency simulated (Hz)')
    ax.set_ylabel('seconds per source run (M5, measured rate)')
    ax.set_title('FDTD cost grows with f⁴: 0.5 s response, 10 points/wavelength, 0.6 Gcell/s', loc='left')
    ax.legend(frameon=False, loc='upper left', fontsize=9)
    save(fig, 'fig7-fdtd-cost.png')


def box(ax, xy, w, h, text, color, sub=None):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                facecolor=color, edgecolor='none', alpha=0.16))
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                facecolor='none', edgecolor=color, lw=1.5))
    ax.text(xy[0] + w / 2, xy[1] + h / 2 + (0.13 if sub else 0), text, ha='center', va='center',
            fontweight='bold', fontsize=9.5, color=INK)
    if sub:
        ax.text(xy[0] + w / 2, xy[1] + h / 2 - 0.2, sub, ha='center', va='center', fontsize=8, color=INK2)


def arrow(ax, a, b):
    ax.annotate('', b, a, arrowprops=dict(arrowstyle='->', color=INK2, lw=1.3))


def pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis('off')
    box(ax, (0.1, 1.8), 1.9, 1.0, 'Scene generator', S1, 'seeded families\nrooms/doors/roofs')
    box(ax, (2.6, 3.2), 2.2, 1.0, 'Wave teacher', S2, 'FDTD ≤ ~0.5–1.4 kHz\nall probes, all freqs')
    box(ax, (2.6, 1.8), 2.2, 1.0, 'Converged tracer', S2, 'Unity, many seeds,\nhigh rays (> crossover)')
    box(ax, (2.6, 0.4), 2.2, 1.0, 'Realtime tracer', S1, 'Unity headless\n1/64…1× budgets')
    box(ax, (5.4, 1.8), 1.9, 1.0, 'Dataset writer', S3, 'hashes, splits,\nper-path buffers')
    box(ax, (7.9, 1.8), 1.4, 1.0, 'Train', S3, 'causal model')
    box(ax, (9.6, 1.8), 1.3, 1.0, 'Deploy', S3, 'ONNX → Unity')
    for y in (3.7, 2.3, 0.9):
        arrow(ax, (2.0, 2.3), (2.6, y))
        arrow(ax, (4.8, y), (5.4, 2.3))
    arrow(ax, (7.3, 2.3), (7.9, 2.3)); arrow(ax, (9.3, 2.3), (9.6, 2.3))
    ax.text(3.7, 4.42, 'target (low band)', ha='center', fontsize=8.5, color=INK2)
    ax.text(3.7, 1.53, 'target (high band)', ha='center', fontsize=8.5, color=INK2)
    ax.text(3.7, 0.12, 'network input', ha='center', fontsize=8.5, color=INK2)
    ax.text(6.35, 1.05, 'validation gate:\nteacher vs BEM/FEM\nreference set', ha='center', fontsize=8, color=INK2)
    fig.suptitle('Training-data pipeline: one scene description feeds every renderer', x=0.01, ha='left',
                 fontweight='bold')
    save(fig, 'fig8-pipeline.png')


def model_io():
    fig, ax = plt.subplots(figsize=(11, 4.4)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis('off')
    ins = [('Per-path-family H(f)', 'D, DD, RD, DR, RR, walk\ncomplex, per pair'),
           ('Sample statistics', 'path counts, variance,\nrays used'),
           ('Geometry hints', 'direct visible, distance,\nopenness, room size'),
           ('History (H solves)', 'previous low solves +\nage, motion, resets')]
    for i, (t, s) in enumerate(ins):
        box(ax, (0.1, 3.3 - i * 1.05), 2.7, 0.9, t, S1, s)
        arrow(ax, (2.8, 3.75 - i * 1.05), (3.9, 2.2))
    box(ax, (3.9, 1.6), 2.4, 1.2, 'Small causal network', S3, 'conv / MLP over\nfrequency × history')
    box(ax, (7.1, 2.6), 3.7, 1.0, 'Low band: complex H(f)', S2, 'loss: complex L1 vs wave teacher')
    box(ax, (7.1, 1.2), 3.7, 1.0, 'High band: log-magnitude + decay', S2, 'loss: log-mag + EDC vs converged tracer')
    arrow(ax, (6.3, 2.4), (7.1, 3.1)); arrow(ax, (6.3, 2.0), (7.1, 1.7))
    ax.text(8.95, 0.55, '+ temporal-stability term; fallback to raw output when invalid/stale',
            ha='center', fontsize=8.5, color=INK2)
    fig.suptitle('Audio DLSS: the acoustic analogue of G-buffers + history → clean frame', x=0.01, ha='left',
                 fontweight='bold')
    save(fig, 'fig9-model-io.png')


if __name__ == '__main__':
    scene_sketch(); wavelength(); accuracy_history(); complex_plane(); low_budget()
    crossover(); fdtd_modes(); fdtd_cost(); pipeline(); model_io()
    print('ok')
