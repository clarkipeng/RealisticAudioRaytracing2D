"""Figures for docs/writeup.md and docs/method.md, from existing data.

    .context/fdtd-teacher/venv/bin/python docs/figures/make_figures.py

Sources: the scoreboard (docs/benchmarks/energy-audio-2026-11-09/scoreboard.json), the lab notebook's tables
(docs/benchmarks/path-denoise-2026-11-12/README.md, the part named beside each table below), the live showcase's
smoke-run timelines (docs/figures/data/*.csv, copied from .context/path-live/smoke-*/<scene>/timeline.csv), and the
method's own formulas (the population profile and the beam width).
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INK, MUTED, GRID = '#1f2328', '#57606a', '#d0d7de'
BLUE, ORANGE, GREEN, GREY = '#0969da', '#bc4c00', '#1a7f37', '#8c959f'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'figure.dpi': 150})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / name)
    plt.close(fig)
    print('wrote', HERE / name)


def population():
    """Population control (method.md section 5): S(a) = min(1, T_f / a), bins of width b(tau), contributions per bin."""
    tf, fine = .016, .00025
    a = np.geomspace(.0005, 6, 400)
    s = np.minimum(1, tf / a)
    b = np.where(a < tf, fine, a / 64)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].loglog(a * 1000, s, color=BLUE, label='rays kept, S(a) = min(1, T$_f$/a)')
    ax[0].loglog(a * 1000, b / b.max(), color=ORANGE, ls='--', label='bin width b(a), relative')
    ax[0].axvline(tf * 1000, color=GREY, lw=.8)
    ax[0].text(tf * 1000 * 1.15, .3, 'T$_f$ = 16 ms', color=MUTED)
    ax[0].set_xlabel('age of the rays, ms'); ax[0].set_ylabel('relative'); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title('Survival profile', color=INK)
    per_bin_flat = s * b / fine
    per_bin_const = np.ones_like(a) * b / fine
    ax[1].loglog(a * 1000, per_bin_flat, color=BLUE, label='S(a) = min(1, T$_f$/a): flat')
    ax[1].loglog(a * 1000, per_bin_const, color=GREY, ls=':', label='every ray to the horizon')
    ax[1].set_xlabel('delay, ms'); ax[1].set_ylabel('contributions per bin (relative)'); ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title('Contributions per histogram bin', color=INK)
    save(fig, 'population.png')


def beam_width():
    """Fresnel-width beams (method.md section 6.1): w = sqrt(lambda L / pi)."""
    L = np.linspace(0, 40, 200)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for f, color in [(125, ORANGE), (500, GREEN), (2000, BLUE), (8000, GREY)]:
        ax.plot(L, np.sqrt(343 / f * L / np.pi), color=color, label=f'{f} Hz')
    ax.set_xlabel('unfolded path length L, m'); ax.set_ylabel('beam radius w, m')
    ax.legend(frameon=False); ax.set_title('Beam width w² = λL/π', color=INK)
    save(fig, 'beam-width.png')


def scoreboard():
    d = json.loads((ROOT / 'docs/benchmarks/energy-audio-2026-11-09/scoreboard.json').read_text())['suite']
    groups = ['all', 'in view', 'shadow', 'closed scenes']
    methods = [('baseline-static-ir', 'static-IR baseline', GREY), ('energy', 'energy method (replaced)', ORANGE),
               ('path', 'path method', BLUE)]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(groups)); w = .26
    for i, (key, label, color) in enumerate(methods):
        v = [d[key]['groups'][g]['level'] for g in groups]
        ax.bar(x + (i - 1) * w, v, w, color=color, label=label)
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi + .15, f'{vi:.1f}', ha='center', fontsize=7, color=MUTED)
    ax.set_xticks(x, groups); ax.set_ylabel('level error against FDTD, dB RMS')
    ax.legend(frameon=False, fontsize=8); ax.set_title('Wave ground truth, 11 scenes', color=INK)
    ax.grid(axis='x', visible=False)
    save(fig, 'scoreboard.png')


def denoiser():
    """Notebook part 33: the Church, window error against 8 N targets, by population."""
    n = [4000, 6000, 11000, 16000, 23000]
    noisy = [13.03, 11.76, 10.35, 9.74, 8.92]
    denoised = [8.09, 7.48, 6.34, 6.24, 6.12]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(n, noisy, 'o-', color=GREY, label='one update, raw')
    ax.plot(n, denoised, 'o-', color=BLUE, label='denoised')
    ax.axhspan(4.5, 6.0, color=GRID, alpha=.6, lw=0)
    ax.text(4200, 5.0, "the targets' own noise", color=MUTED, fontsize=8)
    ax.set_xscale('log'); ax.set_xticks(n, [f'{v // 1000}k' for v in n]); ax.minorticks_off()
    ax.set_xlabel('rays in flight, N'); ax.set_ylabel('window error, dB'); ax.set_ylim(4, 14)
    ax.legend(frameon=False); ax.set_title('Denoiser in the Church', color=INK)
    save(fig, 'denoiser.png')


def cost():
    """Notebook parts 27-37: GPU time per update in the Church at N = 11,000, by round."""
    rounds = ['part 27\nstart', 'part 27\nmesh trees', 'part 29\ngroup spectra', 'part 31\nsorted pool',
              'part 33\n64-wide denoiser', 'part 37\nkernel build']
    tracer = [24.0, 6.2, 5.15, 4.2, 4.2, 4.2]
    audio = [1.4, 2.3, 2.2, 2.07, 2.07, 1.45]
    denoise = [1.6, 1.0, 1.0, 1.0, 0.64, 0.64]
    fig, ax = plt.subplots(figsize=(7, 3.3))
    x = np.arange(len(rounds))
    ax.bar(x, tracer, color=BLUE, label='tracer')
    ax.bar(x, audio, bottom=tracer, color=GREEN, label='audio')
    ax.bar(x, denoise, bottom=np.add(tracer, audio), color=ORANGE, label='denoiser')
    for xi, t in zip(x, np.add(np.add(tracer, audio), denoise)):
        ax.text(xi, t + .4, f'{t:.1f}', ha='center', fontsize=8, color=INK)
    ax.axhline(21.3, color=GREY, lw=.8, ls='--'); ax.text(5.5, 21.8, 'one update of audio, 21.3 ms', ha='right', color=MUTED, fontsize=8)
    ax.set_xticks(x, rounds, fontsize=7); ax.set_ylabel('GPU ms per update'); ax.legend(frameon=False, fontsize=8)
    ax.set_title('The Church, 11,000 rays in flight', color=INK); ax.grid(axis='x', visible=False)
    save(fig, 'cost.png')


def itd():
    """Notebook parts 42 and 45: echo ITD against the direct path's."""
    az = [-90, -60, -30, 0, 30, 60, 90]
    static_echo = [-734.4, -515.6, -262.7, 0.0, 260.7, 513.7, 734.4]
    direct = [-736.3, -515.6, -260.7, 0.0, 260.7, 515.6, 736.3]
    moving_echo = [-753.9, -531.3, -266.6, -1.0, 264.6, 529.3, 753.9]
    moving_expected = [-757.8, -530.7, -268.3, 0.0, 268.3, 530.7, 757.8]
    woodworth = [-655.8, -488.1, -261.1, 0.0, 261.1, 488.1, 655.8]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(az, direct, '-', color=GREY, label='direct path, same HRTF blend')
    ax.plot(az, static_echo, 'o', color=BLUE, label='echo, static scene')
    ax.plot(az, moving_expected, '--', color=GREY, lw=.8, label='expected, image receding at 10 m/s')
    ax.plot(az, moving_echo, 's', mfc='none', color=ORANGE, label='echo, moving')
    ax.plot(az, woodworth, ':', color=MUTED, label='Woodworth, 8.75 cm head')
    ax.set_xlabel('azimuth of the reflection, degrees'); ax.set_ylabel('interaural delay, µs')
    ax.legend(frameon=False, fontsize=7); ax.set_title('Echo ITD', color=INK)
    save(fig, 'itd.png')


def live():
    """The live showcase's frame GPU time over a scripted walk (docs/figures/data/<scene>.csv)."""
    files = sorted((HERE / 'data').glob('*.csv'))
    if not files:
        return
    fig, axes = plt.subplots(len(files), 1, figsize=(7, 2.2 * len(files)), squeeze=False)
    for ax, path in zip(axes[:, 0], files):
        rows = list(csv.DictReader(path.open()))
        t = np.array([float(r['seconds']) for r in rows]); g = np.array([float(r['frameGpuMs']) for r in rows])
        ax.plot(t, g, lw=.5, color=BLUE)
        ax.axhline(1000 / 60, color=GREY, lw=.8, ls='--')
        phases, last = [], None
        for r in rows:
            if r['phase'] != last:
                phases.append((float(r['seconds']), r['phase'])); last = r['phase']
        for s, p in phases:
            ax.axvline(s, color=GRID, lw=.6)
        ax.set_ylabel('frame GPU, ms'); ax.set_title(path.stem, color=INK, fontsize=9)
    axes[-1, 0].set_xlabel('seconds into the walk')
    save(fig, 'live.png')


if __name__ == '__main__':
    population(); beam_width(); scoreboard(); denoiser(); cost(); itd(); live()
