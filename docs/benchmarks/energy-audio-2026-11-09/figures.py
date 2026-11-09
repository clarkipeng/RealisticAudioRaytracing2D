"""Figures of the energy-audio benchmark (closed-room decay before/after, wall law).

    PY=.context/fdtd-teacher/venv/bin/python
    $PY docs/benchmarks/energy-audio-2026-11-09/lattice.py shoebox-0020-presets 1.5   # lattice npz (wall law)
    $PY docs/benchmarks/energy-audio-2026-11-09/figures.py

Inputs: the comparison root (.context/method-compare/data: ground truth, hybrid and the
re-rendered energy outputs), the 6 November energy outputs kept in
.context/method-compare/energy-before-2026-11-09, and the diagnostic roulette-floor-only
runs (.context/methods-2026-11-06/raw/<scene>-rr6, -rr6-r10).
"""
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import method_compare as mc  # noqa: E402

OUT = Path(__file__).parent
DATA = ROOT / '.context/method-compare/data'
BEFORE = ROOT / '.context/method-compare/energy-before-2026-11-09'
RAW = ROOT / '.context/methods-2026-11-06/raw'
SCENES = ['shoebox-0020-presets', 'shoebox-0010-presets', 'furnished-0005-presets']
COLORS = dict(gt='#000000', hybrid='#0072B2', energy='#009E73', before='#009E73', lattice='#E69F00', impedance='#CC79A7')


def raw(folder):
    run = json.loads((folder / 'run.json').read_text())
    e = np.fromfile(folder / 'energy.f32', '<f4').reshape(run['receivers'], 8, run['bins']).astype(np.float64)
    return np.concatenate([e, np.zeros(e.shape[:2] + (mc.TIME_BINS - e.shape[2],))], 2) if e.shape[2] < mc.TIME_BINS else e


def curves(ax_etc, ax_edc, entries, band):
    t = (np.arange(mc.TIME_BINS) + .5) * mc.BIN_SECONDS
    for label, h, color, style in entries:
        mean = h[:, band].mean(0)
        w = mean.reshape(-1, 5).sum(1)
        ax_etc.plot(t[::5] + .0025, 10 * np.log10(np.maximum(w, 1e-30) / w.max()), style, color=color, lw=1, label=label)
        with np.errstate(divide='ignore', invalid='ignore'):
            edc = np.array([mc.schroeder(x) for x in h[:, band]])
            edc = np.where(np.isfinite(edc), edc, np.nan)
        ax_edc.plot(t, np.nanmean(edc, 0), style, color=color, lw=1.2, label=label)


def decay_figure(band=1):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex='col')
    for col, sid in enumerate(SCENES):
        gt = mc.read_output(DATA / sid / 'gt/ref/0')[1]
        entries = [('ground truth (FDTD)', gt, COLORS['gt'], '-'),
                   ('hybrid', mc.read_output(DATA / sid / 'hybrid/r02048/0')[1], COLORS['hybrid'], '-'),
                   ('energy, 6 Nov (roulette at energy rate, retrace detection)', mc.smooth(mc.read_output(BEFORE / sid / 'energy/r02048/0')[1]), COLORS['before'], ':'),
                   ('energy, 9 Nov (roulette floor, arrivals + sphere)', mc.smooth(mc.read_output(DATA / sid / 'energy/r02048/0')[1]), COLORS['energy'], '-')]
        curves(axes[0, col], axes[1, col], entries, band)
        axes[0, col].set_title(f'{sid}, {int(mc.BANDS[band])} Hz, 2,048 rays', fontsize=9)
        axes[0, col].set_ylim(-70, 2); axes[1, col].set_ylim(-60, 2); axes[1, col].set_xlabel('time (s)')
        for ax in axes[:, col]:
            ax.grid(alpha=.3)
    axes[0, 0].set_ylabel('receiver-mean ETC, 5 ms (dB)'); axes[1, 0].set_ylabel('mean Schroeder decay (dB)')
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'fig-decay.png', dpi=120); plt.close(fig)


def diagnosis_figure(band=1):
    """Furnished room: what each change does (roulette floor, smaller detection cone, sphere)."""
    sid = 'furnished-0005-presets'
    gt = mc.read_output(DATA / sid / 'gt/ref/0')[1]
    entries = [('ground truth (FDTD)', gt, COLORS['gt'], '-'),
               ('6 Nov: roulette at energy rate, retrace r = 0.25 m', mc.smooth(mc.read_output(BEFORE / sid / 'energy/r02048/0')[1]), '#999999', ':'),
               ('roulette floor 1e-6, retrace r = 0.25 m', mc.smooth(raw(RAW / f'{sid}-rr6/energy/r02048/0')), '#E69F00', '-'),
               ('roulette floor 1e-6, retrace r = 0.10 m', mc.smooth(raw(RAW / f'{sid}-rr6-r10/energy/r02048/0')), '#CC79A7', '-'),
               ('9 Nov: floor, arrivals + receiver sphere', mc.smooth(mc.read_output(DATA / sid / 'energy/r02048/0')[1]), COLORS['energy'], '-')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    curves(axes[0], axes[1], entries, band)
    axes[0].set_ylim(-70, 2); axes[1].set_ylim(-60, 2)
    axes[0].set_title(f'{sid}, {int(mc.BANDS[band])} Hz: receiver-mean ETC', fontsize=9); axes[1].set_title('mean Schroeder decay', fontsize=9)
    for ax in axes:
        ax.grid(alpha=.3); ax.set_xlabel('time (s)')
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'fig-furnished.png', dpi=120); plt.close(fig)


def wall_law_figure():
    sid = 'shoebox-0020-presets'
    path = ROOT / f'.context/energy-audio/lattice-{sid}.npz'
    if not path.exists():
        print('run lattice.py first'); return
    data = np.load(path)
    pick = data['pick']
    gt = mc.read_output(DATA / sid / 'gt/ref/0')[1][pick]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, band in zip(axes, (0, 3)):
        entries = [('ground truth (FDTD, impedance walls)', gt, COLORS['gt'], '-'),
                   ('image lattice, fixed absorption (the methods)', mc.smooth(data['a'][pick]), COLORS['lattice'], '-'),
                   ('image lattice, |R(θ)|² of the FDTD impedance', mc.smooth(data['b'][pick]), COLORS['impedance'], '--'),
                   ('energy method, 9 Nov', mc.smooth(mc.read_output(DATA / sid / 'energy/r02048/0')[1][pick]), COLORS['energy'], '-')]
        t = (np.arange(mc.TIME_BINS) + .5) * mc.BIN_SECONDS
        for label, h, color, style in entries:
            with np.errstate(divide='ignore', invalid='ignore'):
                edc = np.array([mc.schroeder(x) for x in h[:, band]])
            ax.plot(t, np.nanmean(np.where(np.isfinite(edc), edc, np.nan), 0), style, color=color, lw=1.2, label=label)
        ax.set_xlim(0, 1.2); ax.set_ylim(-60, 2); ax.grid(alpha=.3); ax.set_xlabel('time (s)')
        ax.set_title(f'{sid}, {int(mc.BANDS[band])} Hz, 8 receivers: mean Schroeder decay', fontsize=9)
    axes[0].set_ylabel('dB'); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'fig-wall-law.png', dpi=120); plt.close(fig)


if __name__ == '__main__':
    decay_figure()
    diagnosis_figure()
    wall_law_figure()
    print('figures written to', OUT)
