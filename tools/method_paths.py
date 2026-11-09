"""Top-down path pictures of the three acoustic methods from their debug captures.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY tools/method_paths.py SCENE_ID [--out docs/benchmarks/methods-2026-11-06]

Reads .context/methods-2026-11-06/raw/<scene>/<method>/<budget>/<seed>/capture.json
(written by TakeChecks.MeasureMethods with captureReceiver >= 0) and the scene's
boxes, and draws one panel per method: every captured path as a polyline whose
segments are coloured by the event that ends them (specular, scatter, transmission,
edge) and whose width/opacity follow the path weight; listener connections are thin
lines in the method's colour, opacity by contribution. A second row plots each
captured path's weight against its event index (log scale), which is where the
pressure walk's weight growth shows.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '.context/methods-2026-11-06/raw'
SPECS = ROOT / '.context/methods-2026-11-06/specs'
METHOD_COLORS = dict(gt='#000000', hybrid='#0072B2', pressure='#D55E00', energy='#009E73')
# Event colours (Okabe-Ito, distinct from nothing else on the panel).
EVENTS = {'specular': '#56B4E9', 'scatter': '#E69F00', 'transmission': '#CC79A7', 'edge': '#000000', 'direct': '#BBBBBB'}
METHOD_KIND = {1: 'source', 2: 'specular', 3: 'scatter', 4: 'transmission', 5: 'edge', 6: 'connection', 7: 'chain',
               8: 'escape', 9: 'end', 10: 'edgeConnection'}
DRAWN_EVENTS, DRAWN_RAYS = 12, 12   # drawn per panel (the weight plot keeps every captured path)
HYBRID_KIND = {1: 'transmission', 2: 'specular', 3: 'scatter', 4: 'edge', 5: 'listener', 6: 'escape', 7: 'rejected'}


def find_capture(scene, method):
    found = sorted((RAW / scene / method).glob('*/*/capture.json'))
    return json.loads(found[0].read_text()) if found else None


def draw_boxes(ax, spec, axes):
    for wall in spec['walls']:
        c, h = np.array(wall['center']), np.array(wall['half'])
        ax.add_patch(Rectangle((c[axes[0]] - h[axes[0]], c[axes[1]] - h[axes[1]]), 2 * h[axes[0]], 2 * h[axes[1]],
                               facecolor='#dddddd', edgecolor='#888888', lw=.6, zorder=1))


def weight_style(w, top):
    """Width and alpha from a weight relative to the panel's largest (log scale over 6 decades)."""
    level = np.clip(1 + np.log10(np.maximum(w, 1e-30) / top) / 6, 0, 1)
    return .3 + 1.7 * level, .08 + .8 * level


def method_panel(ax, wax, capture, method, axes):
    rays = capture['rays']
    weights = [e[4] for ray in rays for e in ray if METHOD_KIND.get(e[0]) in ('specular', 'scatter', 'transmission', 'edge') and e[4] > 0]
    contributions = [e[5] for ray in rays for e in ray if e[5] > 0]
    top_w = max(weights) if weights else 1
    top_c = max(contributions) if contributions else 1
    listener = np.array(capture['listener'])
    segments, colors, widths = [], [], []
    links, link_alpha = [], []
    for index, ray in enumerate(rays):
        position, previous, trace = None, None, []
        drawn = 0 if index < DRAWN_RAYS else DRAWN_EVENTS   # later rays: weight plot only
        for e in ray:
            kind = METHOD_KIND.get(e[0], '?')
            p = np.array(e[1:4])
            if kind == 'source':
                position = p
                continue
            if kind in ('connection', 'chain') and drawn < DRAWN_EVENTS:
                links.append([p[axes], listener[axes]]); link_alpha.append(e[5] / top_c)
                continue
            if kind == 'edgeConnection':
                if position is not None:
                    links.append([position[axes], p[axes]]); link_alpha.append(e[5] / top_c)
                links.append([p[axes], listener[axes]]); link_alpha.append(e[5] / top_c)
                continue
            if position is not None and kind != 'end' and drawn < DRAWN_EVENTS:
                drawn += 1
                segments.append([position[axes], p[axes]])
                style = 'direct' if kind == 'escape' else kind
                colors.append(EVENTS.get(style, '#999999'))
                w, a = weight_style(e[4] if e[4] > 0 else (trace[-1] if trace else top_w), top_w)
                widths.append((.4, .12) if kind == 'escape' else (w, a))   # escaping rays faint
            if kind in ('specular', 'scatter', 'transmission', 'edge') and e[4] > 0:
                trace.append(e[4])
            position = p
        if trace:
            wax.semilogy(np.arange(1, len(trace) + 1), trace, color=METHOD_COLORS[method], alpha=.35, lw=.8)
    if links:
        alpha = np.clip(.05 + .9 * np.clip(1 + np.log10(np.maximum(link_alpha, 1e-30)) / 6, 0, 1), 0, 1)
        rgba = [matplotlib.colors.to_rgba(METHOD_COLORS[method], a) for a in alpha]
        ax.add_collection(LineCollection(links, colors=rgba, linewidths=.5, zorder=2))
    if segments:
        rgba = [matplotlib.colors.to_rgba(c, a) for c, (w, a) in zip(colors, widths)]
        ax.add_collection(LineCollection(segments, colors=rgba, linewidths=[w for w, a in widths], zorder=3))


def hybrid_panel(ax, capture, axes):
    groups = {}
    for s in capture['segments']:
        groups.setdefault(s[1], []).append(s)
    top = max([abs(s[9]) for s in capture['segments'] if s[9] is not None] + [1e-30])
    segments, colors, widths = [], [], []
    for path in groups.values():
        for s in path:
            kind = HYBRID_KIND.get(s[0], '?')
            a, b = np.array(s[2:5]), np.array(s[5:8])
            w, alpha = weight_style(abs(s[9] or 0), top)
            colors.append(matplotlib.colors.to_rgba(METHOD_COLORS['hybrid'] if kind == 'listener' else EVENTS.get(kind, '#999999'), alpha))
            segments.append([a[axes], b[axes]]); widths.append(w if kind != 'listener' else .5)
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, zorder=3))


def figure(scene, out, methods=('hybrid', 'pressure', 'energy')):
    spec = json.loads((SPECS / f'{scene}.json').read_text())
    meta = json.loads((ROOT / '.context/method-compare/data' / scene / 'scene.json').read_text()) \
        if (ROOT / '.context/method-compare/data' / scene / 'scene.json').exists() else {}
    axes = meta.get('viewAxes', [0, 2])
    captures = {m: find_capture(scene, m) for m in methods}
    methods = [m for m in methods if captures[m]]
    if not methods:
        raise SystemExit(f'No captures for {scene}')
    fig, grid = plt.subplots(2, len(methods), figsize=(5.2 * len(methods), 9), gridspec_kw=dict(height_ratios=[3, 1.3]), squeeze=False)
    pts = np.array(spec['points']).reshape(-1, 3)
    lo = np.minimum(pts.min(0), np.min([np.array(w['center']) - w['half'] for w in spec['walls']], 0)) - .6
    hi = np.maximum(pts.max(0), np.max([np.array(w['center']) + w['half'] for w in spec['walls']], 0)) + .6
    for i, m in enumerate(methods):
        ax, wax = grid[0, i], grid[1, i]
        draw_boxes(ax, spec, axes)
        c = captures[m]
        if m == 'hybrid':
            hybrid_panel(ax, c, axes)
            wax.text(.5, .5, 'deterministic images and\nquadrature routes (no walk)', ha='center', va='center', transform=wax.transAxes, color='#555555')
            wax.set_xticks([]); wax.set_yticks([])
        else:
            method_panel(ax, wax, c, m, axes)
            wax.set_xlabel('event index along the path'); wax.set_ylabel('path weight')
            wax.grid(alpha=.3, which='both', lw=.4)
        s, l = np.array(c['source']), np.array(c['listener'])
        ax.plot(*s[axes], marker='*', ms=13, color='#F0E442', mec='k', zorder=5)
        ax.plot(*l[axes], marker='o', ms=8, color='w', mec=METHOD_COLORS[m], mew=2, zorder=5)
        ax.set_xlim(lo[axes[0]], hi[axes[0]]); ax.set_ylim(lo[axes[1]], hi[axes[1]]); ax.set_aspect('equal')
        ax.set_title(m, color=METHOD_COLORS[m], fontweight='bold')
        ax.set_xlabel('xyz'[axes[0]] + ' (m)'); ax.set_ylabel('xyz'[axes[1]] + ' (m)')
        for spine in ax.spines.values():
            spine.set_edgecolor(METHOD_COLORS[m]); spine.set_linewidth(1.5)
    handles = [plt.Line2D([], [], color=c, lw=2, label=k) for k, c in EVENTS.items() if k != 'direct'] + \
              [plt.Line2D([], [], color='#777777', lw=.6, label='listener connection (method colour)')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False)
    fig.suptitle(f'{scene}: captured paths, top view (receiver {captures[methods[0]]["receiver"]})')
    fig.tight_layout(rect=(0, .04, 1, .97))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'paths-{scene}.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scenes', nargs='+')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/benchmarks/methods-2026-11-06')
    args = parser.parse_args()
    for scene in args.scenes:
        figure(scene, args.out)
