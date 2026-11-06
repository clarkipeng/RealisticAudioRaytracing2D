"""Static HTML viewer and PNG figures for the acoustic method comparison.

    python tools/method_compare.py view [--root R] [--out D] [--figures DOCS_DIR]

Scores every method under the root (tools/method_compare.py), then writes
`<out>/index.html` plus one data script per scene (`<out>/data/*.js`, loaded on
demand so the page works from file:// with no server) and convergence PNGs.
The page has a summary (one row per method), per-scene views (sketch, level and
error maps per band, echograms and Schroeder curves at a chosen receiver,
complex pressure where available) and convergence plots. It adapts to whatever
methods exist under the root; colours are fixed per method (Okabe-Ito).
"""
import json
import math
from pathlib import Path

import numpy as np

import method_compare as mc

CURVE_BIN = 2          # ms per bin in the viewer's echograms
CURVE_FLOOR = -90.0    # dB re free-field 1 m


def best_runs(runs):
    """{method: (budget, seed, path)}: each method's largest budget, lowest seed."""
    out = {}
    for method, budgets in runs.items():
        def key(b):
            seeds = budgets[b]
            meta = json.loads((seeds[min(seeds)] / 'meta.json').read_text())
            return (meta.get('budgetValue') if meta.get('budgetValue') is not None else -1, meta.get('wallSeconds', 0))
        budget = max(budgets, key=key)
        seed = min(budgets[budget])
        out[method] = (budget, seed, budgets[budget][seed])
    return out


def project_walls(scene, axes, height):
    a, b = axes
    fixed = 3 - a - b
    rects = []
    for w in scene['walls']:
        c, h = np.array(w['center']), np.array(w['half'])
        lo, hi = c - h, c + h
        rects.append(dict(x0=lo[a], x1=hi[a], y0=lo[b], y1=hi[b],
                          cut=bool(lo[fixed] <= height <= hi[fixed])))
    return rects


def curve_db(e, valid):
    """Energy [T] -> dB per CURVE_BIN ms up to the valid length (ints of 0.1 dB)."""
    n = valid // CURVE_BIN * CURVE_BIN
    x = np.asarray(e[:n], float).reshape(-1, CURVE_BIN).sum(1)
    with np.errstate(divide='ignore'):
        d = 10 * np.log10(np.maximum(x, 10 ** (CURVE_FLOOR / 10)))
    return np.round(d * 10).astype(int).tolist()


def scene_data(root, sid, runs, report_rows):
    scene = mc.load_scene(root, sid)
    gt = mc.read_output(runs['gt']['ref'][0])
    axes = scene.get('viewAxes', [0, 2])
    pts = np.array(scene['points'], float).reshape(-1, 3)
    fixed = 3 - sum(axes)
    height = float(np.median(pts[:, fixed]))
    bands = [b for b, fc in enumerate(mc.BANDS) if float(fc) in map(float, gt[0]['bandsProvided'])]
    best = best_runs(runs)
    methods = ['gt'] + sorted(m for m in best if m != 'gt')
    valid_gt = int(round(gt[0]['validSeconds'] / mc.BIN_SECONDS))
    data = dict(id=sid, family=scene.get('family'), variant=scene.get('variant'), closed=bool(scene.get('closed')),
                axes=axes, height=height, walls=project_walls(scene, axes, height),
                source=[scene['source'][axes[0]], scene['source'][axes[1]]],
                receivers=np.round(pts[:, axes], 4).tolist(), shadow=[int(v) for v in scene['shadow']],
                spacing=scene.get('receiverGrid', {}).get('spacing'), bands=[float(mc.BANDS[b]) for b in bands],
                curveBin=CURVE_BIN, methods=[], levels={}, curves={}, complex={})
    for m in methods:
        budget, seed, path = best[m] if m != 'gt' else ('ref', 0, runs['gt']['ref'][0])
        meta, e, cp, _ = mc.read_output(path)
        if m != 'gt' and not meta.get('bandLimited', False):
            e = mc.smooth(e)
        valid = min(valid_gt, int(round(meta['validSeconds'] / mc.BIN_SECONDS)))
        levels, curves = {}, {}
        complex_levels = mc.levels_from_complex(cp, meta['frequenciesHz']) if cp is not None else None
        for b in bands:
            tot = e[:, b, :valid].sum(1)
            if not np.isfinite(tot).any() and complex_levels is not None:
                tot = complex_levels[:, b]
            with np.errstate(divide='ignore', invalid='ignore'):
                lv = 10 * np.log10(tot)
            levels[f'{mc.BANDS[b]:g}'] = [round(float(v), 2) if np.isfinite(v) else None for v in lv]
            if np.isfinite(e[:, b]).all():
                curves[f'{mc.BANDS[b]:g}'] = [curve_db(e[r, b], valid) for r in range(e.shape[0])]
        data['levels'][m] = levels
        data['curves'][m] = curves
        if cp is not None:
            data['complex'][m] = dict(hz=meta['frequenciesHz'], re=np.round(cp.real, 6).tolist(),
                                      im=np.round(cp.imag, 6).tolist())
        data['methods'].append(dict(name=m, budget=budget, seed=seed, color=mc.method_color(m, methods),
                                    valid=valid * mc.BIN_SECONDS, bandLimited=bool(meta.get('bandLimited', False))))
    return data


def summary_table(report):
    """Rows for the summary page: per method (top budget) x band x group x openness."""
    rows = report['rows']
    out = []
    for method in sorted({r['method'] for r in rows}):
        budget = mc.top_budget(rows, method)
        sel = [r for r in rows if r['method'] == method and r['budget'] == budget]
        cost = [r['wallSeconds'] for r in sel if r['wallSeconds'] is not None and r['group'] == 'all']
        entry = dict(method=method, budget=budget, cost=float(np.mean(cost)) if cost else None,
                     color=mc.method_color(method, {r['method'] for r in rows}), cells={})
        for metric in ('level', 'levelBias', 'T30', 'EDT', 'C50', 'onset', 'etc', 'complex'):
            for band in sorted({r['band'] for r in sel}) + ['all']:
                for openness in ('open', 'closed', 'all'):
                    for group in mc.GROUPS:
                        rs = [r for r in sel if (band == 'all' or r['band'] == band) and r['group'] == group
                              and (openness == 'all' or r['closed'] == (openness == 'closed'))]
                        v = mc.aggregate(rs, ('method',), metric).get((method,)) if rs else None
                        if v is not None:
                            entry['cells'][f'{metric}|{band}|{openness}|{group}'] = round(v, 4)
        out.append(entry)
    return out


def convergence_figure(report, path, metric='level', title=None):
    """Error vs budget per method (all bands with GT), with seed spread; one panel per group."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in report['rows'] if r['method'] not in ('gt', 'bem') and r.get(metric) is not None]
    methods = sorted({r['method'] for r in rows})
    if not methods:
        return False
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, constrained_layout=True)
    for ax, group in zip(axs, mc.GROUPS):
        for m in methods:
            color = mc.method_color(m, methods)
            sel = [r for r in rows if r['method'] == m and r['group'] == group]
            budgets = sorted({r['budget'] for r in sel},
                             key=lambda b: max((r['budgetValue'] or 0) for r in sel if r['budget'] == b))
            xs, mean, lo, hi = [], [], [], []
            for b in budgets:
                rb = [r for r in sel if r['budget'] == b]
                per_seed = mc.aggregate(rb, ('seed',), metric)
                vals = np.array(list(per_seed.values()))
                if not len(vals):
                    continue
                x = max((r['budgetValue'] or 0) for r in rb) or (budgets.index(b) + 1)
                xs.append(x)
                mean.append(vals.mean())
                lo.append(vals.min())
                hi.append(vals.max())
            if not xs:
                continue
            ax.fill_between(xs, lo, hi, color=color, alpha=.18, linewidth=0)
            ax.plot(xs, mean, '-o', color=color, lw=2, ms=6, label=m)
            ax.annotate(m, (xs[-1], mean[-1]), xytext=(4, 0), textcoords='offset points', color='#333333',
                        fontsize=8, va='center')
        ax.set_xscale('log')
        from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
        values = sorted({max(1, r['budgetValue'] or 1) for r in rows})
        ax.xaxis.set_major_locator(FixedLocator(values))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(bottom=0)
        ax.set_title(f'{group} receivers', fontsize=10)
        ax.set_xlabel('budget (budgetValue)')
        ax.grid(True, color='#e6e6e6', lw=.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    unit = dict(level='dB RMS', etc='dB RMS', T30='% RMS', EDT='% RMS', C50='dB RMS', onset='ms RMS',
                complex='relative L2').get(metric, '')
    axs[0].set_ylabel(f'{metric} error ({unit})')
    axs[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(title or f'{metric} error vs budget (line: mean over seeds; band: seed range)', fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def sketch_figure(root, scenes, path):
    """Top-down sketches of every scene: geometry cut at receiver height, source, receivers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    n = len(scenes)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, sid in zip(axs, scenes):
        scene = mc.load_scene(root, sid)
        axes = scene.get('viewAxes', [0, 2])
        pts = np.array(scene['points'], float).reshape(-1, 3)
        height = float(np.median(pts[:, 3 - sum(axes)]))
        for w in project_walls(scene, axes, height):
            ax.add_patch(Rectangle((w['x0'], w['y0']), w['x1'] - w['x0'], w['y1'] - w['y0'],
                                   facecolor='#9a9a9a' if w['cut'] else 'none', edgecolor='#9a9a9a',
                                   lw=.8, ls='-' if w['cut'] else '--'))
        sh = np.asarray(scene['shadow'], bool)
        ax.scatter(pts[~sh, axes[0]], pts[~sh, axes[1]], s=12, color='#0072B2', label='in view')
        ax.scatter(pts[sh, axes[0]], pts[sh, axes[1]], s=14, facecolors='none', edgecolors='#D55E00',
                   marker='s', label='shadow')
        ax.plot(scene['source'][axes[0]], scene['source'][axes[1]], '*', color='#000000', ms=12, label='source')
        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_title(sid, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axs[n:]:
        ax.axis('off')
    axs[0].legend(fontsize=7, frameon=False, loc='upper left')
    fig.savefig(path, dpi=120)
    plt.close(fig)


def gt_figure(root, scenes, path):
    """Ground-truth echograms (one in-view and one shadow receiver per scene) and band levels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    band_colors = {125.0: '#0072B2', 250.0: '#009E73', 500.0: '#D55E00', 1000.0: '#CC79A7'}
    fig, axs = plt.subplots(len(scenes), 2, figsize=(11, 2.3 * len(scenes)), constrained_layout=True, squeeze=False)
    found = mc.discover(root)
    for row, sid in zip(axs, scenes):
        scene = mc.load_scene(root, sid)
        meta, e, _, _ = mc.read_output(found[sid]['gt']['ref'][0])
        shadow = np.asarray(scene['shadow'], bool)
        pts = np.array(scene['points']).reshape(-1, 3)
        dist = np.linalg.norm(pts - scene['source'], axis=1)
        valid = int(round(meta['validSeconds'] / mc.BIN_SECONDS))
        for ax, flag, name in ((row[0], False, 'in view'), (row[1], True, 'shadow')):
            idx = np.flatnonzero(shadow == flag)
            if not len(idx):
                ax.text(.5, .5, f'no {name} receivers', ha='center', transform=ax.transAxes, color='#666666')
                ax.axis('off')
                continue
            r = idx[np.argsort(dist[idx])[len(idx) // 2]]
            for b, fc in enumerate(mc.BANDS):
                if not np.isfinite(e[r, b, 0]):
                    continue
                x = e[r, b, :valid].reshape(-1, 2).sum(1) if valid % 2 == 0 else e[r, b, :valid - 1].reshape(-1, 2).sum(1)
                ax.plot(np.arange(len(x)) * 2e-3, 10 * np.log10(np.maximum(x, 1e-12)), color=band_colors.get(fc, '#999'),
                        lw=1.2, label=f'{fc:g} Hz')
            top = 10 * np.log10(np.nanmax(e[r, :, :valid]) * 2)
            ax.set_ylim(top - 70, top + 3)
            ax.set_xlim(0, meta['validSeconds'])
            ax.set_title(f'{sid}: receiver {r} ({name}, {dist[r]:.1f} m)', fontsize=9)
            ax.grid(True, color='#e6e6e6', lw=.6)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=7)
        row[0].set_ylabel('dB / 2 ms', fontsize=8)
    axs[0, 0].legend(fontsize=7, frameon=False, ncol=4)
    axs[-1, 0].set_xlabel('time after emission (s)')
    axs[-1, 1].set_xlabel('time after emission (s)')
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build(root, out, figures=None):
    root, out = Path(root), Path(out)
    (out / 'data').mkdir(parents=True, exist_ok=True)
    report = mc.score(root, out / 'score')
    found = mc.discover(root)
    suite_path = root / 'suite.json'
    order = [s['id'] for s in json.loads(suite_path.read_text())['scenes']] if suite_path.exists() else sorted(found)
    scenes = []
    for sid in order:
        runs = found.get(sid, {})
        if 'gt' not in runs:
            continue
        data = scene_data(root, sid, runs, report['rows'])
        (out / 'data' / f'{sid}.js').write_text(
            'window.MC_SCENES = window.MC_SCENES || {};\nwindow.MC_SCENES[' + json.dumps(sid) + '] = ' +
            json.dumps(data, separators=(',', ':'), allow_nan=False) + ';\n')
        scenes.append(dict(id=sid, family=data['family'], variant=data['variant'], closed=data['closed'],
                           methods=[m['name'] for m in data['methods']]))
    conv = []
    for metric in ('level', 'etc', 'T30', 'complex'):
        name = f'convergence-{metric}.png'
        if convergence_figure(report, out / name, metric):
            conv.append(name)
    if scenes:
        sketch_figure(root, [s['id'] for s in scenes], out / 'scenes.png')
        gt_figure(root, [s['id'] for s in scenes], out / 'gt-echograms.png')
    index = dict(scenes=scenes, summary=summary_table(report), convergence=conv,
                 spreads=report['spreads'], missing=report['missing'],
                 colors={m: mc.method_color(m, {r['method'] for r in report['rows']})
                         for m in {r['method'] for r in report['rows']}})
    (out / 'data' / 'index.js').write_text('window.MC_INDEX = ' + json.dumps(index, allow_nan=False, default=mc._json) + ';\n')
    (out / 'index.html').write_text(HTML.replace('%FLOOR%', str(CURVE_FLOOR)))
    if figures:
        figures = Path(figures)
        figures.mkdir(parents=True, exist_ok=True)
        for name in conv + (['scenes.png', 'gt-echograms.png'] if scenes else []):
            (figures / name).write_bytes((out / name).read_bytes())
    print(f'viewer: {out / "index.html"} ({len(scenes)} scenes, methods '
          f'{sorted({m for s in scenes for m in s["methods"]})})')
    return out / 'index.html'


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Acoustic method comparison</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --ink:#1f1f1f; --muted:#666; --line:#e3e3e3; --bg:#fcfcfb; --panel:#ffffff; }
body { font: 13px/1.4 -apple-system, system-ui, sans-serif; color: var(--ink); background: var(--bg); margin: 0; }
header { padding: 10px 18px; border-bottom: 1px solid var(--line); display: flex; gap: 18px; align-items: baseline; }
header h1 { font-size: 16px; margin: 0; }
nav button { border: 1px solid var(--line); background: var(--panel); padding: 4px 10px; border-radius: 4px; cursor: pointer; }
nav button.on { border-color: #333; font-weight: 600; }
main { padding: 12px 18px; }
.row { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; }
.card h3 { font-size: 12px; margin: 0 0 6px; font-weight: 600; }
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 10px; }
select { font: inherit; padding: 2px 4px; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 3px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th { color: var(--muted); font-weight: 500; }
td.name { text-align: left; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.muted { color: var(--muted); }
#tip { position: fixed; pointer-events: none; background: #1f1f1f; color: #fff; padding: 4px 7px; border-radius: 4px;
       font-size: 12px; display: none; z-index: 10; white-space: pre; }
canvas { display: block; }
.legend span { margin-right: 12px; }
.scale { font-size: 11px; color: var(--muted); }
</style></head>
<body>
<header><h1>Acoustic method comparison</h1>
<nav><button data-page="summary" class="on">Summary</button> <button data-page="scene">Scenes</button>
<button data-page="convergence">Convergence</button></nav>
<span class="muted" id="status"></span></header>
<main>
<section id="page-summary"></section>
<section id="page-scene" hidden>
  <div class="controls">
    <label>Scene <select id="scene"></select></label>
    <label>Band <select id="band"></select></label>
    <label>Error scale ± <select id="errscale"><option>3</option><option selected>6</option><option>10</option><option>20</option></select> dB</label>
    <span class="muted">Click a receiver on any map to plot it.</span>
  </div>
  <div class="row">
    <div class="card"><h3>Scene (top-down at receiver height)</h3><canvas id="sketch" width="300" height="300"></canvas>
      <div class="legend scale"><span>&#9679; in view</span><span>&#9633; shadow</span><span>&#9733; source</span>
      <span>grey fill: geometry cut at receiver height; dashed: above/below</span></div></div>
    <div class="card"><h3>Band level, dB re free-field 1 m (shared scale)</h3><div class="row" id="levelmaps"></div>
      <div class="scale" id="levelscale"></div></div>
  </div>
  <div class="card" style="margin-top:12px"><h3>Level error, method − ground truth (dB)</h3><div class="row" id="errmaps"></div>
    <div class="scale" id="errscalebar"></div></div>
  <div class="row" style="margin-top:12px">
    <div class="card"><h3 id="etctitle">Energy-time curve</h3><canvas id="etc" width="560" height="260"></canvas></div>
    <div class="card"><h3>Schroeder decay curve</h3><canvas id="edc" width="560" height="260"></canvas></div>
  </div>
  <div class="row" style="margin-top:12px" id="complexrow">
    <div class="card"><h3>Complex pressure magnitude (dB)</h3><canvas id="cmag" width="560" height="220"></canvas></div>
    <div class="card"><h3>Complex pressure phase (rad)</h3><canvas id="cph" width="560" height="220"></canvas></div>
  </div>
</section>
<section id="page-convergence" hidden></section>
</main>
<div id="tip"></div>
<script src="data/index.js"></script>
<script>
const IDX = window.MC_INDEX, tip = document.getElementById('tip'), FLOOR = %FLOOR%;
const state = { scene: null, band: null, receiver: 0 };
const CIVIDIS = ['#00224e','#123570','#3b496c','#575d6d','#707173','#8a8678','#a59c74','#c3b369','#e1cc55','#fee838'];
const DIVERGE = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#efefef','#fddbc7','#f4a582','#d6604d','#b2182b'];
function lerpColor(stops, t) {
  t = Math.max(0, Math.min(1, t)); const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const a = parseInt(stops[i].slice(1), 16), b = parseInt(stops[i + 1].slice(1), 16);
  const c = [16, 8, 0].map(s => Math.round(((a >> s) & 255) * (1 - f) + ((b >> s) & 255) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function fmt(v, d = 2) { return v === null || v === undefined || !isFinite(v) ? '–' : (+v).toFixed(d); }
function showTip(e, text) { tip.style.display = 'block'; tip.textContent = text; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; }
function hideTip() { tip.style.display = 'none'; }

// ---------------------------------------------------------------- pages
function showPage(page) {
  document.querySelectorAll('nav button').forEach(x => x.classList.toggle('on', x.dataset.page === page));
  ['summary', 'scene', 'convergence'].forEach(p => document.getElementById('page-' + p).hidden = p !== page);
}
document.querySelectorAll('nav button').forEach(b => b.onclick = () => showPage(b.dataset.page));
// Deep links: #scene=<id>&band=<Hz>&r=<receiver> or #convergence.
const HASH = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=')));

function summaryPage() {
  const el = document.getElementById('page-summary');
  const bands = [...new Set(IDX.summary.flatMap(m => Object.keys(m.cells).map(k => k.split('|')[1])))].filter(b => b !== 'all').sort((a, b) => a - b);
  const metrics = [['level', 'Level error, dB RMS', 2], ['levelBias', 'Level bias, dB', 2], ['T30', 'T30 error, % RMS', 1],
                   ['EDT', 'EDT error, % RMS', 1], ['C50', 'C50 error, dB RMS', 2], ['onset', 'Onset error, ms RMS', 1],
                   ['etc', 'ETC distance, dB RMS', 2], ['complex', 'Complex error, relative L2', 3]];
  let html = '<p class="muted">Each method at its largest budget, pooled over scenes and seeds. Ground truth exists only in the bands shown; higher bands have none. ' +
    '<code>gt</code> against itself is the harness self-test (all zeros); <code>bem</code> is the single-frequency BEM reference (complex only).</p>';
  for (const [metric, title, d] of metrics) {
    const any = IDX.summary.some(m => Object.keys(m.cells).some(k => k.startsWith(metric + '|')));
    if (!any) continue;
    html += `<div class="card" style="margin-bottom:12px"><h3>${title}</h3><table><tr><th class="name">method</th><th>budget</th><th>cost s/scene</th>`;
    for (const o of ['open', 'closed']) for (const g of ['view', 'shadow']) html += `<th>${o} ${g}</th>`;
    for (const b of bands) html += `<th>${+b} Hz</th>`;
    html += '</tr>';
    for (const m of IDX.summary) {
      html += `<tr><td class="name"><span class="sw" style="background:${m.color}"></span>${m.method}</td><td>${m.budget}</td><td>${fmt(m.cost, 3)}</td>`;
      for (const o of ['open', 'closed']) for (const g of ['view', 'shadow']) html += `<td>${fmt(m.cells[`${metric}|all|${o}|${g}`], d)}</td>`;
      for (const b of bands) html += `<td>${fmt(m.cells[`${metric}|${b}|all|all`], d)}</td>`;
      html += '</tr>';
    }
    html += '</table></div>';
  }
  if (IDX.missing.length) html += `<p class="muted">No ground truth yet: ${IDX.missing.map(m => m.scene).join(', ')}</p>`;
  html += '<div class="card"><h3>Suite</h3><img src="scenes.png" style="max-width:100%"></div>';
  el.innerHTML = html;
}

function convergencePage() {
  const el = document.getElementById('page-convergence');
  let html = '<p class="muted">Error of each method against ground truth vs its budget (budgetValue), pooled over scenes and bands with ground truth. Line: mean over seeds; shaded: range over seeds, so a method whose seeds diverge shows a widening band.</p>';
  for (const name of IDX.convergence) html += `<div class="card" style="margin-bottom:12px"><img src="${name}" style="max-width:100%"></div>`;
  if (IDX.spreads.length) {
    html += '<div class="card"><h3>Seed spread (RMS over receivers): level SD in dB / energy-weighted histogram SD</h3><table><tr><th class="name">method</th><th>budget</th><th>scene</th><th>band</th><th>group</th><th>seeds</th><th>level SD</th><th>hist. SD</th></tr>';
    for (const s of IDX.spreads.filter(s => s.group !== 'all'))
      html += `<tr><td class="name"><span class="sw" style="background:${IDX.colors[s.method] || '#999'}"></span>${s.method}</td><td>${s.budget}</td><td>${s.scene}</td><td>${s.band}</td><td>${s.group}</td><td>${s.seeds}</td><td>${fmt(s.levelSD)}</td><td>${fmt(s.histogramSD, 3)}</td></tr>`;
    html += '</table></div>';
  }
  el.innerHTML = html;
}

// ---------------------------------------------------------------- scene page
const sceneSel = document.getElementById('scene'), bandSel = document.getElementById('band'), errSel = document.getElementById('errscale');
IDX.scenes.forEach(s => sceneSel.add(new Option(`${s.id}  (${s.methods.join(', ')})`, s.id)));
sceneSel.onchange = () => loadScene(sceneSel.value);
bandSel.onchange = () => { state.band = bandSel.value; drawScene(); };
errSel.onchange = () => drawScene();

function loadScene(id, receiver = 0) {
  const ready = () => { state.scene = window.MC_SCENES[id]; state.receiver = receiver;
    const keep = bandSel.value; bandSel.innerHTML = '';
    state.scene.bands.forEach(b => bandSel.add(new Option(b + ' Hz', String(b))));
    bandSel.value = state.scene.bands.map(String).includes(keep) ? keep : String(state.scene.bands[0]);
    state.band = bandSel.value; drawScene(); };
  if (window.MC_SCENES && window.MC_SCENES[id]) return ready();
  const s = document.createElement('script'); s.src = `data/${id}.js`; s.onload = ready; document.head.appendChild(s);
}

function frame(S, w, h, pad = 8) {
  const xs = S.receivers.map(p => p[0]).concat(S.walls.flatMap(r => [r.x0, r.x1]), [S.source[0]]);
  const ys = S.receivers.map(p => p[1]).concat(S.walls.flatMap(r => [r.y0, r.y1]), [S.source[1]]);
  const rx = S.receivers.map(p => p[0]), ry = S.receivers.map(p => p[1]);
  // Frame on receivers + source with a margin; geometry outside is clipped (roofs can be large).
  let x0 = Math.min(...rx, S.source[0]) - 1, x1 = Math.max(...rx, S.source[0]) + 1;
  let y0 = Math.min(...ry, S.source[1]) - 1, y1 = Math.max(...ry, S.source[1]) + 1;
  const sc = Math.min((w - 2 * pad) / (x1 - x0), (h - 2 * pad) / (y1 - y0));
  const ox = pad + ((w - 2 * pad) - sc * (x1 - x0)) / 2, oy = pad + ((h - 2 * pad) - sc * (y1 - y0)) / 2;
  return { X: x => ox + (x - x0) * sc, Y: y => h - (oy + (y - y0) * sc), sc };
}
function drawGeometry(ctx, S, F) {
  for (const r of S.walls) {
    const x = F.X(r.x0), y = F.Y(r.y1), w = (r.x1 - r.x0) * F.sc, h = (r.y1 - r.y0) * F.sc;
    if (r.cut) { ctx.fillStyle = '#a8a8a8'; ctx.fillRect(x, y, w, h); }
    else { ctx.setLineDash([3, 3]); ctx.strokeStyle = '#b5b5b5'; ctx.strokeRect(x, y, w, h); ctx.setLineDash([]); }
  }
}
function cellSize(S, F) {
  if (S.spacing) return Math.max(4, S.spacing * F.sc * .92);
  let best = 1e9; const P = S.receivers;
  for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) best = Math.min(best, Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]));
  return Math.max(4, best * F.sc * .92);
}
function hitReceiver(S, F, e, canvas) {
  const b = canvas.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top;
  let best = -1, bd = 1e9;
  S.receivers.forEach((p, i) => { const d = Math.hypot(F.X(p[0]) - mx, F.Y(p[1]) - my); if (d < bd) { bd = d; best = i; } });
  return bd < Math.max(10, cellSize(S, F)) ? best : -1;
}
function drawSketch() {
  const S = state.scene, c = document.getElementById('sketch'), ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height); const F = frame(S, c.width, c.height);
  drawGeometry(ctx, S, F);
  S.receivers.forEach((p, i) => {
    const x = F.X(p[0]), y = F.Y(p[1]);
    ctx.lineWidth = i === state.receiver ? 2.5 : 1.2;
    if (S.shadow[i]) { ctx.strokeStyle = '#D55E00'; ctx.strokeRect(x - 4, y - 4, 8, 8); }
    else { ctx.fillStyle = '#0072B2'; ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill(); }
    if (i === state.receiver) { ctx.strokeStyle = '#000'; ctx.beginPath(); ctx.arc(x, y, 8, 0, 7); ctx.stroke(); }
  });
  star(ctx, F.X(S.source[0]), F.Y(S.source[1]));
  c.onmousemove = e => { const i = hitReceiver(S, F, e, c); i >= 0 ? showTip(e, `receiver ${i} (${S.shadow[i] ? 'shadow' : 'in view'})`) : hideTip(); };
  c.onmouseleave = hideTip; c.onclick = e => { const i = hitReceiver(S, F, e, c); if (i >= 0) { state.receiver = i; drawScene(); } };
}
function star(ctx, x, y) {
  ctx.fillStyle = '#000'; ctx.beginPath();
  for (let k = 0; k < 10; k++) { const r = k % 2 ? 3.5 : 8, a = -Math.PI / 2 + k * Math.PI / 5; ctx.lineTo(x + r * Math.cos(a), y + r * Math.sin(a)); }
  ctx.closePath(); ctx.fill();
}
function mapCanvas(parent, title, values, colorOf, label) {
  const S = state.scene, wrap = document.createElement('div'), c = document.createElement('canvas');
  c.width = 230; c.height = 230; wrap.innerHTML = `<div class="scale">${title}</div>`; wrap.appendChild(c); parent.appendChild(wrap);
  const ctx = c.getContext('2d'), F = frame(S, c.width, c.height), cs = cellSize(S, F);
  drawGeometry(ctx, S, F);
  S.receivers.forEach((p, i) => {
    const v = values[i]; const x = F.X(p[0]) - cs / 2, y = F.Y(p[1]) - cs / 2;
    ctx.fillStyle = v === null || v === undefined ? '#ffffff' : colorOf(v); ctx.fillRect(x, y, cs, cs);
    if (S.shadow[i]) { ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, y + cs); ctx.lineTo(x + cs, y); ctx.stroke(); }
    if (i === state.receiver) { ctx.strokeStyle = '#000'; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, cs + 2, cs + 2); }
  });
  star(ctx, F.X(S.source[0]), F.Y(S.source[1]));
  c.onmousemove = e => { const i = hitReceiver(S, F, e, c); i >= 0 ? showTip(e, label(i)) : hideTip(); };
  c.onmouseleave = hideTip; c.onclick = e => { const i = hitReceiver(S, F, e, c); if (i >= 0) { state.receiver = i; drawScene(); } };
}
function drawMaps() {
  const S = state.scene, band = state.band, lm = document.getElementById('levelmaps'), em = document.getElementById('errmaps');
  lm.innerHTML = ''; em.innerHTML = '';
  const gt = (S.levels.gt || {})[band] || [];
  const all = S.methods.flatMap(m => ((S.levels[m.name] || {})[band] || [])).filter(v => v !== null);
  const hi = Math.ceil(Math.max(...gt.filter(v => v !== null))), lo = Math.floor(Math.min(...gt.filter(v => v !== null))) - 3;
  const lev = v => lerpColor(CIVIDIS, (v - lo) / Math.max(1, hi - lo));
  const E = +errSel.value, err = v => lerpColor(DIVERGE, (v + E) / (2 * E));
  for (const m of S.methods) {
    const vals = (S.levels[m.name] || {})[band];
    if (!vals || vals.every(v => v === null)) continue;
    mapCanvas(lm, `<span class="sw" style="background:${m.color}"></span>${m.name} ${m.budget}`, vals, lev,
      i => `${m.name} receiver ${i}${S.shadow[i] ? ' (shadow)' : ''}\nlevel ${fmt(vals[i])} dB\nGT ${fmt(gt[i])} dB`);
    if (m.name !== 'gt') {
      const d = vals.map((v, i) => v === null || gt[i] === null ? null : v - gt[i]);
      mapCanvas(em, `<span class="sw" style="background:${m.color}"></span>${m.name} − gt`, d, err,
        i => `${m.name} − gt, receiver ${i}${S.shadow[i] ? ' (shadow)' : ''}\n${fmt(d[i])} dB`);
    }
  }
  document.getElementById('levelscale').innerHTML = scaleBar(CIVIDIS, lo, hi, 'dB') + ' &nbsp; diagonal stroke: shadow receiver; white: not provided';
  document.getElementById('errscalebar').innerHTML = scaleBar(DIVERGE, -E, E, 'dB (blue: too quiet, red: too loud)');
}
function scaleBar(stops, lo, hi, unit) {
  const g = stops.map((c, i) => `${c} ${(100 * i / (stops.length - 1)).toFixed(0)}%`).join(',');
  return `<span>${lo}</span> <span style="display:inline-block;width:160px;height:10px;vertical-align:-1px;background:linear-gradient(90deg,${g})"></span> <span>${hi} ${unit}</span>`;
}

// ---------------------------------------------------------------- line charts
function lineChart(canvas, series, opt) {
  const ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height, L = 48, R = 90, T = 10, B = 30;
  ctx.clearRect(0, 0, W, H);
  const xs = series.flatMap(s => s.x), ys = series.flatMap(s => s.y).filter(v => isFinite(v));
  if (!xs.length) { ctx.fillStyle = '#666'; ctx.fillText('not provided', L, H / 2); canvas.onmousemove = null; return; }
  const x0 = opt.x0 ?? Math.min(...xs), x1 = opt.x1 ?? Math.max(...xs);
  const y1 = opt.y1 ?? Math.max(...ys), y0 = opt.y0 ?? Math.min(...ys);
  const X = x => L + (x - x0) / (x1 - x0 || 1) * (W - L - R), Y = y => T + (y1 - y) / (y1 - y0 || 1) * (H - T - B);
  ctx.strokeStyle = '#e6e6e6'; ctx.fillStyle = '#666'; ctx.font = '11px system-ui'; ctx.lineWidth = 1;
  for (const t of ticks(y0, y1, 5)) { ctx.beginPath(); ctx.moveTo(L, Y(t)); ctx.lineTo(W - R, Y(t)); ctx.stroke(); ctx.fillText(String(+t.toFixed(3)), 4, Y(t) + 4); }
  for (const t of ticks(x0, x1, 7)) { ctx.fillText(String(+t.toFixed(3)), X(t) - 8, H - 12); }
  ctx.fillText(opt.xlabel, W - R - 60, H - 1); ctx.save(); ctx.translate(12, T + 60); ctx.rotate(-Math.PI / 2); ctx.restore();
  ctx.fillText(opt.ylabel, L + 4, T + 10);
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash || []); ctx.beginPath(); let pen = false;
    s.x.forEach((x, i) => { const y = s.y[i]; if (!isFinite(y) || x < x0 || x > x1) { pen = false; return; }
      const py = Y(Math.max(y0, Math.min(y1, y))); pen ? ctx.lineTo(X(x), py) : ctx.moveTo(X(x), py); pen = true; });
    ctx.stroke(); ctx.setLineDash([]);
    if (s.markers) s.x.forEach((x, i) => { if (isFinite(s.y[i])) { ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(x), Y(s.y[i]), 4, 0, 7); ctx.fill(); } });
  }
  // Direct labels at the right edge (legend), in ink, with a colour swatch.
  series.forEach((s, k) => { ctx.fillStyle = s.color; ctx.fillRect(W - R + 8, T + 6 + 16 * k, 10, 3); ctx.fillStyle = '#1f1f1f'; ctx.fillText(s.label, W - R + 22, T + 10 + 16 * k); });
  canvas.onmousemove = e => {
    const b = canvas.getBoundingClientRect(), mx = e.clientX - b.left; if (mx < L || mx > W - R) return hideTip();
    const x = x0 + (mx - L) / (W - L - R) * (x1 - x0);
    const lines = series.map(s => { let bi = 0, bd = 1e9; s.x.forEach((v, i) => { const d = Math.abs(v - x); if (d < bd) { bd = d; bi = i; } }); return `${s.label}: ${fmt(s.y[bi])}`; });
    showTip(e, `${opt.xlabel} ${fmt(x, 3)}\n` + lines.join('\n'));
  };
  canvas.onmouseleave = hideTip;
}
function ticks(a, b, n) { const s = Math.pow(10, Math.floor(Math.log10((b - a) / n || 1))); const st = [1, 2, 5, 10].map(k => k * s).find(k => (b - a) / k <= n) || s * 10;
  const out = []; for (let t = Math.ceil(a / st) * st; t <= b + 1e-9; t += st) out.push(t); return out; }

function drawCurves() {
  const S = state.scene, band = state.band, r = state.receiver;
  document.getElementById('etctitle').textContent = `Energy-time curve, receiver ${r} (${S.shadow[r] ? 'shadow' : 'in view'}), ${band} Hz, ${S.curveBin} ms bins`;
  const etc = [], edc = [];
  for (const m of S.methods) {
    const c = (S.curves[m.name] || {})[band]; if (!c) continue;
    const y = c[r].map(v => v / 10), x = y.map((_, i) => i * S.curveBin / 1000);
    etc.push({ x, y, color: m.color, label: m.name });
    const e = y.map(v => v <= FLOOR + .05 ? 0 : Math.pow(10, v / 10)); let acc = 0;   // floor bins hold no energy const rev = e.map((_, i) => e[e.length - 1 - i]).map(v => acc += v).reverse();
    edc.push({ x, y: rev.map(v => 10 * Math.log10(v / rev[0])), color: m.color, label: m.name });
  }
  const top = Math.max(...etc.flatMap(s => s.y));
  lineChart(document.getElementById('etc'), etc, { xlabel: 'time (s)', ylabel: 'dB re free-field 1 m', y0: top - 70, y1: top + 3 });
  lineChart(document.getElementById('edc'), edc, { xlabel: 'time (s)', ylabel: 'EDC (dB)', y0: -60, y1: 0 });
  const cm = [], cp = [];
  const g = S.complex.gt;
  for (const m of S.methods) {
    const c = S.complex[m.name]; if (!c) continue;
    const sel = g ? c.hz.map((f, i) => [f, i]).filter(([f]) => g.hz.some(h => Math.abs(h - f) < 1e-6)) : c.hz.map((f, i) => [f, i]);
    const x = sel.map(([f]) => f), re = sel.map(([, i]) => c.re[r][i]), im = sel.map(([, i]) => c.im[r][i]);
    cm.push({ x, y: re.map((v, k) => 20 * Math.log10(Math.hypot(v, im[k]))), color: m.color, label: m.name, markers: x.length < 40 });
    cp.push({ x, y: re.map((v, k) => Math.atan2(im[k], v)), color: m.color, label: m.name, markers: x.length < 40 });
  }
  document.getElementById('complexrow').style.display = cm.length ? '' : 'none';
  if (cm.length) { lineChart(document.getElementById('cmag'), cm, { xlabel: 'Hz', ylabel: 'dB' });
    lineChart(document.getElementById('cph'), cp, { xlabel: 'Hz', ylabel: 'rad', y0: -Math.PI, y1: Math.PI }); }
}
function drawScene() { if (!state.scene) return; drawSketch(); drawMaps(); drawCurves(); }

summaryPage(); convergencePage();
document.getElementById('status').textContent = `${IDX.scenes.length} scenes · methods: ${[...new Set(IDX.scenes.flatMap(s => s.methods))].join(', ')}`;
if ('convergence' in HASH) showPage('convergence');
if (IDX.scenes.length) {
  const first = HASH.scene && IDX.scenes.some(s => s.id === HASH.scene) ? HASH.scene : IDX.scenes[0].id;
  if (HASH.scene) { showPage('scene'); if (HASH.band) bandSel.add(new Option(HASH.band, HASH.band)), bandSel.value = HASH.band; }
  sceneSel.value = first; loadScene(first, HASH.r ? +HASH.r : 0);
}
</script></body></html>
"""
