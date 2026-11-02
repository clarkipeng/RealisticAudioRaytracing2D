"""Build the denoiser report, tables and figures from tools/dlss_train.py outputs.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY docs/benchmarks/denoise-model-2026-11-02/analyze.py [.context/audio-dlss-model]

Reads baselines.json, selection.json, runs/*/metrics.json, test.json and
timing.json; writes report.json, tables.md and fig*.png next to this file.
Needs NumPy and Matplotlib.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_train as dt  # noqa: E402

V = dt.VARIANTS
SHORT = {v: f'1/{f}' for v, f in zip(V, dt.FRACTIONS)}
# Validated categorical palette (dataviz reference, light surface), fixed order; markers double-encode.
C = dict(raw='#2a78d6', linear='#eb6834', model='#1baf7a', mean4='#eda100', directR='#e87ba4')
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
ABLATIONS = ['selected'] + list(dt.ABLATIONS)
TIMING_BATCH = '1'       # conservative: one pair per call, single CPU thread


def err(summary, family='all'):
    return [summary[family]['budgets'][v]['error'] for v in V]


def seeds_of(models, name):
    return [m for k, m in sorted(models.items()) if k.rsplit('-s', 1)[0] == name]


def seed_stats(summaries, family='all', key='error'):
    """Mean, min and max over initialisation seeds per budget."""
    vals = np.array([[s[family]['budgets'][v][key] for v in V] for s in summaries])
    return dict(mean=vals.mean(0).tolist(), low=vals.min(0).tolist(), high=vals.max(0).tolist(), seeds=len(vals))


def loglog(x, y, at):
    """Log-log interpolation of a measured cost/error curve (estimate between points)."""
    lx, ly = np.log(x), np.log(y)
    if at <= x[0]:
        return None
    return float(np.exp(np.interp(np.log(at), lx, ly))) if at <= x[-1] else None


def matched(seconds, base, model_err, inference_ms):
    """Compare the model at budget i (solve + inference) with a baseline curve in time."""
    cost = [1e3 * seconds[v] for v in V]
    rows = []
    for i, v in enumerate(V):
        total = cost[i] + inference_ms
        nxt = i + 1 < len(V)
        rows.append(dict(budget=v, modelMs=total, modelError=model_err[i],
                         baselineInterpolated=loglog(cost, base, total),
                         nextBudgetMs=cost[i + 1] if nxt else None, nextBudgetError=base[i + 1] if nxt else None,
                         beatsNextBudget=(model_err[i] < base[i + 1] and total <= cost[i + 1]) if nxt else None))
    return rows


def build(out):
    load = lambda name: json.loads((out / name).read_text())
    base, selection, test, timing = load('baselines.json'), load('selection.json'), load('test.json'), load('timing.json')
    runs = {p.name: json.loads((p / 'metrics.json').read_text()) for p in sorted((out / 'runs').iterdir())}
    seconds = base['seconds']['test']
    inf = timing['models']['selected-s0']['cpu1'][TIMING_BATCH]['perPairMs']
    sel = seeds_of(test['models'], 'selected')
    model = seed_stats(sel)
    tb = test['baselines']
    report = dict(
        datasetSha256=test['datasetSha256'], selected=selection['selected'], spec=selection['spec'],
        selectionRule=selection['rule'], evaluatedAt=test['evaluatedAt'], seconds=seconds, gains=base['gains'],
        sweep={n: dict(validation=selection['validation'][n], score=selection['scores'][n],
                       parameters=runs[n]['parameters'], macs=runs[n]['macs'], bestEpoch=runs[n]['bestEpoch'],
                       trainSeconds=runs[n]['trainSeconds']) for n in selection['validation']},
        test=dict(baselines=tb, model=model,
                  modelTails={k: seed_stats(sel, key=k) for k in ['over1', 'over3', 'level']},
                  families={f: dict(model=seed_stats(sel, f), **{b: err(tb[b], f) for b in tb})
                            for f in tb['raw'] if f != 'all'},
                  ablations={n: seed_stats(seeds_of(test['models'], n)) for n in ABLATIONS if seeds_of(test['models'], n)},
                  ablationFamilies={n: {f: seed_stats(seeds_of(test['models'], n), f) for f in tb['raw'] if f != 'all'}
                                    for n in ABLATIONS if seeds_of(test['models'], n)}),
        validation=dict(baselines=base['splits']['validation'],
                        ablations={n: seed_stats([r['validation'] for k, r in runs.items() if k.rsplit('-s', 1)[0] == n])
                                   for n in ABLATIONS if any(k.rsplit('-s', 1)[0] == n for k in runs)}),
        train=dict(baselines=base['splits']['train'],
                   ablations={n: seed_stats([r['train'] for k, r in runs.items() if k.rsplit('-s', 1)[0] == n])
                              for n in ABLATIONS if any(k.rsplit('-s', 1)[0] == n for k in runs)}),
        inferenceMsPerPair=inf, timing=timing,
        matchedTime=dict(raw=matched(seconds, err(tb['raw']), model['mean'], inf),
                         linear=matched(seconds, err(tb['linear']), model['mean'], inf)),
        curves={k: [dict(epoch=h['epoch'], val=h['val']) for h in runs[k]['history'] if 'val' in h]
                for k in runs if k.endswith('-s0')})
    return report


def tables(r):
    t, out = r['test'], []
    fmt = lambda x: 'n/a' if x is None else f'{x:.0f}%'
    rng = lambda s, i: f'{s["mean"][i]:.1f}% ({s["low"][i]:.1f}–{s["high"][i]:.1f})'
    out += ['### Candidates (validation = corridor family; selection on validation only)', '',
            '| Candidate | Parameters | MACs / pair | ' + ' | '.join(SHORT.values()) + ' | Score | Best epoch |',
            '|---|---:|---:|' + '---:|' * len(V) + '---:|---:|']
    for n, s in sorted(r['sweep'].items(), key=lambda kv: kv[1]['score']):
        mark = ' **(selected)**' if n == r['selected'] else ''
        out.append(f'| {n}{mark} | {s["parameters"]:,} | {s["macs"]:,} | ' + ' | '.join(f'{s["validation"][v]:.1f}%' for v in V) +
                   f' | {s["score"]:.3f} | {s["bestEpoch"]} |')
    vb = r['validation']['baselines']
    out.append('| *raw / linear gains* | | | ' + ' | '.join(f'{a:.0f}% / {b:.1f}%' for a, b in zip(err(vb['raw']), err(vb['linear']))) + ' | | |')
    b = t['baselines']
    out += ['', '### Test split at matched time (early field; canopy family + doorways wider than 1.5 m)', '',
            '| Budget | Solve ms | Raw | Raw, 4× rays | Direct + R only | Linear gains | Linear gains, 4× rays | Mean of 4 seeds | Network (3 seeds, range) | Network ms |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for i, v in enumerate(V):
        nxt = lambda k: f'{err(b[k])[i + 1]:.{1 if k == "linear" else 0}f}%' if i + 1 < len(V) else 'n/a'
        out.append(f'| {SHORT[v]} | {1e3 * r["seconds"][v]:.2f} | {err(b["raw"])[i]:.0f}% | {nxt("raw")} | {err(b["directR"])[i]:.0f}% | '
                   f'{err(b["linear"])[i]:.1f}% | {nxt("linear")} | {err(b["mean4"])[i]:.0f}% | {rng(t["model"], i)} | '
                   f'{r["matchedTime"]["raw"][i]["modelMs"]:.2f} |')
    out += ['', '### Test split per family', '',
            '| Family | Budget | Raw | Direct + R | Linear gains | Network |', '|---|---|---:|---:|---:|---:|']
    for f, e in t['families'].items():
        for i, v in enumerate(V):
            out.append(f'| {f} | {SHORT[v]} | {e["raw"][i]:.0f}% | {e["directR"][i]:.0f}% | {e["linear"][i]:.1f}% | {rng(e["model"], i)} |')
    out += ['', '### Test split tails and level', '',
            '| Budget | > 1× RMS raw / linear / network | > 3× RMS raw / linear / network | Energy ratio raw / linear / network |',
            '|---|---:|---:|---:|']
    for i, v in enumerate(V):
        g = lambda k, key: b[k]['all']['budgets'][v][key]
        mt = t['modelTails']
        out.append(f'| {SHORT[v]} | {g("raw", "over1"):.1f}% / {g("linear", "over1"):.1f}% / {mt["over1"]["mean"][i]:.1f}% | '
                   f'{g("raw", "over3"):.2f}% / {g("linear", "over3"):.2f}% / {mt["over3"]["mean"][i]:.2f}% | '
                   f'{g("raw", "level"):.2f} / {g("linear", "level"):.2f} / {mt["level"]["mean"][i]:.2f} |')
    out += ['', '### Ablations (mean of 3 initialisation seeds; range in brackets)', '',
            '| Variant | Validation ' + ' / '.join(SHORT.values()) + ' | Test ' + ' / '.join(SHORT.values()) + ' |', '|---|---|---|']
    for n in ABLATIONS:
        if n in t['ablations']:
            va, te = r['validation']['ablations'][n], t['ablations'][n]
            out.append(f'| {n} | ' + ' / '.join(f'{x:.1f}' for x in va['mean']) + ' | ' +
                       ' / '.join(f'{x:.1f} ({lo:.1f}–{hi:.1f})' for x, lo, hi in zip(te['mean'], te['low'], te['high'])) + ' |')
    for k, label in [('linear', 'per-family linear gains'), ('linearSlots', 'linear gains on slot ranges'), ('linearSummed', 'linear gains on early sum + walk'), ('raw', 'raw early sum')]:
        out.append(f'| *{label}* | ' + ' / '.join(f'{x:.1f}' for x in err(vb[k])) + ' | ' + ' / '.join(f'{x:.1f}' for x in err(b[k])) + ' |')
    out += ['', '### Inference per pair (median of 200 calls, host tensors in and out, features included)', '',
            '| Model | Parameters | MACs / pair | CPU 1 thread, batch 1 / 4 / 16 / 64 (µs per pair) | MPS, batch 1 / 4 / 16 / 64 (µs per pair) |',
            '|---|---:|---:|---|---|']
    for n, m in r['timing']['models'].items():
        cell = lambda dev: ' / '.join(f'{1e3 * m[dev][k]["perPairMs"]:.1f}' for k in ['1', '4', '16', '64']) if dev in m else 'n/a'
        out.append(f'| {n} | {m["parameters"]:,} | {m["macs"]:,} | {cell("cpu1")} | {cell("mps")} |')
    return '\n'.join(out) + '\n'


def style(plt):
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                         'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.edgecolor': GRID, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                         'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'lines.linewidth': 2})


def figures(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)
    t, b = r['test'], r['test']['baselines']
    cost = [1e3 * r['seconds'][v] for v in V]
    inf = r['inferenceMsPerPair']

    # Fig 1: error vs measured solve + readback (+ inference) time, test split.
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.axhline(100, color=MUTED, lw=1, ls=':')
    ax.text(cost[0], 104, 'predict silence = 100%', color=MUTED, fontsize=8)
    ax.plot(cost, err(b['raw']), marker='o', color=C['raw'], label='raw, one seed (more rays)')
    ax.plot([4 * c for c in cost], err(b['mean4']), marker='s', color=C['mean4'], label='mean of 4 seeds (4 solves)')
    ax.plot(cost, err(b['directR']), marker='v', color=C['directR'], label='direct + first reflections only')
    ax.plot(cost, err(b['linear']), marker='D', color=C['linear'], label='per-family linear gains (fit on train)')
    m = t['model']
    mc = [c + inf for c in cost]
    ax.errorbar(mc, m['mean'], yerr=[np.subtract(m['mean'], m['low']), np.subtract(m['high'], m['mean'])],
                marker='^', color=C['model'], capsize=3, label=f'network (+{1e3 * inf:.0f} µs inference, 3 seeds)')
    for c, v in zip(cost, V):
        ax.annotate(SHORT[v], (c, err(b['raw'])[V.index(v)]), textcoords='offset points', xytext=(4, 4), color=MUTED, fontsize=8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_yticks([15, 20, 30, 50, 100, 200]); ax.set_yticklabels(['15', '20', '30', '50', '100', '200'])
    ax.minorticks_off()
    ax.set_xticks([1.5, 2, 3, 5, 10, 20]); ax.set_xticklabels(['1.5', '2', '3', '5', '10', '20'])
    ax.set_xlabel('Measured solve + readback (+ inference) time per pair (ms)')
    ax.set_ylabel('Early-field error vs label, test split (%)')
    ax.set_title('Network vs more rays at matched time (held-out canopy + wide doorways)', color=INK, loc='left')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout(); fig.savefig(HERE / 'fig1-matched-time.png', dpi=160); plt.close(fig)

    # Fig 2: per family, error vs budget.
    fams = list(t['families'])
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.8), sharey=True)
    x = np.arange(len(V))
    for ax, f in zip(np.atleast_1d(axes), fams):
        e = t['families'][f]
        ax.plot(x, e['raw'], marker='o', color=C['raw'], label='raw')
        ax.plot(x, e['directR'], marker='v', color=C['directR'], label='direct + R')
        ax.plot(x, e['linear'], marker='D', color=C['linear'], label='linear gains')
        ax.errorbar(x, e['model']['mean'], yerr=[np.subtract(e['model']['mean'], e['model']['low']),
                                                 np.subtract(e['model']['high'], e['model']['mean'])],
                    marker='^', color=C['model'], capsize=3, label='network')
        ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(SHORT.values()); ax.minorticks_off()
        ax.set_yticks([10, 20, 50, 100, 200]); ax.set_yticklabels(['10', '20', '50', '100', '200'])
        ax.set_title(f'{f} (test)', color=INK, loc='left'); ax.set_xlabel('Input budget')
    np.atleast_1d(axes)[0].set_ylabel('Early-field error (%)'); np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / 'fig2-families.png', dpi=160); plt.close(fig)

    # Fig 3: ablations on the test split, grouped by budget.
    names = [n for n in ABLATIONS if n in t['ablations']]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    width = .8 / (len(names) + 1)
    palette = ['#1baf7a', '#2a78d6', '#eb6834', '#eda100', '#e87ba4', '#52514e', '#c3c2b7']   # one per variant, never cycled
    for j, n in enumerate(names):
        s = t['ablations'][n]
        pos = x + (j - len(names) / 2) * width
        ax.bar(pos, s['mean'], width * .92, color=palette[j], label=n, hatch='///' if n == 'shuffled-labels' else None,
               yerr=[np.subtract(s['mean'], s['low']), np.subtract(s['high'], s['mean'])], capsize=2, ecolor=MUTED)
    for i in range(len(V)):
        ax.plot([x[i] - .45, x[i] + .4], [err(b['linear'])[i]] * 2, color=INK, lw=1.2, ls='--', label='linear gains' if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(SHORT.values()); ax.grid(axis='x', visible=False)
    ax.set_xlabel('Input budget'); ax.set_ylabel('Early-field error, test split (%)')
    ax.set_title('Ablations (bars: mean of 3 seeds, whiskers: range)', color=INK, loc='left')
    ax.legend(fontsize=7.5, ncol=4, loc='upper right'); ax.set_ylim(0, max(max(t['ablations'][n]['high']) for n in names) * 1.25)
    fig.tight_layout(); fig.savefig(HERE / 'fig3-ablations.png', dpi=160); plt.close(fig)

    # Fig 4: inference time per pair vs batch.
    tm = r['timing']['models']['selected-s0']
    batches = [int(k) for k in tm['cpu1']]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for dev, color, marker, label in [('cpu1', C['raw'], 'o', 'CPU, 1 thread'), ('mps', C['linear'], 's', 'MPS (incl. upload + readback)')]:
        if dev in tm:
            ax.plot(batches, [1e3 * tm[dev][str(k)]['perPairMs'] for k in batches], marker=marker, color=color, label=label)
    ax.axhline(1e3 * (cost[1] - cost[0]), color=MUTED, lw=1, ls=':')
    ax.text(batches[-1], 1e3 * (cost[1] - cost[0]) * 1.1, '1/256 → 1/64 extra solve time', ha='right', color=MUTED, fontsize=8)
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.minorticks_off()
    ax.set_xticks(batches); ax.set_xticklabels([str(k) for k in batches])
    ax.set_xlabel('Pairs per call'); ax.set_ylabel('Inference time per pair (µs)')
    ax.set_title('Selected network inference cost', color=INK, loc='left'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / 'fig4-inference.png', dpi=160); plt.close(fig)

    # Fig 5: validation curves (overfitting and shuffle control).
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, color in [('selected-s0', C['model']), ('shuffled-labels-s0', C['raw']), ('summed-s0', C['mean4'])]:
        if name in r['curves']:
            h = r['curves'][name]
            ax.plot([p['epoch'] for p in h], [np.exp(np.mean(np.log(list(p['val'].values())))) for p in h],
                    color=color, label=name.replace('-s0', ''))
    lin = r['validation']['baselines']['linear']['all']['budgets']
    ax.axhline(np.exp(np.mean(np.log([lin[v]['error'] for v in V]))), color=C['linear'], ls='--', lw=1.2, label='linear gains')
    ax.set_yscale('log'); ax.minorticks_off()
    ax.set_yticks([40, 50, 70, 100, 200, 300]); ax.set_yticklabels(['40', '50', '70', '100', '200', '300'])
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation error, geometric mean over budgets (%)')
    ax.set_title('Training curves (corridor validation family)', color=INK, loc='left'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / 'fig5-training.png', dpi=160); plt.close(fig)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else dt.OUT
    report = build(out)
    (HERE / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    (HERE / 'tables.md').write_text(tables(report))
    figures(report)
    print(tables(report))


if __name__ == '__main__':
    main()
