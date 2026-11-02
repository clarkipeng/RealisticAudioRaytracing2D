"""Reproduce the audio-DLSS denoiser pilot report and figures from a packed dataset.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY docs/benchmarks/denoise-pilot-2026-10-29/analyze.py \
        .context/audio-dlss-data/datasets/denoise-pilot-v1

Writes report.json and fig*.png next to this file. Needs NumPy and Matplotlib.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_baselines as bl  # noqa: E402
import dlss_dataset as dd  # noqa: E402
import dlss_scenes as ds  # noqa: E402

FAMILY_ORDER = list(ds.FAMILIES)
# Validated categorical palette (dataviz reference, light surface), fixed order.
COLORS = dict(shoebox='#2a78d6', doorway='#eb6834', corridor='#1baf7a', canopy='#eda100', barriers='#e87ba4',
              furnished='#008300')
MARKERS = dict(shoebox='o', doorway='s', corridor='^', canopy='D', barriers='v', furnished='P')
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
TRUNCATION_LIMIT = .05   # FDTD scenes whose response changes more than 5% between 0.75 T and T are excluded


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float('nan')


def scene_metrics(bundle, data, mask, variants, neighbours):
    """Per-scene errors for one target definition (family mask over the F axis)."""
    label_runs = data['label_family_runs'][..., mask].sum(-1)          # [K, P, Q]
    k = label_runs.shape[0]
    target = label_runs.mean(0)
    variance = (np.abs(label_runs - target) ** 2).sum(0) / (k - 1)
    energy = np.sum(np.abs(target) ** 2)
    out = dict(scene=bundle['bundle_id'], family=bundle['family'], split=bundle['split'], receivers=bundle['receivers'],
               accepted=bundle['quality']['accepted'], validFraction=bundle['quality']['validFraction'],
               labelRelSE=float(np.sqrt(variance.sum() / k / energy)), budgets={})
    bands = bl.band_masks(data['frequencies'])
    for name in variants:
        runs = data['inputs'][name][..., mask]                             # [S, P, Q, F]
        totals = runs.sum(-1)
        s = totals.shape[0]
        seed_var = (np.abs(totals - totals.mean(0)) ** 2).sum(0) / (s - 1)
        spikes = bl.spike_scores(totals, target)
        family_err = np.abs(runs - data['label_family_runs'][..., mask].mean(0)) ** 2
        worst = family_err.argmax(-1)[spikes > 1]
        names = np.array(data['families'])[mask]
        row = dict(
            error2=bl.relative_error_squared(totals, target),
            inputRelSD=float(np.sqrt(seed_var.sum() / energy)),
            bandError2={b: bl.relative_error_squared(totals[:, :, m], target[:, m]) for b, m in bands.items()},
            spikeP50=float(np.percentile(spikes, 50)), spikeP99=float(np.percentile(spikes, 99)),
            spikeMax=float(spikes.max()), spikeOver1=float(np.mean(spikes > 1)), spikeOver3=float(np.mean(spikes > 3)),
            spikeFamilies={str(n): int(np.sum(worst == i)) for i, n in enumerate(names)},
            mean4Error2=bl.relative_error_squared(bl.seed_mean(totals, 4), target),
            mean2Error2=bl.relative_error_squared(bl.seed_mean(totals, 2), target),
            neighbourError2=bl.relative_error_squared(bl.neighbour_mean(totals, neighbours), target),
            clampError2=bl.relative_error_squared(bl.family_clamp(runs), target),
            seconds=float(np.mean(data['seconds'][name])))
        out['budgets'][name] = row
    out['labelSeconds'] = float(np.mean(data['seconds_label']))
    return out


def aggregate(rows, key, budget=None, sub=None):
    """Scene-balanced RMS of a squared relative error, as a percentage."""
    values = []
    for r in rows:
        v = r['budgets'][budget][key] if budget else r[key]
        values.append(v[sub] if sub else v)
    return 100 * float(np.sqrt(np.mean(values))) if values else float('nan')


def load_all(target):
    manifest, bundles, base = dd.open_dataset(target)
    out = []
    for bundle in bundles:
        data = dd.load_bundle(base, bundle)
        run = json.loads((base / bundle['tracer_run']['file']).read_text())
        data['families'] = run['families']
        data['frequencies'] = run['frequencies']
        data['seconds_label'] = data['seconds'].pop(bundle['descriptors']['label'])
        scene = json.loads((base / bundle['scene']['file']).read_text())
        out.append((bundle, data, scene))
    return manifest, out, base


def wave_bias(bundle, data, scene):
    """Rigid tracer label vs FDTD at the shared frequencies, whole map and LoS/shadow."""
    if 'wave' not in data or 'rigid_target' not in data:
        return None
    wave_hz = bundle['wave']['frequencies']
    index = [data['frequencies'].index(hz) for hz in wave_hz]
    runs = data['rigid_label_runs']                                          # [K, P, Q, F]
    early = runs[..., :7].sum(-1).mean(0)[:, index]
    total = runs.sum(-1).mean(0)[:, index]
    k = runs.shape[0]
    se = np.sqrt((np.abs(runs[..., :7].sum(-1) - runs[..., :7].sum(-1).mean(0)) ** 2).sum(0) / (k - 1) / k)[:, index]
    material = data['label_family_runs'][..., :7].sum(-1).mean(0)[:, index]
    wave = data['wave']
    los = np.array(scene['lineOfSight'], bool)
    out = dict(scene=bundle['bundle_id'], family=bundle['family'], split=bundle['split'],
               truncation=max(bundle['wave']['truncationChange075']), shadowReceivers=int((~los).sum()), rows=[])
    for j, hz in enumerate(wave_hz):
        def err(x, m=slice(None)):
            return float(np.linalg.norm(x[m, j] - wave[m, j]) / np.linalg.norm(wave[m, j])) if np.any(np.ones(len(wave))[m]) else None
        out['rows'].append(dict(hz=hz, truncation=bundle['wave']['truncationChange075'][j], early=err(early), total=err(total), earlyLos=err(early, los) if los.any() else None,
                                earlyShadow=err(early, ~los) if (~los).any() else None,
                                labelSE=float(np.linalg.norm(se[:, j]) / np.linalg.norm(wave[:, j])),
                                materialVsRigid=float(np.linalg.norm(material[:, j] - early[:, j]) / np.linalg.norm(early[:, j]))))
    return out


def throughput(base, bundles):
    """Wall-clock throughput per stage from the run logs (paced, shared machine)."""
    logs = base / 'logs'
    batches = []
    for folder in sorted(logs.glob('render-*')):
        res = folder / 'resources.json'
        if not res.exists():
            continue
        r = json.loads(res.read_text())
        jobs = [line.split('\t') for line in (folder / 'jobs.txt').read_text().splitlines() if line]
        def produced(job):   # The scene's accepted render records which batch produced it.
            record = Path(job[0]) / (job[1] if len(job) > 1 else 'tracer') / 'provenance.json'
            return record.exists() and json.loads(record.read_text())['batch'] == folder.name
        done = [j for j in jobs if produced(j)]
        batches.append(dict(batch=folder.name, variant=jobs[0][1] if len(jobs[0]) > 1 else 'tracer', scenes=len(done),
                            wallSeconds=r['wallSeconds'], pausedSeconds=r.get('pausedSeconds', 0), passed=r.get('passed'),
                            peakRssGiB=r['peakTreeRssBytes'] / 1024 ** 3))
    out = {}
    for variant in sorted({b['variant'] for b in batches}):
        rows = [b for b in batches if b['variant'] == variant]   # scenes counted by verified outputs
        wall = sum(b['wallSeconds'] - b['pausedSeconds'] for b in rows)
        n = sum(b['scenes'] for b in rows)
        out[variant] = dict(scenes=n, wallHours=wall / 3600, scenesPerHour=n / (wall / 3600) if wall else None,
                            batches=len(rows), peakRssGiB=max([b['peakRssGiB'] for b in rows] or [0]))
    fdtd = []
    for bundle in bundles:
        if bundle.get('wave'):
            run = json.loads((base / bundle['wave']['run']['file']).read_text())
            fdtd.append(dict(scene=bundle['bundle_id'], gpuSeconds=run['timing']['gpuSeconds'], wallSeconds=run['timing']['wallSeconds'],
                             cells=run['cells'], steps=run['timing']['steps'], gcell=run['timing']['gcellPerSecond'],
                             peakDriverGiB=run['timing']['peakDriverBytes'] / 1024 ** 3))
    if fdtd:
        out['fdtd'] = dict(scenes=len(fdtd), gpuHours=sum(f['gpuSeconds'] for f in fdtd) / 3600,
                           wallHours=sum(f['wallSeconds'] for f in fdtd) / 3600,
                           scenesPerHour=len(fdtd) / (sum(f['wallSeconds'] for f in fdtd) / 3600),
                           medianGcellPerSecond=float(np.median([f['gcell'] for f in fdtd])),
                           peakDriverGiB=max(f['peakDriverGiB'] for f in fdtd), runs=fdtd)
    return out, batches


def summarise(rows, variants, families):
    """Label noise over every rendered scene; input errors over accepted scenes only."""
    table = {}
    for family in families + ['all']:
        every = [r for r in rows if family == 'all' or r['family'] == family]
        sel = [r for r in every if r['accepted']]
        if not every:
            continue
        entry = dict(scenes=len(every), accepted=len(sel), labelRelSE=100 * rms([r['labelRelSE'] for r in every]),
                     labelRelSEMax=100 * max(r['labelRelSE'] for r in every),
                     validFraction=float(np.mean([r['validFraction'] for r in every])), budgets={})
        if not sel:
            table[family] = entry
            continue
        for v in variants:
            b = [r['budgets'][v] for r in sel]
            entry['budgets'][v] = dict(
                error=aggregate(sel, 'error2', v), inputRelSD=100 * rms([x['inputRelSD'] for x in b]),
                noiseRatio=rms([x['inputRelSD'] for x in b]) / rms([r['labelRelSE'] for r in sel]),
                bands={k: aggregate(sel, 'bandError2', v, k) for k in bl.BANDS},
                spikeP99=float(np.median([x['spikeP99'] for x in b])), spikeMax=float(max(x['spikeMax'] for x in b)),
                spikeOver1=100 * float(np.mean([x['spikeOver1'] for x in b])),
                spikeOver3=100 * float(np.mean([x['spikeOver3'] for x in b])),
                spikeFamilies={f: int(sum(x['spikeFamilies'].get(f, 0) for x in b)) for f in b[0]['spikeFamilies']},
                mean2=aggregate(sel, 'mean2Error2', v), mean4=aggregate(sel, 'mean4Error2', v),
                neighbour=aggregate(sel, 'neighbourError2', v), clamp=aggregate(sel, 'clampError2', v),
                seconds=float(np.mean([x['seconds'] for x in b])))
        entry['labelSeconds'] = float(np.mean([r['labelSeconds'] for r in sel]))
        table[family] = entry
    return table


def style(plt):
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                         'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.edgecolor': GRID, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                         'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'lines.linewidth': 2})


def figures(report, variants, fractions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)
    early = report['early']['families']
    total = report['total']['families']
    fams = [f for f in FAMILY_ORDER if f in early and early[f]['budgets']]
    x = [1 / f for f in fractions]

    # Fig 1: input error vs budget, early target and total target, with label noise floor.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, table, title in [(axes[0], early, 'Early-field label (primary)'), (axes[1], total, 'Total label incl. residual walk')]:
        for f in fams:
            y = [table[f]['budgets'][v]['error'] for v in variants]
            ax.plot(x, y, marker=MARKERS[f], color=COLORS[f], markersize=6, label=f)
            ax.plot(x[-1] * 2.5, table[f]['labelRelSE'], marker=MARKERS[f], color=COLORS[f], markersize=5, alpha=.6, ls='none')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(x + [x[-1] * 2.5]); ax.set_xticklabels([f'1/{f}' for f in fractions] + ['label\nSE'])
        ax.minorticks_off()
        ax.set_title(title, color=INK, loc='left')
    axes[0].set_ylabel('Complex relative L2 error vs label (%)')
    axes[1].legend(loc='upper right', ncol=2)
    fig.supxlabel('Input budget (fraction of one label run); label SE = standard error of the 8-seed label', color=MUTED, fontsize=9.5)
    fig.tight_layout(); fig.savefig(HERE / 'fig1-error-vs-budget.png', dpi=160); plt.close(fig)

    # Fig 2: spike tails, share of entries above 1x and 3x the field RMS, early target.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for f in fams:
        ax.plot(x, [max(.01, early[f]['budgets'][v]['spikeOver1']) for v in variants], marker=MARKERS[f], color=COLORS[f], label=f)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.minorticks_off(); ax.set_ylim(.007, 150)
    ax.set_xticks(x); ax.set_xticklabels([f'1/{f}' for f in fractions])
    ax.text(x[-1], .0085, 'zero shown at 0.01%', ha='right', color=MUTED, fontsize=8)
    ax.set_xlabel('Input budget'); ax.set_ylabel('Entries with |error| > field RMS (%)')
    ax.set_title('Outliers in early-field inputs', color=INK, loc='left'); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(HERE / 'fig2-spikes.png', dpi=160); plt.close(fig)

    # Fig 3: baselines on the test split vs measured cost.
    test = report['early']['test']['all']['budgets']
    label_s = report['early']['test']['all']['labelSeconds'] / 1
    fig, ax = plt.subplots(figsize=(6.6, 4))
    cost = [test[v]['seconds'] * 1000 for v in variants]
    ax.plot(cost, [test[v]['error'] for v in variants], marker='o', color='#2a78d6', label='raw, one seed (more rays)')
    ax.plot([4 * c for c in cost], [test[v]['mean4'] for v in variants], marker='s', color='#eb6834', label='mean of 4 seeds')
    ax.plot(cost, [test[v]['neighbour'] for v in variants], marker='^', color='#1baf7a', label='mean with grid neighbours')
    ax.plot(cost, [test[v]['clamp'] for v in variants], marker='D', color='#eda100', label='per-family 3x-median clamp')
    for c, v, f in zip(cost, variants, fractions):
        ax.annotate(f'1/{f}', (c, test[v]['error']), textcoords='offset points', xytext=(4, 4), color=MUTED, fontsize=8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Measured solve + readback time per receiver (ms)'); ax.set_ylabel('Error vs label, test split (%)')
    ax.set_title('Non-learned baselines a network must beat', color=INK, loc='left'); ax.legend()
    fig.tight_layout(); fig.savefig(HERE / 'fig3-baselines.png', dpi=160); plt.close(fig)

    # Fig 4: tracer vs FDTD bias per family.
    bias = report.get('waveBias', {}).get('families', {})
    if bias:
        fam = [f for f in FAMILY_ORDER if f in bias]
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        width = .38
        for j, (hz, color) in enumerate([(125.0, '#2a78d6'), (250.0, '#eb6834')]):
            vals = [np.nan if bias[f][str(hz)]['early'] is None else bias[f][str(hz)]['early'] for f in fam]
            pos = np.arange(len(fam)) + (j - .5) * width
            ax.bar(pos, vals, width - .04, color=color, label=f'{hz:g} Hz')
            for p, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.text(p, v + .5, f'{v:.0f}%', ha='center', va='bottom', color=MUTED, fontsize=8)
        ax.set_xticks(np.arange(len(fam)))
        ax.set_xticklabels([f'{f}\n({bias[f]["125.0"]["settled"]} / {bias[f]["250.0"]["settled"]} settled)' for f in fam])
        ax.set_ylabel('Rigid early-field label vs FDTD (%)'); ax.grid(axis='x', visible=False)
        top = max(bias[f][str(hz)]['early'] or 0 for f in fam for hz in (125.0, 250.0))
        ax.set_ylim(0, top * 1.25)
        ax.set_title('Rigid tracer vs FDTD, settled scenes only', color=INK, loc='left'); ax.legend(loc='upper left')
        fig.tight_layout(); fig.savefig(HERE / 'fig4-wave-bias.png', dpi=160); plt.close(fig)


def scene_figure(base, bundles_data, path):
    """Top view of one generated scene per family with receivers coloured by 1/64 error."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    style(plt)
    picks = {}
    for bundle, data, scene in bundles_data:
        picks.setdefault(bundle['family'], (bundle, data, scene))
    fams = [f for f in FAMILY_ORDER if f in picks]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, f in zip(axes.ravel(), fams):
        bundle, data, scene = picks[f]
        for wall, mat in zip(scene['walls'], scene['surfaceMaterials']):
            c, h = np.array(wall['center']), np.array(wall['half'])
            if c[1] - h[1] > 2.2 or c[1] + h[1] < .5:
                continue     # skip roofs/floors in the plan view
            ax.add_patch(Rectangle((c[0] - h[0], c[2] - h[2]), 2 * h[0], 2 * h[2], facecolor='#c3c2b7', edgecolor=MUTED, lw=.6))
        pts = np.array(scene['points']).reshape(-1, 3)
        target = data['target']
        x = data['inputs']['input-1-64'][..., :7].sum(-1)
        err = np.sqrt(np.mean(np.abs(x - target) ** 2, axis=(0, 2)) / np.mean(np.abs(target) ** 2))
        sc = ax.scatter(pts[:, 0], pts[:, 2], c=100 * err, cmap='Blues', vmin=0, vmax=150, s=22, edgecolors=MUTED, linewidths=.3)
        ax.plot(scene['source'][0], scene['source'][2], marker='*', color='#eb6834', markersize=13, markeredgecolor=INK, mew=.5)
        ax.set_aspect('equal'); ax.set_title(f'{bundle["bundle_id"]} ({bundle["split"]})', color=INK, loc='left', fontsize=9)
        ax.grid(False)
    fig.colorbar(sc, ax=axes, shrink=.6, label='Receiver error at 1/64 budget, early field (%)')
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dataset', type=Path)
    p.add_argument('--no-figures', action='store_true')
    a = p.parse_args()
    manifest, loaded, base = load_all(a.dataset)
    variants = manifest['low_variants']
    fractions = [int(v.split('-')[-1]) for v in variants]
    report = dict(dataset=str(a.dataset), datasetSha256=dd.sha256(a.dataset / 'dataset.json'),
                  variants=variants, frequencies=manifest['grid']['frequencies_hz'])
    for name, mask in [('early', list(range(7))), ('total', list(range(8)))]:
        rows = [scene_metrics(b, d, mask, variants, ds.neighbours(s)) for b, d, s in loaded]
        report[name] = dict(families=summarise(rows, variants, FAMILY_ORDER),
                            splits={sp: summarise([r for r in rows if r['split'] == sp], variants, [])['all']
                                    for sp in ['train', 'validation', 'test'] if any(r['split'] == sp for r in rows)},
                            test=summarise([r for r in rows if r['split'] == 'test'], variants, FAMILY_ORDER),
                            scenes=rows)
    biases = [x for x in (wave_bias(b, d, s) for b, d, s in loaded) if x]
    if biases:
        fam = {}
        for f in FAMILY_ORDER:
            sel = [x for x in biases if x['family'] == f]
            if not sel:
                continue
            fam[f] = dict(scenes=len(sel))
            for hz in sorted({r['hz'] for x in sel for r in x['rows']}):
                every = [r for x in sel for r in x['rows'] if r['hz'] == hz]
                # A response still changing between 0.75 T and T has no settled single-frequency value.
                rows = [r for r in every if r['truncation'] <= TRUNCATION_LIMIT]
                pick = lambda k: 100 * rms([r[k] for r in rows if r[k] is not None]) if any(r[k] is not None for r in rows) else None
                fam[f][str(hz)] = {k: pick(k) for k in ['early', 'total', 'earlyLos', 'earlyShadow', 'labelSE', 'materialVsRigid']}
                fam[f][str(hz)].update(settled=len(rows), unsettled=len(every) - len(rows))
        report['waveBias'] = dict(families=fam, scenes=biases)
    report['throughput'], report['batches'] = throughput(base, [b for b, _, _ in loaded])
    report['verify'] = dd.verify(a.dataset, deep=False)
    (HERE / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    (HERE / 'tables.md').write_text(tables(report))
    if not a.no_figures:
        figures(report, variants, fractions)
        scene_figure(base, loaded, HERE / 'fig5-scenes.png')
    print(json.dumps({k: report[k] for k in ['throughput']}, indent=1))



def tables(report):
    """Markdown tables for the README, generated so no number is transcribed by hand."""
    early, total = report['early']['families'], report['total']['families']
    variants = report['variants']
    short = {v: '1/' + v.split('-')[-1] for v in variants}
    fams = [f for f in FAMILY_ORDER if f in early] + ['all']
    out = ['### Label noise (early-field label, K = 8)', '',
           '| Family | Scenes | Accepted | Label SE (RMS) | Label SE (worst) | Valid entries | Input SD / label SE at ' +
           ' / '.join(short.values()) + ' | Total-label SE |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for f in fams:
        e = early[f]
        ratio = ' / '.join(f'{e["budgets"][v]["noiseRatio"]:.0f}' for v in variants) if e['budgets'] else 'n/a'
        out.append(f'| {f} | {e["scenes"]} | {e["accepted"]} | {e["labelRelSE"]:.1f}% | {e["labelRelSEMax"]:.1f}% | '
                   f'{100 * e["validFraction"]:.0f}% | {ratio} | {total[f]["labelRelSE"]:.0f}% |')
    out += ['', '### Input error against the label (accepted scenes, early field)', '',
            '| Family | ' + ' | '.join(short.values()) + ' | 125–500 Hz at 1/64 | 2–4 kHz at 1/64 |',
            '|---|' + '---:|' * (len(variants) + 2)]
    for f in fams:
        b = early[f]['budgets']
        out.append(f'| {f} | ' + ' | '.join(f'{b[v]["error"]:.0f}%' for v in variants) +
                   f' | {b[variants[1]]["bands"]["low"]:.0f}% | {b[variants[1]]["bands"]["high"]:.0f}% |')
    out += ['', '### Outliers (early field, all accepted scenes)', '',
            '| Budget | Median p99 of \\|error\\| / field RMS | Largest | Entries > 1× RMS | > 3× RMS | Family with the largest error in >1× entries |',
            '|---|---:|---:|---:|---:|---|']
    for v in variants:
        b = early['all']['budgets'][v]
        top = sorted(b['spikeFamilies'].items(), key=lambda kv: -kv[1])
        n = sum(b['spikeFamilies'].values()) or 1
        share = ', '.join(f'{k} {100 * c / n:.0f}%' for k, c in top[:3] if c)
        out.append(f'| {short[v]} | {b["spikeP99"]:.2f} | {b["spikeMax"]:.0f} | {b["spikeOver1"]:.1f}% | {b["spikeOver3"]:.2f}% | {share} |')
    test = report['early']['test']['all']['budgets']
    out += ['', '### Baselines on the test split (early field; canopy family and wide doorways)', '',
            '| Budget | Solve ms | Raw, 1 seed | Raw at the next budget (4× rays) | Mean of 4 seeds (4× solves) | Mean of 2 seeds | Grid-neighbour mean | Family clamp |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for i, v in enumerate(variants):
        b = test[v]
        nxt = f'{test[variants[i + 1]]["error"]:.0f}%' if i + 1 < len(variants) else 'n/a'
        out.append(f'| {short[v]} | {1000 * b["seconds"]:.1f} | {b["error"]:.0f}% | {nxt} | {b["mean4"]:.0f}% | {b["mean2"]:.0f}% | '
                   f'{b["neighbour"]:.0f}% | {b["clamp"]:.0f}% |')
    bias = report.get('waveBias', {}).get('families', {})
    if bias:
        out += ['', '### Rigid tracer label against FDTD', '',
                '| Family | Settled scenes 125 / 250 Hz | 125 Hz early | 125 Hz incl. walk | 125 Hz LoS / shadow | 250 Hz early | 250 Hz incl. walk | 250 Hz LoS / shadow | Label SE 125 / 250 Hz |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        fmt = lambda v: 'n/a' if v is None or v != v else f'{v:.0f}%'
        fmt1 = lambda v: 'n/a' if v is None or v != v else f'{v:.1f}%'
        for f, e in bias.items():
            a, c = e['125.0'], e['250.0']
            out.append(f'| {f} | {a["settled"]} / {c["settled"]} of {e["scenes"]} | {fmt(a["early"])} | {fmt(a["total"])} | {fmt(a["earlyLos"])} / {fmt(a["earlyShadow"])} | '
                       f'{fmt(c["early"])} | {fmt(c["total"])} | {fmt(c["earlyLos"])} / {fmt(c["earlyShadow"])} | '
                       f'{fmt1(a["labelSE"])} / {fmt1(c["labelSE"])} |')
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    main()
