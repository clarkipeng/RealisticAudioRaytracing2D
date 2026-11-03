"""Per-path-family gains: fit on train, select on validation, report on test.

    PY=.context/fdtd-teacher/venv/bin/python
    B=docs/benchmarks/family-gains-2026-11-03
    nice -n 19 $PY $B/analyze.py select    # train + validation only; writes selection.json
    nice -n 19 $PY $B/analyze.py test      # held-out test, rigid-vs-FDTD and walk checks; writes report.json, tables.md
    nice -n 19 $PY $B/analyze.py export    # the selected gains as the renderer's data asset
    $PY $B/analyze.py figures

`select` never loads a test scene. `test` refuses to run without selection.json and
evaluates every candidate, so the test numbers of the rejected candidates are
reported but cannot change the choice.

Error metric (as in the pilot and the denoiser report): complex relative L2 over
receivers and the 32 frequencies, against the 8-seed converged early field (every
family except the residual walk), RMS over scenes. Receivers are split by the
scene generator's analytic line of sight ("view") versus none ("shadow"); each
subset is normalised by its own label energy, so the shadow number says how wrong
the shadow field is, not how small it is. A scene contributes to a subset only if
it has receivers there.

Level metrics: "energy" is the predicted over the label energy of the subset (summed
over receivers, frequencies and seeds), averaged over scenes and given in dB; silence
is -inf, the raw tracer's spikes make it positive, and a level-preserving estimate is
near 0 dB even when its phases are wrong. "Band bias" is descriptive only: per seed,
receiver and 1 kHz band (8 frequencies), 10 log10 of the predicted over the label
energy, clipped to +-30 dB, then averaged; it is biased low for any noisy estimate.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_dataset as dd  # noqa: E402

DATASET = dd.DATA / 'datasets/denoise-pilot-v1'
WALK_LABELS = ROOT / '.context/residual-walk/label-after/scenes'
VARIANTS = ['input-1-256', 'input-1-64', 'input-1-16', 'input-1-4']
SHORT = ['1/256', '1/64', '1/16', '1/4']
SAMPLES = [256, 1024, 4096, 16384]          # completeRouteSamples per budget
RAYS = [2, 8, 32, 128]                      # raysPerBin per budget
LABEL_SAMPLES = 65536
MIRROR_CANDIDATES = 128                     # TakeRender.Settings.reflectionCandidates default
FAMILIES = dd.FAMILIES                      # direct, R, D, DD, RD, DR, RR, walk
EARLY = 7
B, F = len(VARIANTS), len(FAMILIES)
BANDS = 4                                   # 1 kHz bands of 8 frequencies
CLIP_DB = 30.0
TRUNCATION_LIMIT = .05                      # pilot: FDTD responses changing more than 5% are unsettled

# ---------------------------------------------------------------------------------------------
# Selection rule, fixed before any validation or test evaluation (see README "Selection rule").
RULE = dict(
    eligibility='validation energy ratio (view and shadow separately, scene mean, in dB) within +-3 dB at every budget',
    levelLimitDb=3.0,
    score='mean over the four budgets of (ln error_view + ln error_shadow) / 2 on validation; lowest eligible wins',
    fallback='if no candidate is eligible, the lowest score wins and the violation is reported',
    defaultOn='on test at 1/256 (live budget): error_view and error_shadow both below raw, energy ratio within +-3 dB '
              'in view and shadow, and the rigid-vs-FDTD shadow error not above raw; otherwise the setting ships opt-in',
)
CANDIDATES = ['raw', 'ls', 'ls-shadow', 'energy', 'rho-wiener', 'rho-energy']
DESCRIPTION = {
    'raw': 'every family at gain 1 (current renderer)',
    'ls': 'least-squares real gain per family and budget (denoiser report baseline, direct fixed at 1)',
    'ls-shadow': 'least squares with shadow receivers given half of every scene\'s weight',
    'energy': 'per family and budget, the gain that preserves the family\'s expected energy',
    'rho-wiener': 'g = 1 / (1 + k_f rho), rho = routes per sample of that family at run time; one k per family',
    'rho-energy': 'g = (1 + k_f rho)^(-1/2): the level-preserving form of rho-wiener with the same k',
}


# --------------------------------------------------------------------------------------- data

def edge_count(boundary):
    """Diffracting edges as RayTransport.Edges builds them from the baked boundary.

    Edges are keyed per boundary group (one scene object per material group); an edge
    with one face, or two faces that are not coplanar, is kept; three or more faces drop
    it. Every dataset material (stone, wood, glass, rigid) is opaque.
    """
    n = 0
    for b in boundary['boundaries']:
        v = np.array(b['vertices'], np.float64).reshape(-1, 3)
        faces = {}
        for tri in np.array(b['triangles']).reshape(-1, 3):
            a, p, c = v[tri]
            normal = np.cross(p - a, c - a)
            normal /= np.linalg.norm(normal)
            for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (min(i, j), max(i, j))
                if key in faces:
                    first, keep, count = faces[key]
                    faces[key] = (first, count == 1 and abs(first @ normal) < .99999, count + 1)
                else:
                    faces[key] = (normal, True, 1)
        n += sum(1 for f in faces.values() if f[1])
    return n


def routes_per_sample(edges, primitives):
    """rho [B, F]: candidate routes (or mirror primitives) per sample, as the shader sees them.

    Complete routes: with every candidate enumerated (total <= samples) each route gets
    samples/total samples, so rho = total/samples for every family; otherwise each
    nonempty family gets samples/families draws over its own routes, rho_f = size_f *
    families / samples (the shader's route weight). Mirrors: primitives / candidates.
    Direct and walk are not sampled per route (rho = 0, unused).
    """
    sizes = np.array([edges, edges * edges, primitives * edges, primitives * edges, primitives * primitives], float)
    total, families = sizes.sum(), int((sizes > 0).sum())
    rho = np.zeros((B, F))
    for b, n in enumerate(SAMPLES):
        rho[b, 2:7] = total / n if total <= n else sizes * families / n
        rho[b, 1] = primitives / min(primitives, max(MIRROR_CANDIDATES, RAYS[b]))
    return rho


def load(splits):
    manifest, bundles, base = dd.open_dataset(DATASET)
    scenes = []
    for b in bundles:
        if not b['quality']['accepted'] or b['split'] not in splits:
            continue
        d = dd.load_bundle(base, b)
        desc = json.loads((base / b['scene']['file']).read_text())
        run = json.loads((base / f"scenes/{b['bundle_id']}/tracer/run.json").read_text())
        edges = edge_count(json.loads((base / b['boundary']['file']).read_text()))
        s = dict(id=b['bundle_id'], family=b['family'], split=b['split'],
                 x=np.stack([d['inputs'][v] for v in VARIANTS]).astype(np.complex128),    # [B, S, P, Q, F]
                 t=d['target'].astype(np.complex128),                                     # [P, Q]
                 los=np.array(desc['lineOfSight'], bool), edges=edges, primitives=run['primitives'])
        s['rho'] = routes_per_sample(edges, run['primitives'])
        if b.get('wave') and b.get('rigid'):
            trunc = np.array(b['wave']['truncationChange075'])
            hz = b['wave']['frequencies']
            index = [desc['frequencies'].index(h) for h in hz]
            s['rigid'] = dict(x=np.stack([d['rigid_inputs'][v] for v in VARIANTS])[..., index, :].astype(np.complex128),
                              label=d['rigid_label_runs'][..., :EARLY].sum(-1).mean(0)[:, index].astype(np.complex128),
                              wave=d['wave'].astype(np.complex128), settled=trunc <= TRUNCATION_LIMIT, hz=hz)
        s['G'] = grams(s)
        scenes.append(s)
    return manifest, scenes


def subsets(s):
    return [(n, m) for n, m in (('all', np.ones_like(s['los'])), ('view', s['los']), ('shadow', ~s['los'])) if m.any()]


def grams(s):
    """Quadratic-form statistics so any real per-family gain vector scores in O(F^2)."""
    out = {}
    S = s['x'].shape[1]
    for name, m in subsets(s):
        xm, tm = s['x'][:, :, m], s['t'][m]
        out[name] = dict(A=np.einsum('bspqf,bspqg->bfg', xm.conj(), xm).real,
                         r=np.einsum('bspqf,pq->bf', xm.conj(), tm).real,
                         c=S * np.sum(np.abs(tm) ** 2), receivers=int(m.sum()))
    return out


# ------------------------------------------------------------------------------------ candidates

def table_gains(spec):
    return np.array(spec['gains'])


def gains_for(spec, s):
    """[B, F] gains of one candidate for one scene (rho-based ones depend on the scene)."""
    if spec['kind'] == 'table':
        return np.array(spec['gains'])
    k = np.array(spec['k'])
    g = np.ones((B, F))
    g[:, 1:7] = (1 + k[None] * s['rho'][:, 1:7]) ** -spec['power']
    return g


def fit_ls(train, shadow_weight=None):
    """Least squares on the train split, scene-balanced; direct and walk fixed at 1."""
    g = np.ones((B, F))
    for b in range(B):
        A, r = np.zeros((F, F)), np.zeros(F)
        for s in train:
            parts = [('all', 1.0)] if shadow_weight is None else [('view', 1 - shadow_weight), ('shadow', shadow_weight)]
            if shadow_weight is not None and 'shadow' not in s['G']:
                parts = [('view', 1.0)]
            for name, w in parts:
                G = s['G'][name]
                A += w * G['A'][b] / G['c']
                r += w * G['r'][b] / G['c']
        free = slice(1, EARLY)
        g[b, free] = np.linalg.solve(A[free, free], r[free] - A[free, 0])     # direct gain 1
    return g


def fit_energy(train, labels):
    """Gain preserving each family's expected energy: sqrt(label energy / input energy)."""
    g = np.ones((B, F))
    for b in range(B):
        num, den = np.zeros(F), np.zeros(F)
        for s in train:
            c = s['G']['all']['c']
            num += labels[s['id']] * s['x'].shape[1] / c
            den += np.diagonal(s['G']['all']['A'][b]) / c
        g[b, 1:EARLY] = np.sqrt(num[1:EARLY] / den[1:EARLY])
    return g


def fit_rho(train):
    from scipy.optimize import minimize

    def objective(logk):
        spec = dict(kind='rho', k=np.exp(logk).tolist(), power=1.0)
        return sum(scene_error(s['G']['all'], gains_for(spec, s))[0].sum() for s in train)

    best = minimize(objective, np.zeros(6), method='Nelder-Mead',
                    options=dict(maxiter=6000, xatol=1e-5, fatol=1e-8))
    return np.exp(best.x)


def per_family_label_energy(splits):
    """Sum over receivers and frequencies of |mean over label seeds|^2 per family."""
    manifest, bundles, base = dd.open_dataset(DATASET)
    out = {}
    for b in bundles:
        if b['quality']['accepted'] and b['split'] in splits:
            d = dd.load_bundle(base, b)
            out[b['bundle_id']] = np.sum(np.abs(d['label_family_runs'].astype(np.complex128).mean(0)) ** 2, axis=(0, 1))
    return out


def fit_candidates(train):
    ls = fit_ls(train)
    k = fit_rho(train)
    raw = np.ones((B, F))
    return {
        'raw': dict(kind='table', gains=raw.tolist()),
        'ls': dict(kind='table', gains=ls.tolist()),
        'ls-shadow': dict(kind='table', gains=fit_ls(train, shadow_weight=.5).tolist()),
        'energy': dict(kind='table', gains=fit_energy(train, per_family_label_energy({'train'})).tolist()),
        'rho-wiener': dict(kind='rho', k=k.tolist(), power=1.0),
        'rho-energy': dict(kind='rho', k=k.tolist(), power=.5),
    }


# ------------------------------------------------------------------------------------- metrics

def scene_error(G, g):
    """Squared relative error and energy ratio per budget for gains g [B, F] (walk excluded)."""
    g = g.copy()
    g[:, EARLY:] = 0
    q = np.einsum('bf,bfg,bg->b', g, G['A'], g)
    return (q - 2 * np.einsum('bf,bf->b', g, G['r']) + G['c']) / G['c'], q / G['c']


def band_levels(s, g, m):
    """Mean and mean-absolute band level error (dB) for receivers m, per budget."""
    y = np.einsum('bspqf,bf->bspq', s['x'][:, :, m, :, :EARLY], g[:, :EARLY])
    t = s['t'][m]
    ye = (np.abs(y) ** 2).reshape(*y.shape[:3], BANDS, -1).sum(-1)
    te = (np.abs(t) ** 2).reshape(t.shape[0], BANDS, -1).sum(-1)
    with np.errstate(divide='ignore'):
        db = np.clip(10 * np.log10(ye / te[None, None]), -CLIP_DB, CLIP_DB)
    return db.mean((1, 2, 3)), np.abs(db).mean((1, 2, 3))


def evaluate(spec, scenes):
    """Per scene: error^2, energy ratio and band level per budget and subset."""
    rows = []
    for s in scenes:
        g = gains_for(spec, s)
        row = dict(scene=s['id'], family=s['family'], split=s['split'], subsets={})
        for name, m in subsets(s):
            e2, level = scene_error(s['G'][name], g)
            bias, absolute = band_levels(s, g, m)
            row['subsets'][name] = dict(receivers=int(m.sum()), error2=e2.tolist(), energy=level.tolist(),
                                        biasDb=bias.tolist(), absDb=absolute.tolist())
        rows.append(row)
    return rows


def summarise(rows):
    """Scene-RMS error (%), mean energy ratio (dB) and band level bias/abs (dB) per subset."""
    out = {}
    for name in ['all', 'view', 'shadow']:
        sel = [r['subsets'][name] for r in rows if name in r['subsets']]
        if not sel:
            continue
        e2 = np.array([x['error2'] for x in sel])
        out[name] = dict(scenes=len(sel), receivers=int(sum(x['receivers'] for x in sel)),
                         error=(100 * np.sqrt(e2.mean(0))).tolist(),
                         energyDb=(10 * np.log10(np.mean([x['energy'] for x in sel], 0))).tolist(),
                         biasDb=np.mean([x['biasDb'] for x in sel], 0).tolist(),
                         absDb=np.mean([x['absDb'] for x in sel], 0).tolist())
    return out


def by_family(rows):
    return {f: summarise([r for r in rows if r['family'] == f]) for f in sorted({r['family'] for r in rows})}


def score(summary):
    return float(np.mean([.5 * (math.log(summary['view']['error'][b]) + math.log(summary['shadow']['error'][b]))
                          for b in range(B)]))


def eligible(summary):
    return all(abs(summary[n]['energyDb'][b]) <= RULE['levelLimitDb'] for n in ['view', 'shadow'] for b in range(B))


def rigid_fdtd(spec, scenes):
    """Rigid-wall low-budget inputs, raw or gained, against FDTD at 125/250 Hz (settled only)."""
    rows = []
    for s in scenes:
        r = s.get('rigid')
        if r is None:
            continue
        g = gains_for(spec, s)
        y = np.einsum('bspqf,bf->bspq', r['x'][..., :EARLY], g[:, :EARLY])          # [B, S, P, 2]
        for j, hz in enumerate(r['hz']):
            if not r['settled'][j]:
                continue
            row = dict(scene=s['id'], family=s['family'], split=s['split'], hz=hz)
            for name, m in subsets(s):
                w = r['wave'][m, j]
                norm = np.sum(np.abs(w) ** 2)
                err = np.sqrt(np.mean(np.sum(np.abs(y[:, :, m, j] - w) ** 2, axis=2), axis=1) / norm)
                row[name] = dict(error=err.tolist(),
                                 label=float(np.sqrt(np.sum(np.abs(r['label'][m, j] - w) ** 2) / norm)))
            rows.append(row)
    return rows


def summarise_fdtd(rows, family):
    out = {}
    for hz in sorted({r['hz'] for r in rows}):
        sel = [r for r in rows if r['family'] == family and r['hz'] == hz]
        if not sel:
            continue
        out[str(hz)] = {'scenes': len(sel)}
        for name in ['all', 'view', 'shadow']:
            v = [r[name] for r in sel if name in r]
            if v:
                out[str(hz)][name] = dict(error=(100 * np.sqrt(np.mean(np.square([x['error'] for x in v]), 0))).tolist(),
                                          label=100 * math.sqrt(np.mean(np.square([x['label'] for x in v]))))
    return out


# ------------------------------------------------------------------------------------------ walk

def walk_check(spec_early):
    """v13 walk at the input budgets against walk-including labels (residual-walk re-renders).

    For each re-rendered pilot scene: the converged walk's size relative to the early
    field, the low-budget walk buffer's RMS relative to the early field, and the error
    against the total label (early + walk) of the selected early gains with walk gain 0
    or 1. Open scenes (barriers, canopy) are the only ones whose walk label converges.
    """
    out = []
    for d in sorted(WALK_LABELS.iterdir()):
        run = json.loads((d / 'tracer/run.json').read_text())
        p = run['points']

        def read(name, seeds):
            a = np.fromfile(d / f'tracer/responses-{name}.f32', '<f4').reshape(seeds, p, 32, F, 2)
            return (a[..., 0] + 1j * a[..., 1]).astype(np.complex128)
        label = read('label', 8)
        early, walk = label[..., :EARLY].sum(-1).mean(0), label[..., EARLY].mean(0)
        total = early + walk
        ce, ct = np.sum(np.abs(early) ** 2), np.sum(np.abs(total) ** 2)
        row = dict(scene=d.name, family=d.name.split('-')[0], walkLabelOverEarly=100 * math.sqrt(np.sum(np.abs(walk) ** 2) / ce),
                   budgets=[])
        for b, v in enumerate(VARIANTS):
            x = read(v, 4)
            g = np.array(spec_early['gains'])[b] if spec_early['kind'] == 'table' else None
            if g is None:
                edges = edge_count(json.loads((d / 'tracer/boundary.json').read_text()))
                g = gains_for(spec_early, dict(rho=routes_per_sample(edges, run['primitives'])))[b]
            y = np.einsum('spqf,f->spq', x[..., :EARLY], g[:EARLY])
            err = [100 * math.sqrt(np.mean(np.sum(np.abs(y + a * x[..., EARLY] - total) ** 2, (1, 2))) / ct) for a in (0, 1)]
            row['budgets'].append(dict(budget=v, walkOverEarly=100 * math.sqrt(np.mean(np.sum(np.abs(x[..., EARLY]) ** 2, (1, 2))) / ce),
                                       errorWalk0=err[0], errorWalk1=err[1]))
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------- stages

def provenance(manifest):
    return dict(dataset='denoise-pilot-v1', datasetSha256=dd.sha256(DATASET / 'dataset.json'),
                script=str(Path(__file__).relative_to(ROOT)),
                scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))


def stage_select():
    manifest, scenes = load({'train', 'validation'})
    train = [s for s in scenes if s['split'] == 'train']
    val = [s for s in scenes if s['split'] == 'validation']
    specs = fit_candidates(train)
    results = {}
    for name in CANDIDATES:
        v = summarise(evaluate(specs[name], val))
        results[name] = dict(validation=v, train=summarise(evaluate(specs[name], train)),
                             score=score(v), eligible=eligible(v))
    ok = [n for n in CANDIDATES if results[n]['eligible']]
    pool = ok or CANDIDATES
    chosen = min(pool, key=lambda n: results[n]['score'])
    out = dict(provenance(manifest), rule=RULE, candidates=specs, description=DESCRIPTION, results=results,
               selected=chosen, anyEligible=bool(ok),
               scenes=dict(train=[s['id'] for s in train], validation=[s['id'] for s in val]),
               geometry={s['id']: dict(edges=s['edges'], primitives=s['primitives']) for s in scenes})
    (HERE / 'selection.json').write_text(json.dumps(out, indent=1) + '\n')
    for n in CANDIDATES:
        r = results[n]
        print(f"{n:11s} score {r['score']:.3f} eligible {r['eligible']}  view "
              + ' '.join(f"{e:6.1f}" for e in r['validation']['view']['error']) + '  shadow '
              + ' '.join(f"{e:6.1f}" for e in r['validation']['shadow']['error']) + '  energy dB view/shadow '
              + ' '.join(f"{a:+.1f}/{b:+.1f}" for a, b in zip(r['validation']['view']['energyDb'], r['validation']['shadow']['energyDb'])))
    print('selected', chosen)


def stage_test():
    sel = json.loads((HERE / 'selection.json').read_text())
    specs = sel['candidates']
    manifest, scenes = load({'train', 'validation', 'test'})
    test = [s for s in scenes if s['split'] == 'test']
    report = dict(provenance(manifest), rule=sel['rule'], selected=sel['selected'], candidates=specs,
                  description=sel['description'], selection=sel['results'], test={}, fdtd={}, walk=None)
    for name in CANDIDATES:
        rows = evaluate(specs[name], test)
        report['test'][name] = dict(all=summarise(rows), families=by_family(rows), scenes=rows)
        fd = rigid_fdtd(specs[name], scenes)
        report['fdtd'][name] = dict(families={f: summarise_fdtd(fd, f) for f in sorted({r['family'] for r in fd})}, scenes=fd)
    report['walk'] = walk_check(specs[sel['selected']] if sel['selected'] != 'raw' else specs['ls'])
    t, raw = report['test'][sel['selected']]['all'], report['test']['raw']['all']
    fs, fr = (report['fdtd'][n]['families'].get('canopy', {}) for n in (sel['selected'], 'raw'))
    fdtd_ok = all(fs[h]['shadow']['error'][0] <= fr[h]['shadow']['error'][0] for h in fs if 'shadow' in fs[h])
    report['defaultOn'] = dict(
        viewBelowRaw=t['view']['error'][0] < raw['view']['error'][0],
        shadowBelowRaw=t['shadow']['error'][0] < raw['shadow']['error'][0],
        levelWithinLimit=all(abs(t[n]['energyDb'][0]) <= RULE['levelLimitDb'] for n in ['view', 'shadow']),
        fdtdShadowNotWorse=bool(fdtd_ok))
    report['defaultOn']['decision'] = all(report['defaultOn'].values())
    (HERE / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    (HERE / 'tables.md').write_text(tables(report))
    print(tables(report))


def fmt(v, d=1):
    return '–' if v is None else f'{v:.{d}f}'


def tables(r):
    out = ['# Generated tables (analyze.py test)', '']
    out += ['## Test split: early-field error (%) and level bias (dB), in view / shadow', '',
            '| Candidate | Budget | View error | Shadow error | View energy dB | Shadow energy dB | View band bias dB | Shadow band bias dB |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for n in CANDIDATES:
        a = r['test'][n]['all']
        for b in range(B):
            out.append(f"| {n} | {SHORT[b]} | {fmt(a['view']['error'][b])} | {fmt(a['shadow']['error'][b])} | "
                       f"{a['view']['energyDb'][b]:+.1f} | {a['shadow']['energyDb'][b]:+.1f} | {a['view']['biasDb'][b]:+.1f} | {a['shadow']['biasDb'][b]:+.1f} |")
    out += ['', '## Validation (selection)', '', '| Candidate | Score | Eligible | View 1/256 / 1/64 / 1/16 / 1/4 | Shadow | View energy dB | Shadow energy dB |',
            '|---|---:|---|---|---|---|---|']
    for n in CANDIDATES:
        v = r['selection'][n]['validation']
        out.append(f"| {n} | {r['selection'][n]['score']:.3f} | {r['selection'][n]['eligible']} | "
                   + ' / '.join(fmt(e) for e in v['view']['error']) + ' | ' + ' / '.join(fmt(e) for e in v['shadow']['error'])
                   + ' | ' + ' / '.join(f'{e:+.1f}' for e in v['view']['energyDb']) + ' | ' + ' / '.join(f'{e:+.1f}' for e in v['shadow']['energyDb']) + ' |')
    out += ['', '## Test families', '', '| Candidate | Family | View 1/256 / 1/64 / 1/16 / 1/4 | Shadow | Shadow energy dB |', '|---|---|---|---|---|']
    for n in CANDIDATES:
        for f, v in r['test'][n]['families'].items():
            sh = v.get('shadow')
            out.append(f"| {n} | {f} | " + ' / '.join(fmt(e) for e in v['view']['error']) + ' | '
                       + (' / '.join(fmt(e) for e in sh['error']) if sh else '–') + ' | '
                       + (' / '.join(f'{e:+.1f}' for e in sh['energyDb']) if sh else '–') + ' |')
    out += ['', '## Rigid tracer against FDTD (settled scenes), error % at 1/256 / 1/64 / 1/16 / 1/4; label = 8-seed rigid label', '',
            '| Candidate | Family | Hz | Scenes | View | Shadow | Label view / shadow |', '|---|---|---:|---:|---|---|---|']
    for n in CANDIDATES:
        for f, hzs in r['fdtd'][n]['families'].items():
            for hz, v in hzs.items():
                out.append(f"| {n} | {f} | {hz} | {v['scenes']} | " + ' / '.join(fmt(e) for e in v['view']['error']) + ' | '
                           + (' / '.join(fmt(e) for e in v['shadow']['error']) if 'shadow' in v else '–') + ' | '
                           + f"{fmt(v['view']['label'])} / {fmt(v['shadow']['label']) if 'shadow' in v else '–'} |")
    out += ['', '## Walk (v13) against walk-including labels', '',
            '| Scene | Converged walk / early | Walk buffer / early, 1/256 / 1/64 / 1/16 / 1/4 | Error, walk gain 0 | Error, walk gain 1 |', '|---|---:|---|---|---|']
    for w in r['walk']:
        out.append(f"| {w['scene']} | {fmt(w['walkLabelOverEarly'])}% | " + ' / '.join(fmt(x['walkOverEarly']) for x in w['budgets'])
                   + ' | ' + ' / '.join(fmt(x['errorWalk0']) for x in w['budgets']) + ' | ' + ' / '.join(fmt(x['errorWalk1']) for x in w['budgets']) + ' |')
    out += ['', f"Selected: **{r['selected']}**; default-on checks: {json.dumps(r['defaultOn'])}", '']
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['select', 'test', 'export', 'figures'])
    a = p.parse_args()
    if a.stage == 'select':
        stage_select()
    elif a.stage == 'test':
        stage_test()
    elif a.stage == 'export':
        import export
        export.main()
    else:
        import figures
        figures.main()


if __name__ == '__main__':
    main()
