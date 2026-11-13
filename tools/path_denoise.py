"""A small denoiser for the Path method's histograms, trained and evaluated on the path-v2 dataset.

    PY=.context/fdtd-teacher/venv/bin/python
    python3 tools/heavy_guard.py run --limit-gib 3 -- $PY tools/path_denoise.py train                # all budgets
    python3 tools/heavy_guard.py run --limit-gib 3 -- $PY tools/path_denoise.py evaluate --rays 512
    python3 tools/heavy_guard.py run --limit-gib 3 -- $PY tools/path_denoise.py church SOURCE TARGET  # exported render

Input: the histograms of K = 4 frames of one listener pose (a moving listener's last 4 frames, or 4
independent solves of a static pose; each frame has its own seed), energy [direction 6][band 8][bin]
and contribution counts [bin], at one ray budget. Output: the denoised histogram of the newest frame.
Model: a 1D convolutional network along the time bins (dilated, about 0.24 M parameters) that
predicts each cell as a gain on the K-frame mean plus a non-negative multiple of the normaliser
(the smoothed mean plus floors; this fills cells that no frame reached), then rescales every band's
energy in each time segment (0-50 ms, 50-200 ms, later) to the K-frame mean's: those totals stay
unbiased and the network only redistributes energy within them (predict(), conserve()).

Targets and loss (Noise2Noise): the target is the mean of two independent 8,192-ray solves, unbiased
but noisy per fine bin (about 25% energy-weighted). With an L2 loss the minimiser is the conditional
mean of the target, which equals the clean mean because the target's noise is independent of the
input and has zero mean; so a noisy target trains toward the clean answer, it only slows training.
The loss is L2 on energies summed into perceptual windows (1 ms early, about t/8 late; per direction
and band), each divided by the window sum of a normaliser computed from the inputs only (see
normaliser()); per fine bin the loss was dominated by single specular arrivals that no input frame
had detected (up to 10^4 times the local level) and did not train.

Evaluation (held-out families: corridor = validation, canopy and the Church = test; two doorway and
two furnished scenes fall in test and validation by the frozen split rule), against the target mean:
  (a) the newest frame alone (the raw input);
  (b) more rays at equal compute: plain averages of independent solves (1..4 seeds x 256..2048 rays),
      interpolated in log-log at rays x (1 + denoiser time / trace time);
      (bK) the same for the average of K frames at that budget (the strict bar: a static listener has
      its last K frames for free, so plain temporal averaging uses them too);
  (b') the plain average of the same K frames (same information, no learning);
  (c) one 8,192-ray seed against the two-seed mean (the target's own noise level).
Metrics (see metrics()): band energies in perceptual windows (RMS dB), band level (RMS dB), decay
curve (RMS dB). All are against the noisy target, so they flatten toward (c).
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.context/path-dataset-v2/datasets/path-v2'
WORK = ROOT / '.context/path-denoise'
K = 4
CH = 48


def load(split_filter=None):
    """Per scene: inputs[rays][seed] [R, 48, B], counts[rays][seed] [R, B], target mean/var [R, 48, B], kinds."""
    scenes = []
    examples = [json.loads(l) for l in open(DATA / 'examples.jsonl')]
    for line in open(DATA / 'bundles.jsonl'):
        b = json.loads(line)
        if split_filter and b['split'] not in split_filter:
            continue
        root = DATA.parents[1]
        def arr(d, dtype):
            return np.fromfile(root / d['file'], dtype).reshape(d['shape'])
        mean = arr(b['files']['targetMean'], '<f4'); R, B = mean.shape[0], mean.shape[-1]
        var = arr(b['files']['targetVariance'], '<f4')
        inputs, counts, seconds = {}, {}, b['cost']['lowSeconds']
        for i in b['files']['inputs']:
            inputs.setdefault(i['rays'], {})[i['seed']] = arr(i['histogram'], '<f4').reshape(R, CH, B)
            counts.setdefault(i['rays'], {})[i['seed']] = arr(i['counts'], '<u4').astype(np.float32)
        ex = sorted([e for e in examples if e['scene'] == b['scene']], key=lambda e: e['pose'])
        scenes.append(dict(scene=b['scene'], family=b['family'], split=b['split'], mean=mean.reshape(R, CH, B), var=var.reshape(R, CH, B),
                           inputs=inputs, counts=counts, examples=ex, seconds=seconds, poses=R))
    return scenes


def frames(scene, rays, pose):
    """The K input frames of a pose (newest last): a moving listener's last K frames (seed 0), or K seeds of a static pose."""
    e = scene['examples'][pose]
    if e['kind'] == 'moving' and e['frame'] is not None:
        first = pose - e['frame']
        idx = [max(first, pose - K + 1 + j) for j in range(K)]
        return [scene['inputs'][rays][0][i] for i in idx], [scene['counts'][rays][0][i] for i in idx]
    seeds = sorted(scene['inputs'][rays])[:K]
    return [scene['inputs'][rays][s][pose] for s in seeds], [scene['counts'][rays][s][pose] for s in seeds]


def features(hist, cnt):
    """Network input [K*(48+1), B] and the K-frame mean [48, B] (scaled), and the scale."""
    h = np.stack(hist).astype(np.float64)        # [K, 48, B]
    mean = h.mean(0)
    scale = max(mean.sum(), 1e-30)
    x = np.log10(h / scale + 1e-9)               # [K, 48, B]
    c = np.log1p(np.stack(cnt))[:, None, :] / 10  # [K, 1, B]
    return np.concatenate([x, c], 1).reshape(-1, h.shape[-1]).astype(np.float32), (mean / scale).astype(np.float32), scale


def smooth(m, width=5):
    kernel = np.ones(width) / width
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode='same'), -1, m)


def normaliser(m):
    """Per-cell scale from the inputs only: the K-frame mean smoothed over 5 bins, plus the mean over all 48
    direction-band cells smoothed over 31 bins (the local level), plus 1% of the mean cell.
    Any positive weight that depends on the inputs only keeps the conditional mean as the L2 minimiser; the
    floors keep cells that the inputs missed (specular arrivals that no frame detected, up to 10^4 times the
    first term) from dominating the gradient."""
    s = smooth(m)
    return (s + smooth(s.mean(0, keepdims=True), 31) + .01 * m.mean()).astype(np.float32)


def model():
    import torch.nn as nn
    layers, width = [], 96
    layers += [nn.Conv1d(K * (CH + 1), width, 3, padding=1), nn.GELU()]
    for d in (1, 2, 4, 8, 16, 1):
        layers += [nn.Conv1d(width, width, 3, padding=d, dilation=d), nn.GELU()]
    layers += [nn.Conv1d(width, 2 * CH, 1)]
    return nn.Sequential(*layers)


def symmetry(h, k):
    """One of the 8 rotations and reflections about the vertical axis (k in 0..7), on [..., 48, B] with
    directions +x, -x, +y, -y, +z, -z: swap x and z, then flip x, then flip z."""
    order = [0, 1, 2, 3, 4, 5]
    if k & 1: order = [order[i] for i in (4, 5, 2, 3, 0, 1)]
    if k & 2: order = [order[i] for i in (1, 0, 2, 3, 4, 5)]
    if k & 4: order = [order[i] for i in (0, 1, 2, 3, 5, 4)]
    shape = h.shape
    return h.reshape(shape[:-2] + (6, 8, shape[-1]))[..., order, :, :].reshape(shape)


def batch(scenes, budgets, rng, size):
    """Random poses at random budgets, each under a random symmetry about the vertical axis."""
    xs, ms, ts, ns = [], [], [], []
    while len(xs) < size:
        s = scenes[rng.integers(len(scenes))]; p = int(rng.integers(s['poses'])); k = int(rng.integers(8))
        hist, cnt = frames(s, int(budgets[rng.integers(len(budgets))]), p)
        if not any(h.sum() > 0 for h in hist):
            continue   # no frame found anything: nothing to normalise (evaluation counts these poses)
        x, m, scale = features([symmetry(h, k) for h in hist], cnt)
        t = symmetry(s['mean'][p], k) / scale
        xs.append(x); ms.append(m); ts.append(t.astype(np.float32)); ns.append(normaliser(m))
    return [np.stack(v) for v in (xs, ms, ts, ns)]


def device():
    import torch
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def predict(net, x, m, n, S):
    """The denoised histogram (torch, scaled units): a * (K-frame mean) + b * normaliser per cell, with a a
    soft-bounded gain (starts at one: the identity is the starting point) and b >= 0 (starts near zero: fills
    cells no frame reached), then rescaled per band and time segment to the K-frame mean's totals (conserve)."""
    import torch
    raw = net(x)
    a = torch.exp(4 * torch.tanh(raw[:, :CH] / 4))
    b = torch.nn.functional.softplus(raw[:, CH:] - 4)
    return conserve(a * m + b * n, m, S)


SEGMENTS = (0, .05, .2)   # seconds: early, middle, late


def segment_matrix(bins):
    """[bins, 3] 0/1 matrix of the time segments (0-50 ms, 50-200 ms, 200 ms on)."""
    edges = np.fromfile(DATA / 'bin-edges.f32', '<f4') if EDGES is None else EDGES
    starts = np.concatenate([edges[:-1], edges[-1:] + np.arange(max(0, bins - len(edges) + 1)) * 0])[:bins]
    if bins > len(edges) - 1:   # longer renders (the Church walk, 6 s): extend the last bin layout (t/64 widths)
        t = list(edges[:-1])
        while len(t) < bins:
            t.append(t[-1] * (1 + 1 / 64))
        starts = np.array(t[:bins])
    seg = np.searchsorted(np.array(SEGMENTS[1:]), starts, side='right')
    m = np.zeros((bins, len(SEGMENTS)), np.float32); m[np.arange(bins), seg] = 1
    return m


def conserve(den, m, S):
    """Rescale the denoised [.., 48, B] so every band's energy in each time segment (directions summed) equals
    the K-frame mean's: those totals stay unbiased, the network only redistributes within them."""
    xp = np if isinstance(den, np.ndarray) else __import__('torch')
    sh = den.shape
    d = den.reshape(sh[:-2] + (6, 8, sh[-1])); k = m.reshape(sh[:-2] + (6, 8, sh[-1]))
    ds, ks = (d.sum(-3) @ S), (k.sum(-3) @ S)                       # [.., 8, 3]
    ratio = xp.where(ds > 0, ks / xp.where(ds > 0, ds, 1), 0 * ds)
    return (d * (ratio @ S.T)[..., None, :, :]).reshape(sh)


def fixed_set(scenes, rays):
    """Every non-empty pose of the given scenes as one evaluation batch (inputs, normalisers, targets)."""
    xs, ms, ns, ts = [], [], [], []
    for s in scenes:
        for p in range(s['poses']):
            hist, cnt = frames(s, rays, p)
            if not any(h.sum() > 0 for h in hist):
                continue
            x, m, scale = features(hist, cnt)
            xs.append(x); ms.append(m); ns.append(normaliser(m)); ts.append((s['mean'][p] / scale).astype(np.float32))
    return [np.stack(v) for v in (xs, ms, ns, ts)]


def window_matrix(bins):
    """[bins, windows] 0/1 matrix summing bins into the perceptual windows."""
    w = [(a, b) for a, b in windows() if b <= bins]
    m = np.zeros((bins, len(w)), np.float32)
    for j, (a, b) in enumerate(w):
        m[a:b, j] = 1
    return m


def train(budgets, steps, tag, seed=0):
    """Noise2Noise training on the train families. The loss is L2 on the energies of the perceptual windows
    (per direction and band; both prediction and target are summed into windows, which keeps the target
    unbiased) divided by the normaliser's window sums; the checkpoint with the best mean window error
    (metrics()) on the validation poses is kept."""
    import torch
    torch.manual_seed(seed)
    scenes = load({'train'})
    dev = device(); net = model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    valid = load({'validation'})
    sets = [fixed_set(valid, b) for b in budgets]
    W = torch.from_numpy(window_matrix(sets[0][0].shape[-1])).to(dev)
    S = torch.from_numpy(segment_matrix(sets[0][0].shape[-1])).to(dev)

    def validation():
        net.eval(); errors = []
        with torch.no_grad():
            for vx, vm, vn, vt in sets:
                for i in range(0, len(vx), 64):
                    d = predict(net, *(torch.from_numpy(v[i:i + 64]).to(dev) for v in (vx, vm, vn)), S).cpu().numpy()
                    errors += [metrics(d[j], vt[i + j])['window'] for j in range(len(d))]
        net.train()
        return float(np.mean(errors))
    log, best = [], (math.inf, -1)
    start = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    for step in range(steps + 1):
        if step % 250 == 0:
            v = validation()
            log.append((step, v)); print(step, f'validation window error {v:.3f} dB', f'{time.time() - start:.0f} s', flush=True)
            if v < best[0]:
                best = (v, step)
                torch.save(net.state_dict(), WORK / f'denoiser-{tag}.pt')
        if step == steps:
            break
        x, m, t, n = (torch.from_numpy(v).to(dev) for v in batch(scenes, budgets, rng, 32))
        pred = predict(net, x, m, n, S)
        loss = (((pred - t) @ W / (n @ W)) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
    (WORK / f'train-{tag}.json').write_text(json.dumps(dict(budgets=list(budgets), steps=steps, seconds=time.time() - start, validation=log,
                                                               bestStep=best[1], bestValidation=best[0],
                                                               parameters=sum(p.numel() for p in net.parameters()),
                                                               trainScenes=len(scenes)), indent=1))


EDGES = None


def windows():
    """Perceptual windows over the bins: 1 ms (four fine bins) up to 16 ms, then eight bins (about t/8)."""
    global EDGES
    if EDGES is None:
        EDGES = np.fromfile(DATA / 'bin-edges.f32', '<f4')
    out, first = [], 0
    while first < len(EDGES) - 1:
        width = 4 if EDGES[first] < .016 else 8
        out.append((first, min(first + width, len(EDGES) - 1))); first = out[-1][1]
    return out


def window_energy(h):
    """[48, B] -> [band 8, window] energies (directions summed)."""
    e = h.reshape(6, 8, -1).sum(0)
    return np.stack([e[:, a:b].sum(1) for a, b in windows() if b <= e.shape[1]], 1)


def metrics(est, target):
    """Errors of one estimate against the target, [48, B] each:
    window: RMS dB of the band energies in perceptual windows where the target's is within 40 dB of its
    band peak (errors clipped to +-20 dB, so a missed window counts 20 dB);
    level: RMS over bands of the dB error of the band's total energy;
    edc: RMS dB of the Schroeder curves where the target's is above -40 dB."""
    we, wt = window_energy(est), window_energy(target)
    ok = wt > wt.max(1, keepdims=True) * 1e-4
    d = np.clip(10 * np.log10(np.maximum(we, 1e-300) / np.maximum(wt, 1e-300)), -20, 20)
    window = float(np.sqrt((d[ok] ** 2).mean())) if ok.any() else float('nan')
    eb, tb = est.reshape(6, 8, -1).sum(0), target.reshape(6, 8, -1).sum(0)
    good = tb.sum(1) > 0
    level = np.clip(10 * np.log10(np.maximum(eb.sum(1), 1e-300) / np.maximum(tb.sum(1), 1e-300)), -40, 40)[good]
    edc_e = np.cumsum(eb[:, ::-1], 1)[:, ::-1]; edc_t = np.cumsum(tb[:, ::-1], 1)[:, ::-1]
    ce = 10 * np.log10(np.maximum(edc_e, 1e-300) / np.maximum(edc_e[:, :1], 1e-300))
    ct = 10 * np.log10(np.maximum(edc_t, 1e-300) / np.maximum(edc_t[:, :1], 1e-300))
    sel = (ct > -40) & good[:, None]
    edc = float(np.sqrt((np.clip(ce - ct, -40, 40)[sel] ** 2).mean())) if sel.any() else float('nan')
    return dict(window=window, level=float(np.sqrt((level ** 2).mean())) if level.size else float('nan'), edc=edc)


NAMES = ('window', 'level', 'edc')
ESTIMATES = ('noisy', 'kmean', 'denoised', 'target')


def evaluate(rays, tag, splits=('validation', 'test')):
    """Errors against the target mean for (a) one frame, (b') the K-frame mean and the denoiser; (b) / (bK)
    more rays at equal compute (log-log interpolation of plain averages of 1..4 seeds x 256..2048 rays);
    (c) one 8,192-ray seed against the mean of two, from the per-cell variance (the target's own error
    is about this over the square root of two)."""
    import torch
    scenes = load(set(splits))
    net = model(); net.load_state_dict(torch.load(WORK / f'denoiser-{tag}.pt', map_location='cpu')); net.eval()
    torch.set_num_threads(1)
    rows, empty, t_infer, curve = [], 0, [], {}
    S = torch.from_numpy(segment_matrix(scenes[0]['mean'].shape[-1]))
    for s in scenes:
        trace_solve = s['seconds'][str(rays)] / (len(s['inputs'][rays]) * s['poses'])   # one solve at this budget
        for p in range(s['poses']):
            target = s['mean'][p].astype(np.float64)
            if target.sum() <= 0:
                continue
            hist, cnt = frames(s, rays, p)
            if not any(h.sum() > 0 for h in hist):
                empty += 1   # every estimate is zero here (inputs, their mean and the denoiser alike)
                continue
            x, m, scale = features(hist, cnt)
            t0 = time.perf_counter()
            with torch.no_grad():
                d = predict(net, *(torch.from_numpy(v)[None] for v in (x, m, normaliser(m))), S)[0].numpy()
            t_infer.append(time.perf_counter() - t0)
            den = d.astype(np.float64) * scale
            r = dict(scene=s['scene'], family=s['family'], split=s['split'], pose=p, kind=s['examples'][p]['kind'], traceSolveSeconds=float(trace_solve))
            for name, est in (('noisy', hist[-1].astype(np.float64)), ('kmean', np.mean(hist, 0).astype(np.float64)), ('denoised', den)):
                r[name] = metrics(est, target)
            # A single seed differs from the two-seed mean by +-sqrt(var / 2) per cell (var: their sample variance).
            sd = np.sqrt(np.maximum(s['var'][p], 0) / 2)
            signs = np.where(np.random.default_rng(p).random(target.shape) < .5, -1., 1.)
            r['target'] = metrics(np.maximum(target + sd * signs, 0), target)
            for b in sorted(s['inputs']):
                seeds = sorted(s['inputs'][b])
                for n in range(1, len(seeds) + 1):
                    est = np.mean([s['inputs'][b][q][p] for q in seeds[:n]], 0).astype(np.float64)
                    curve.setdefault(b * n, []).append(metrics(est, target))
            rows.append(r)
    infer = float(np.median(t_infer))
    trace = float(np.median([r['traceSolveSeconds'] for r in rows]))
    equal = rays * (1 + infer / trace)      # the inference time spent on rays instead (cost linear in rays)
    xs = np.array(sorted(curve))
    summary = dict(model=tag, rays=rays, inferenceSeconds=infer, traceSolveSeconds=trace, equalComputeRays=equal, poses=len(rows), emptyInputPoses=empty,
                   curve={str(int(x)): {k: float(np.nanmean([c[k] for c in curve[x]])) for k in NAMES} for x in xs},
                   moreRays={}, moreRaysK={})
    for k in NAMES:
        ys = np.array([summary['curve'][str(int(x))][k] for x in xs])
        summary['moreRays'][k] = float(np.exp(np.interp(np.log(equal), np.log(xs), np.log(ys))))       # (b)
        summary['moreRaysK'][k] = float(np.exp(np.interp(np.log(K * equal), np.log(xs), np.log(ys))))  # (bK)
    for name in ESTIMATES:
        summary[name] = {k: float(np.nanmean([r[name][k] for r in rows])) for k in NAMES}
    fam = {}
    for r in rows:
        fam.setdefault(r['family'], []).append(r)
    summary['families'] = {f: dict(n=len(v), **{name: {k: float(np.nanmean([r[name][k] for r in v])) for k in NAMES} for name in ESTIMATES})
                           for f, v in fam.items()}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f'eval-{tag}-r{rays}.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'curve'}, indent=1))
    return summary


def church(tag, source, target):
    """Denoise an exported render (PathReverb export: histogram.f32 [response][6][8][B], counts.u32
    [response][B]; one source/listener pair): each response with its K - 1 predecessors as context,
    written as <target>/histogram.f32 for the render's import (pathHistogramImport)."""
    import torch
    source, target = Path(source), Path(target)
    counts = np.fromfile(source / 'counts.u32', '<u4')
    hist = np.fromfile(source / 'histogram.f32', '<f4')
    n = len(np.fromfile(source / 'samples.i64', '<i8'))
    B = counts.size // n
    hist = hist.reshape(n, CH, B); counts = counts.reshape(n, B).astype(np.float32)
    net = model(); net.load_state_dict(torch.load(WORK / f'denoiser-{tag}.pt', map_location='cpu')); net.eval()
    out = np.zeros_like(hist)
    S = torch.from_numpy(segment_matrix(B))
    with torch.no_grad():
        for i in range(n):
            idx = [max(0, i - K + 1 + j) for j in range(K)]
            x, m, scale = features([hist[j] for j in idx], [counts[j] for j in idx])
            if hist[i].sum() <= 0:
                continue
            out[i] = predict(net, *(torch.from_numpy(v)[None] for v in (x, m, normaliser(m))), S)[0].numpy() * scale
    target.mkdir(parents=True, exist_ok=True)
    out.astype('<f4').tofile(target / 'histogram.f32')
    print(f'denoised {n} responses of {B} bins into {target}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train'); t.add_argument('--budgets', default='256,512,1024,2048'); t.add_argument('--steps', type=int, default=8000)
    t.add_argument('--model', default='all')
    e = sub.add_parser('evaluate'); e.add_argument('--rays', type=int, default=512); e.add_argument('--model', default='all')
    c = sub.add_parser('church'); c.add_argument('--model', default='all'); c.add_argument('source'); c.add_argument('target')
    args = parser.parse_args(argv)
    if args.command == 'train':
        train([int(b) for b in args.budgets.split(',')], args.steps, args.model)
    elif args.command == 'church':
        church(args.model, args.source, args.target)
    else:
        evaluate(args.rays, args.model)


if __name__ == '__main__':
    main()
