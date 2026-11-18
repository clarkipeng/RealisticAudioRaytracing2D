"""A denoiser for persistent rays (docs/benchmarks/path-denoise-2026-11-12, part 25), trained on
tools/path_persistent_dataset.py.

    PY=.context/fdtd-teacher/venv/bin/python
    python3 tools/heavy_guard.py run --limit-gib 3 -- $PY tools/path_persistent_denoise.py train
    python3 tools/heavy_guard.py run --limit-gib 3 -- $PY tools/path_persistent_denoise.py evaluate
    $PY tools/path_persistent_denoise.py export          # weights for the compute kernel (PathDenoise.compute)

Input at update u (persistent rays, the scene's 23 ms population N): the histograms of the last K = 4
updates, energy [direction 6][band 8][bin] and contribution counts [bin], plus two exponential moving
averages of the histogram (decays .7 and .9, bias-corrected from the first update). A persistent update's
bin at delay t holds the cohort emitted t earlier, so the same delay in consecutive updates comes from
different cohorts (independent for a static listener) and the same rays reappear one update later at
t + 23 ms; the convolution sees both (its receptive field is 66 bins). Output: the update's histogram.
Model, normaliser, conservation and loss are those of tools/path_denoise.py: a 1-D dilated convolution
along the bins predicting non-negative weights on the K-update mean, the two averages and a floor, then
every band's energy per time segment rescaled to the K-update mean's (band levels stay unbiased: the
last four updates are, for a static listener and, to 35 ms of lag, a moving one).

Targets (Noise2Noise, L2 in perceptual windows): the same update's histogram at 8 N (at most 2^19) with
two independent seeds, averaged; along the same track, so a moving listener's target is the same
retarded-time quantity.
Baseline: the exponential moving average of the persistent input with its decay chosen on validation
(the best temporal filter at equal compute: it costs nothing, and the denoiser's GPU time would buy 1-2%
more rays, about 0.05 dB). Updates 32-63 of each track are scored, so every average has 32 updates of
history (a decay of .95 keeps 81% of its asymptotic weight).
Splits: the frozen rule of tools/dlss_scenes.py (corridor and crowded furnished rooms validation; canopy,
wide doorways and the Church test).
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import path_denoise as pdn  # noqa: E402

DATA = ROOT / '.context/path-dataset-persistent'
WORK = ROOT / '.context/path-denoise/persistent'
K, CH = 4, 48
DECAYS = (.7, .9)                 # the network's averages
CIN = K * (CH + 1) + len(DECAYS) * CH
FIRST_TRAIN, FIRST_SCORED = 16, 32
EMA_GRID = (0, .3, .5, .6, .7, .8, .85, .9, .93, .95, .97)
WIDTH = 96
MIX = 8 * 3                       # per band: logits of the three totals (K-update mean, the two averages)
DILATIONS = (1, 1, 2, 4, 8, 16, 1)   # the first is the input layer


def ema(h, decay):
    """Bias-corrected exponential moving average along axis 0 (updates)."""
    out, acc, weight = np.empty_like(h, dtype=np.float64), 0., 0.
    for u in range(len(h)):
        acc = decay * acc + (1 - decay) * h[u]; weight = decay * weight + (1 - decay)
        out[u] = acc / weight
    return out


def load(splits):
    tracks = []
    for e in json.loads((DATA / 'index.json').read_text()):
        if e['split'] not in splits:
            continue
        z = np.load(e['file'])
        low, counts, high = z['low'], z['lowCounts'], z['high'].astype(np.float64)
        # float32 throughout (the training set is about a gigabyte in float64): the target is the two high seeds'
        # mean, and one high seed is kept for the target's own noise level.
        tracks.append(dict(e, low=low, counts=counts, target=high.mean(0).astype(np.float32), high=high[:1].astype(np.float32), edges=z['edges'],
                           emas=[[ema(low[s].astype(np.float64), d).astype(np.float32) for d in DECAYS] for s in range(len(low))]))
    return tracks


def set_edges(tracks):
    """The windows of tools/path_denoise.py follow the longest bin layout (the Church's 6 s); shorter ones are its prefix."""
    longest = max((t['edges'] for t in tracks), key=len)
    for t in tracks:
        assert np.allclose(longest[:len(t['edges'])], t['edges']), t['file']
    if pdn.EDGES is None or len(pdn.EDGES) < len(longest):
        pdn.EDGES = longest


def features(track, seed, u):
    """Network input [CIN, B], the K-update mean m and the normaliser n (scaled), and the scale; None if empty."""
    h = track['low'][seed][u - K + 1:u + 1].astype(np.float64)   # [K, 48, B]
    m = h.mean(0)
    e = [track['emas'][seed][j][u].astype(np.float64) for j in range(len(DECAYS))]
    scale = m.sum() if m.sum() > 0 else e[-1].sum()
    if scale <= 0:
        return None
    x = np.log10(np.concatenate([h, np.stack(e)]) / scale + 1e-9)          # [K + 2, 48, B]
    c = np.log1p(track['counts'][seed][u - K + 1:u + 1].astype(np.float64)) / 10   # [K, B]
    B = h.shape[-1]
    x = np.concatenate([np.concatenate([x[:K], c[:, None]], 1).reshape(-1, B), x[K:].reshape(-1, B)])   # [K * 49 + 2 * 48, B]
    ms = np.stack([m] + e) / scale                          # [3, 48, B]: the K-update mean and the averages
    n = pdn.normaliser(.5 * (ms[0] + ms[-1]))
    return x.astype(np.float32), ms.astype(np.float32), n, scale


def model():
    import torch.nn as nn
    layers = [nn.Conv1d(CIN, WIDTH, 3, padding=1), nn.GELU()]
    for d in DILATIONS[1:]:
        layers += [nn.Conv1d(WIDTH, WIDTH, 3, padding=d, dilation=d), nn.GELU()]
    layers += [nn.Conv1d(WIDTH, 4 * CH + MIX, 1)]
    return nn.Sequential(*layers)


def predict(net, x, ms, n, S):
    """Non-negative weights (starting near one) on the K-update mean and the two averages plus a floor weight
    (starting near zero) on the normaliser; then every band's energy per time segment is rescaled to a convex
    combination of the three inputs' totals there, its weights a softmax of the network's logits averaged over
    the segment. Each input's totals are unbiased for a static listener, so the combination is too: band levels
    stay unbiased, and the network only chooses how much history to trust (a moving listener's, less)."""
    import torch
    raw = net(x)
    N, B = raw.shape[0], raw.shape[-1]
    w = torch.exp(4 * torch.tanh(raw[:, :3 * CH] / 4)).reshape(N, 3, CH, B)
    b = torch.nn.functional.softplus(raw[:, 3 * CH:4 * CH] - 4)
    den = ((w * ms).sum(1) + b * n).reshape(N, 6, 8, B)
    count = S.sum(0).clamp(min=1)
    mix = torch.softmax((raw[:, 4 * CH:].reshape(N, 8, 3, B) @ S) / count, 2)          # [N, band, input, segment]
    totals = (ms.reshape(N, 3, 6, 8, B).sum(2) @ S).permute(0, 2, 1, 3)               # [N, band, input, segment]
    target = (mix * totals).sum(2)                                                    # [N, band, segment]
    have = den.sum(1) @ S
    ratio = torch.where(have > 0, target / torch.where(have > 0, have, torch.ones_like(have)), torch.zeros_like(have))
    return (den * (ratio @ S.T)[:, None]).reshape(N, CH, B)


def sample(track, seed, u, k):
    f = features(track, seed, u)
    if f is None:
        return None
    x, ms, n, scale = f
    B = x.shape[-1]   # the symmetry permutes the direction blocks of every 48-channel group
    frames = x[:K * (CH + 1)].reshape(K, CH + 1, B)
    frames = np.concatenate([pdn.symmetry(frames[:, :CH], k), frames[:, CH:]], 1).reshape(-1, B)
    averages = pdn.symmetry(x[K * (CH + 1):].reshape(len(DECAYS), CH, B), k).reshape(-1, B)
    x = np.concatenate([frames, averages])
    t = pdn.symmetry(track['target'][u], k) / scale
    return x, pdn.symmetry(ms, k), pdn.symmetry(n, k), t.astype(np.float32)


def batch(tracks, rng, size):
    """Random updates of random tracks with the same bin count (one shape per batch), under a random symmetry."""
    out = []
    B = None
    while len(out) < size:
        t = tracks[rng.integers(len(tracks))]
        if B is not None and t['low'].shape[-1] != B:
            continue
        s = sample(t, int(rng.integers(len(t['low']))), int(rng.integers(FIRST_TRAIN, t['low'].shape[1])), int(rng.integers(8)))
        if s is None:
            continue
        B = t['low'].shape[-1]; out.append(s)
    return [np.stack(v) for v in zip(*out)]


def scored(tracks, first=FIRST_SCORED):
    """Every scored (track, seed, update) with its inputs: (track, seed, u, x, ms, n, scale)."""
    for t in tracks:
        for s in range(len(t['low'])):
            for u in range(first, t['low'].shape[1]):
                f = features(t, s, u)
                if f is not None and t['target'][u].sum() > 0:
                    yield (t, s, u) + f


def run(net, S, x, ms, n):
    import torch
    with torch.no_grad():
        return predict(net, *(torch.from_numpy(np.asarray(v, np.float32))[None] for v in (x, ms, n)), S)[0].numpy().astype(np.float64)


def segment(B):
    import torch
    return torch.from_numpy(pdn.segment_matrix(B))


def train(steps, tag, seed=0):
    import torch
    torch.manual_seed(seed)
    tracks, valid = load({'train'}), load({'validation'})
    set_edges(tracks + valid)
    dev = pdn.device(); net = model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    vset = list(scored(valid))[::3]
    torch.set_num_threads(4)

    def validation():
        net.eval()
        e = []
        for t, s, u, x, ms, n, scale in vset:
            with torch.no_grad():
                d = predict(net, *(torch.from_numpy(np.asarray(v, np.float32))[None].to(dev) for v in (x, ms, n)), segment(x.shape[-1]).to(dev))[0]
            m = pdn.metrics(d.cpu().numpy().astype(np.float64), t['target'][u] / scale)
            e.append(m['window'] + m['level'] + m['edc'])   # all three: window error alone let band levels drift
        net.train()
        return float(np.mean(e))
    log, best, start = [], (math.inf, -1), time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    matrices = {}
    for step in range(steps + 1):
        if step % 500 == 0:
            v = validation(); log.append((step, v))
            print(step, f'validation error (window + level + decay) {v:.3f} dB', f'{time.time() - start:.0f} s', flush=True)
            if v < best[0]:
                best = (v, step); torch.save({k: w.cpu() for k, w in net.state_dict().items()}, WORK / f'denoiser-{tag}.pt')
        if step == steps:
            break
        x, ms, n, t = (torch.from_numpy(v).to(dev) for v in batch(tracks, rng, 16))
        B = x.shape[-1]
        if B not in matrices:
            matrices[B] = (torch.from_numpy(pdn.window_matrix(B)).to(dev), segment(B).to(dev))
        W, S = matrices[B]
        loss = (((predict(net, x, ms, n, S) - t) @ W / (n @ W)) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
    (WORK / f'train-{tag}.json').write_text(json.dumps(dict(steps=steps, seconds=time.time() - start, validation=log, bestStep=best[1],
                                                               bestValidation=best[0], parameters=sum(p.numel() for p in net.parameters()),
                                                               trainTracks=len(tracks)), indent=1))


def choose_decay():
    """The EMA decay with the lowest mean window error on the validation tracks."""
    valid = load({'validation'}); set_edges(valid)
    errors = {}
    for d in EMA_GRID:
        e = []
        for t in valid:
            for s in range(len(t['low'])):
                avg = ema(t['low'][s], d)
                e += [pdn.metrics(avg[u], t['target'][u])['window'] for u in range(FIRST_SCORED, len(avg)) if t['target'][u].sum() > 0]
        errors[d] = float(np.mean(e))
    best = min(errors, key=errors.get)
    print('EMA decay on validation:', {k: round(v, 3) for k, v in errors.items()}, '->', best, flush=True)
    return best, errors


NAMES = pdn.NAMES
ESTIMATES = ('noisy', 'kmean', 'ema', 'denoised', 'target')


def evaluate(tag):
    import torch
    decay, grid = choose_decay()
    tracks = load({'validation', 'test'}); set_edges(tracks)
    net = model(); net.load_state_dict(torch.load(WORK / f'denoiser-{tag}.pt', map_location='cpu')); net.eval()
    torch.set_num_threads(1)
    rows, t_infer = [], []
    for t in tracks:
        avg = [ema(t['low'][s], decay) for s in range(len(t['low']))]
        S = segment(t['low'].shape[-1])
        for tr, s, u, x, ms, n, scale in scored([t]):
            t0 = time.perf_counter(); den = run(net, S, x, ms, n) * scale; t_infer.append(time.perf_counter() - t0)
            target = t['target'][u]
            r = dict(scene=t['scene'], family=t['family'], split=t['split'], track=t['track'], moving=t['moving'], seed=s, update=u)
            for name, est in (('noisy', t['low'][s][u]), ('kmean', ms[0] * scale), ('ema', avg[s][u]), ('denoised', den), ('target', t['high'][0][u])):
                r[name] = pdn.metrics(est, target)
            rows.append(r)
    summary = dict(model=tag, decay=decay, decayGrid=grid, rows=len(rows), cpuInferenceSeconds=float(np.median(t_infer)))
    groups = {}
    for r in rows:
        family = r['family'] + (' (moving)' if r['moving'] else ' (static)')
        for g in (r['split'], r['family'], family, 'all held out'):
            groups.setdefault(g, []).append(r)
    summary['groups'] = {g: dict(n=len(v), **{e: {k: float(np.nanmean([r[e][k] for r in v])) for k in NAMES} for e in ESTIMATES})
                         for g, v in groups.items()}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f'eval-{tag}.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print(json.dumps(summary, indent=1))
    return summary


def export(tag):
    """Weights for the compute kernel (Assets/Resources/Acoustics/PathDenoise.compute), float32 little endian:
    per convolution, weight [out][in][tap] then bias [out]; and a JSON header with the shapes."""
    import torch
    net = model(); net.load_state_dict(torch.load(WORK / f'denoiser-{tag}.pt', map_location='cpu'))
    convs = [m for m in net if isinstance(m, torch.nn.Conv1d)]
    target = ROOT / 'Assets/Resources/Acoustics/PathDenoise'
    target.mkdir(parents=True, exist_ok=True)
    blob = np.concatenate([np.concatenate([c.weight.detach().numpy().ravel(), c.bias.detach().numpy().ravel()]) for c in convs]).astype('<f4')
    (target / 'weights.bytes').write_bytes(blob.tobytes())
    header = dict(K=K, channels=CH, decays=list(DECAYS), layers=[dict(input=c.in_channels, output=c.out_channels, taps=c.kernel_size[0],
                                                                       dilation=c.dilation[0]) for c in convs], floats=int(blob.size))
    (target / 'header.json').write_text(json.dumps(header, indent=1) + '\n')
    print(f'{blob.size} weights in {len(convs)} layers -> {target}')
    fixture(net)


FIXTURE = ROOT / 'Assets/Tests/Editor/PathDenoiseFixture.bytes'


def fixture(net, updates=8, bins=64):
    """A reference for the kernel (PathDenoiseChecks): the first `bins` bins of the first `updates` updates of
    a validation walk, and this model's output for updates K - 1 on. Layout, float32: updates, bins, then
    bin starts [bins], histograms [updates][48][bins], counts [updates][bins], outputs [updates - K + 1][48][bins]."""
    t = next(t for t in load({'validation'}) if t['moving'])
    t = dict(t, low=t['low'][:1, :updates, :, :bins], counts=t['counts'][:1, :updates, :bins])
    t['emas'] = [[ema(t['low'][0], d) for d in DECAYS]]
    set_edges([t])
    S = segment(bins)
    outputs = []
    for u in range(K - 1, updates):
        x, ms, n, scale = features(t, 0, u)
        outputs.append(run(net, S, x, ms, n) * scale)
    blob = np.concatenate([[updates, bins], t['edges'][:bins], t['low'][0].ravel(), t['counts'][0].ravel(), np.ravel(outputs)]).astype('<f4')
    FIXTURE.write_bytes(blob.tobytes())
    print(f'fixture: {t["scene"]} {t["track"]}, {updates} updates of {bins} bins -> {FIXTURE}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train'); t.add_argument('--steps', type=int, default=8000); t.add_argument('--model', default='persistent')
    e = sub.add_parser('evaluate'); e.add_argument('--model', default='persistent')
    x = sub.add_parser('export'); x.add_argument('--model', default='persistent')
    a = p.parse_args(argv)
    dict(train=lambda: train(a.steps, a.model), evaluate=lambda: evaluate(a.model), export=lambda: export(a.model))[a.command]()


if __name__ == '__main__':
    main()
