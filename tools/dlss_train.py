"""Train and evaluate the first audio-DLSS denoiser on the packed pilot dataset.

The model maps one source/listener pair's low-budget per-family complex buffers,
path counts and budget, plus distance and direct-path visibility, to the
converged early field (every family except the residual walk). Everything it
reads exists per pair at run time; labels, other receivers, other seeds and
scene IDs are used only by losses and evaluation.

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY tools/dlss_train.py baselines     # deterministic baselines, all splits
    nice -n 19 $PY tools/dlss_train.py sweep         # candidates, selected on validation
    nice -n 19 $PY tools/dlss_train.py ablate        # selected recipe: ablations, shuffle control, seeds
    nice -n 19 $PY tools/dlss_train.py test          # held-out test, once
    nice -n 19 $PY tools/dlss_train.py timing        # inference cost per pair, CPU and MPS

Runs are CPU by default with two threads, under nice 19, pause on battery
(dlss_jobs power hook) and stop if the process RSS passes 4 GiB. Outputs go to
.context/audio-dlss-model/.
"""
import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import dlss_dataset as dd  # noqa: E402

DATASET = dd.DATA / 'datasets/denoise-pilot-v1'
OUT = ROOT / '.context/audio-dlss-model'
VARIANTS = ['input-1-256', 'input-1-64', 'input-1-16', 'input-1-4']
FRACTIONS = [256, 64, 16, 4]
FAMILIES = dd.FAMILIES
EARLY = len(dd.TARGET_FAMILIES)          # families 0..6 form the label; 7 is the residual walk
GIB = 1024 ** 3
RSS_LIMIT = 4 * GIB
SCALE_FLOOR = 1e-6

# Buffer layouts the network may see. Each is a fixed [8 families -> G buffers] grouping
# plus which buffers count as early field (normalisation scale, initial gain 1).
LAYOUTS = {
    'families': (np.eye(8), [1] * 7 + [0]),
    'no-walk': (np.eye(8)[:, :7], [1] * 7),
    'summed-early': (np.stack([[1.] * 7 + [0], [0.] * 7 + [1]], 1), [1, 0]),
    # What the path slot ranges give without the route-table readback: direct, image
    # reflections, all complete routes summed, residual walk.
    'slot-ranges': (np.stack([[1.] + [0] * 7, [0, 1.] + [0] * 6, [0, 0] + [1.] * 5 + [0], [0.] * 7 + [1]], 1), [1, 1, 1, 0]),
    'summed': (np.ones((8, 1)), [1]),
}

BASE = dict(layout='families', counts=True, arch='conv', width=32, depth=2, kernel=1, head='gain',
            loss='huber', delta=1.0, epochs=40, batch=256, lr=2e-3, decay=1e-4, seed=0, shuffle=False)
# Candidates compared on validation only. Selection: the fewest MACs among candidates whose
# mean log validation error over budgets is within SELECTION_TIE (about 1%) of the best.
SELECTION_TIE = .01
CANDIDATES = {
    'gain-k1-w32': {},
    'gain-k1-w64': dict(width=64),
    'gain-k3-w32': dict(kernel=3),
    'gain-k5-w64-d3': dict(kernel=5, width=64, depth=3),
    'direct-k3-w64': dict(kernel=3, width=64, head='direct'),
    'flat-w128': dict(arch='flat', width=128),
    'gain-k1-w32-mse': dict(loss='mse'),
}
ABLATIONS = {
    'summed-early': dict(layout='summed-early'),
    'slot-ranges': dict(layout='slot-ranges'),
    'summed': dict(layout='summed'),
    'no-walk': dict(layout='no-walk'),
    'no-counts': dict(counts=False),
    'shuffled-labels': dict(shuffle=True),
}


def spec_of(**changes):
    spec = dict(BASE)
    unknown = set(changes) - set(spec)
    if unknown:
        raise ValueError(f'unknown spec keys {sorted(unknown)}')
    spec.update(changes)
    if spec['layout'] not in LAYOUTS:
        raise ValueError(f'unknown layout {spec["layout"]}')
    return spec


# ----------------------------------------------------------------------------- data

def load(dataset=DATASET):
    """Accepted scenes as complex arrays [budget, seed, receiver, frequency, family]."""
    manifest, bundles, base = dd.open_dataset(dataset)
    if manifest['low_variants'] != VARIANTS or manifest['grid']['families'] != FAMILIES:
        raise ValueError('dataset layout differs from the trainer')
    scenes = []
    for bundle in bundles:
        if not bundle['quality']['accepted']:
            continue
        d = dd.load_bundle(base, bundle)
        context = d['context']
        scenes.append(dict(
            id=bundle['bundle_id'], family=bundle['family'], split=bundle['split'],
            x=np.stack([d['inputs'][v] for v in VARIANTS]).astype(np.complex64),
            counts=np.stack([d['counts'][v] for v in VARIANTS]).astype(np.float32),
            seconds=np.stack([d['seconds'][v] for v in VARIANTS]).astype(np.float64),
            distance=context[:, dd.CONTEXT_COLUMNS.index('distance')].astype(np.float32),
            visible=context[:, dd.CONTEXT_COLUMNS.index('directVisible')].astype(np.float32),
            target=d['target'].astype(np.complex64), valid=d['valid']))
    return manifest, scenes


def split(scenes, name):
    return [s for s in scenes if s['split'] == name]


def as_pairs(scenes, shuffle=False, seed=0):
    """Flatten scenes to independent pairs. Labels and weights are training-only.

    weight = 1 / receivers, so every scene counts equally (the evaluation metric is
    an RMS over scenes); rms = the scene's label RMS over receivers and frequencies.
    shuffle permutes labels among receivers of a scene (control experiment).
    """
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ['x', 'counts', 'budget', 'distance', 'visible', 'target', 'valid', 'rms', 'weight']}
    for s in scenes:
        b, k, p, q, f = s['x'].shape
        target, valid = s['target'], s['valid']
        if shuffle:
            order = rng.permutation(p)
            target, valid = target[order], valid[order]
        n = b * k
        out['x'].append(s['x'].reshape(-1, q, f))
        out['counts'].append(s['counts'].reshape(-1, q, f))
        out['budget'].append(np.repeat(np.arange(b), k * p))
        out['distance'].append(np.tile(s['distance'], n))
        out['visible'].append(np.tile(s['visible'], n))
        out['target'].append(np.tile(target, (n, 1)))
        out['valid'].append(np.tile(valid, (n, 1)))
        out['rms'].append(np.full(n * p, np.sqrt(np.mean(np.abs(s['target']) ** 2)), np.float32))
        out['weight'].append(np.full(n * p, 1.0 / p, np.float32))
    cat = {k: np.concatenate(v) for k, v in out.items()}
    return dict(x=to_real(cat['x']), counts=torch.from_numpy(cat['counts']),
                budget=torch.from_numpy(cat['budget']), distance=torch.from_numpy(cat['distance']),
                visible=torch.from_numpy(cat['visible']), target=to_real(cat['target']),
                valid=torch.from_numpy(cat['valid']), rms=torch.from_numpy(cat['rms']),
                weight=torch.from_numpy(cat['weight']))


def to_real(z):
    return torch.from_numpy(np.stack([z.real, z.imag], -1).astype(np.float32))


# ----------------------------------------------------------------------------- features

class Features(nn.Module):
    """Runtime-available inputs of one pair -> normalised features and scale.

    Inputs per pair: x [N,Q,8,2] per-family complex buffers, counts [N,Q,8] path
    arrivals, budget index [N], distance [N] (m), visible [N] (renderer direct path).
    The scale s is the median over frequency of the early-buffer magnitude
    sqrt(sum_g |x_g|^2); every feature uses x / s, so the model is exactly
    scale-equivariant (it cannot read absolute level).
    """

    def __init__(self, layout, counts, q):
        super().__init__()
        group, early = LAYOUTS[layout]
        self.register_buffer('group', torch.tensor(group, dtype=torch.float32))
        self.register_buffer('early', torch.tensor(early, dtype=torch.float32))
        self.register_buffer('freq', torch.linspace(0, 1, q))
        self.counts = counts
        g = self.group.shape[1]
        self.buffers_ = g
        self.per_q = 3 * g + (g if counts else 0) + 1
        self.per_pair = 2 * g + (g if counts else 0) + 3

    def forward(self, x, counts, budget, distance, visible):
        xb = torch.einsum('nqfc,fg->nqgc', x, self.group)                     # [N,Q,G,2]
        mag2 = (xb ** 2).sum(-1)
        scale = torch.sqrt((mag2 * self.early).sum(-1)).median(dim=1).values.clamp_min(SCALE_FLOOR)
        u = xb / scale[:, None, None, None]
        m = torch.sqrt((u ** 2).sum(-1) + 1e-12)                              # |x_g| / s
        squash = torch.log1p(m)
        compressed = u * (squash / m)[..., None]                              # phase kept, magnitude log-compressed
        n, q, g = m.shape
        per_q = [compressed.reshape(n, q, 2 * g), squash]
        rms = torch.sqrt((m ** 2).mean(1))
        pair = [torch.log(rms + 1e-3), torch.log1p(m.amax(1)) - torch.log1p(rms)]
        if self.counts:
            cb = torch.einsum('nqf,fg->nqg', counts, self.group)
            per_q.append(torch.log1p(cb))
            pair.append(torch.log1p(cb).mean(1))
        per_q.append(self.freq.expand(n, q)[..., None])
        level = (math.log2(FRACTIONS[0]) - torch.log2(torch.tensor(FRACTIONS, dtype=x.dtype, device=x.device)[budget])) / 6
        pair.append(torch.stack([level, torch.log(distance.clamp_min(.1)), visible], -1))
        return u, scale, torch.cat(per_q, -1), torch.cat(pair, -1)


class Denoiser(nn.Module):
    """Small network over frequency x buffer.

    conv: per-frequency projection, then residual 1D convolutions over frequency
    (kernel 1 = a per-frequency MLP shared across frequencies), pair features
    broadcast to every frequency. flat: one MLP over the whole pair.
    head 'gain' predicts complex per-buffer gains and a residual,
    y = sum_g w_g u_g + r, initialised to the raw early sum; head 'direct'
    predicts y directly. Output is complex [N,Q,2] in units of the input.
    """

    def __init__(self, spec, q=32):
        super().__init__()
        self.spec = dict(spec)
        self.features = Features(spec['layout'], spec['counts'], q)
        g, w = self.features.buffers_, spec['width']
        out = 2 * g + 2 if spec['head'] == 'gain' else 2
        cin = self.features.per_q + self.features.per_pair
        if spec['arch'] == 'conv':
            self.inp = nn.Linear(cin, w)
            k = spec['kernel']
            self.blocks = nn.ModuleList(nn.Conv1d(w, w, k, padding=k // 2) for _ in range(spec['depth']))
            self.head = nn.Linear(w, out)
        elif spec['arch'] == 'flat':
            self.narrow = 8
            self.inp = nn.Linear(self.features.per_q, self.narrow)
            layers = [nn.Linear(q * self.narrow + self.features.per_pair, w), nn.GELU()]
            for _ in range(spec['depth'] - 1):
                layers += [nn.Linear(w, w), nn.GELU()]
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(w, q * out)
        else:
            raise ValueError(spec['arch'])
        self.out = out
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            bias = torch.zeros(self.head.bias.shape[0] // out, out)
            if spec['head'] == 'gain':
                bias[:, 0:2 * g:2] = self.features.early
            self.head.bias.copy_(bias.flatten())

    def forward(self, x, counts, budget, distance, visible):
        u, scale, per_q, pair = self.features(x, counts, budget, distance, visible)
        n, q = per_q.shape[:2]
        if self.spec['arch'] == 'conv':
            h = torch.nn.functional.gelu(self.inp(torch.cat([per_q, pair[:, None].expand(n, q, -1)], -1)))
            h = h.transpose(1, 2)
            for conv in self.blocks:
                h = h + torch.nn.functional.gelu(conv(h))
            o = self.head(h.transpose(1, 2))
        else:
            z = torch.nn.functional.gelu(self.inp(per_q)).reshape(n, -1)
            o = self.head(self.body(torch.cat([z, pair], -1))).reshape(n, q, self.out)
        if self.spec['head'] == 'gain':
            g = u.shape[2]
            w = o[..., :2 * g].reshape(n, q, g, 2)
            y = torch.stack([(w[..., 0] * u[..., 0] - w[..., 1] * u[..., 1]).sum(-1),
                             (w[..., 0] * u[..., 1] + w[..., 1] * u[..., 0]).sum(-1)], -1) + o[..., 2 * g:]
        else:
            y = o
        return y * scale[:, None, None]


def parameters(model):
    return sum(p.numel() for p in model.parameters())


def macs(model, q=32):
    """Multiply-accumulates per pair in linear/conv layers (feature code excluded)."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            per_frequency = model.spec['arch'] == 'conv' or m is model.inp
            total += (q if per_frequency else 1) * m.in_features * m.out_features
        elif isinstance(m, nn.Conv1d):
            total += q * m.in_channels * m.out_channels * m.kernel_size[0]
    return total


# ----------------------------------------------------------------------------- loss and metrics

def loss_fn(pred, batch, spec):
    """Complex error relative to the scene's field RMS; pseudo-Huber or squared.

    Pseudo-Huber is quadratic below delta (field RMS units) and linear above, so a
    few spike-driven errors cannot dominate the gradient. Invalid label entries
    are masked; scenes are balanced by weight.
    """
    e2 = ((pred - batch['target']) ** 2).sum(-1) / batch['rms'][:, None] ** 2
    if spec['loss'] == 'huber':
        d = spec['delta']
        per = d * d * (torch.sqrt(1 + e2 / (d * d)) - 1)
    else:
        per = .5 * e2
    mask = batch['valid'].float()
    per_pair = (per * mask).sum(1) / mask.sum(1).clamp_min(1)
    return (per_pair * batch['weight']).sum() / batch['weight'].sum()


def scene_metrics(pred, scene):
    """Pilot metrics for predictions [B,S,P,Q] complex against the early-field label.

    error2 = mean over seeds of sum|x - t|^2 / sum|t|^2 (dlss_baselines); spikes are
    |x - t| over the receiver-RMS label magnitude at each frequency; level is the
    predicted energy over the label energy.
    """
    t = scene['target'].astype(np.complex128)
    energy = np.sum(np.abs(t) ** 2)
    rq = np.sqrt(np.mean(np.abs(t) ** 2, axis=0, keepdims=True))
    rows = []
    for b in range(pred.shape[0]):
        x = pred[b].astype(np.complex128)
        diff = np.abs(x - t)
        spikes = diff / np.maximum(rq, 1e-30)
        rows.append(dict(error2=float((diff ** 2).sum((1, 2)).mean() / energy),
                         over1=float(np.mean(spikes > 1)), over3=float(np.mean(spikes > 3)),
                         level=float((np.abs(x) ** 2).sum((1, 2)).mean() / energy)))
    return dict(scene=scene['id'], family=scene['family'], split=scene['split'], budgets=rows)


def summarise(rows):
    """Scene-balanced RMS error (%) and mean tail fractions (%) per family and overall."""
    out = {}
    for family in sorted({r['family'] for r in rows}) + ['all']:
        sel = [r for r in rows if family == 'all' or r['family'] == family]
        out[family] = dict(scenes=len(sel), budgets={})
        for i, v in enumerate(VARIANTS):
            b = [r['budgets'][i] for r in sel]
            out[family]['budgets'][v] = dict(
                error=100 * math.sqrt(np.mean([x['error2'] for x in b])),
                over1=100 * float(np.mean([x['over1'] for x in b])),
                over3=100 * float(np.mean([x['over3'] for x in b])),
                level=float(np.mean([x['level'] for x in b])))
    return out


def selection_score(summary):
    """Selection score: mean over budgets of log validation error (lower is better)."""
    return float(np.mean([math.log(summary['all']['budgets'][v]['error']) for v in VARIANTS]))


# ----------------------------------------------------------------------------- baselines

def raw_prediction(scene):
    return scene['x'][..., :EARLY].sum(-1)


def fit_linear_gains(scenes, layout='families'):
    """Real gains per buffer and budget, least squares on training pairs.

    Minimises the scene-balanced sum |sum_g c_g x_g - t|^2 / rms^2 over the buffers
    of a layout: the best fixed shrinkage of each buffer, including how much of the
    walk to add back. Returned as equivalent per-family gains [B, 8].
    """
    group = LAYOUTS[layout][0]
    gains = np.zeros((len(VARIANTS), len(FAMILIES)))
    for b in range(len(VARIANTS)):
        a = np.zeros((group.shape[1], group.shape[1]))
        r = np.zeros(group.shape[1])
        for s in scenes:
            x = np.einsum('spqf,fg->spqg', s['x'][b].astype(np.complex128), group)
            t = s['target'][None].astype(np.complex128)
            w = 1.0 / (x.shape[1] * np.mean(np.abs(t) ** 2))
            a += w * np.einsum('spqf,spqg->fg', x.conj(), x).real
            r += w * np.einsum('spqf,spq->f', x.conj(), np.broadcast_to(t, x.shape[:3])).real
        gains[b] = group @ np.linalg.solve(a, r)
    return gains


def linear_prediction(scene, gains):
    return np.einsum('bspqf,bf->bspq', scene['x'], gains.astype(np.float32))


def seed_mean_prediction(scene, n=4):
    """Mean of n seeds (reference only: needs n solves of one pair)."""
    raw = raw_prediction(scene)
    return np.broadcast_to(raw.mean(1, keepdims=True), raw.shape)


def baselines(scenes, gains, summed_gains, slot_gains):
    keep = np.zeros_like(gains)
    keep[:, :2] = 1                      # direct + first-order reflections only, no fit
    rows = dict(raw=[], zero=[], directR=[], linear=[], linearSlots=[], linearSummed=[], mean4=[])
    for s in scenes:
        raw = raw_prediction(s)
        rows['raw'].append(scene_metrics(raw, s))
        rows['zero'].append(scene_metrics(np.zeros_like(raw), s))
        rows['directR'].append(scene_metrics(linear_prediction(s, keep), s))
        rows['linear'].append(scene_metrics(linear_prediction(s, gains), s))
        rows['linearSlots'].append(scene_metrics(linear_prediction(s, slot_gains), s))
        rows['linearSummed'].append(scene_metrics(linear_prediction(s, summed_gains), s))
        rows['mean4'].append(scene_metrics(seed_mean_prediction(s), s))
    return {k: summarise(v) for k, v in rows.items()}


def solve_seconds(scenes):
    """Measured solve + readback seconds per pair, mean per budget."""
    return {v: float(np.mean(np.concatenate([s['seconds'][i].ravel() for s in scenes]))) for i, v in enumerate(VARIANTS)}


# ----------------------------------------------------------------------------- training

class Guard:
    """Battery pause (dlss_jobs hook) and a process RSS ceiling."""

    def __init__(self, rss_limit=RSS_LIMIT, period=30):
        import dlss_jobs
        self.power = dlss_jobs.PowerCheck(period)
        self.rss_limit = rss_limit

    def __call__(self):
        self.power()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # bytes on macOS
        if rss > self.rss_limit:
            raise MemoryError(f'peak RSS {rss / GIB:.2f} GiB exceeds the {self.rss_limit / GIB:.1f} GiB limit')


def predict(model, scene, device='cpu', chunk=4096):
    """Model predictions for every budget, seed and receiver of a scene: [B,S,P,Q] complex."""
    b, k, p, q, f = scene['x'].shape
    x = to_real(scene['x'].reshape(-1, q, f))
    counts = torch.from_numpy(scene['counts'].reshape(-1, q, f))
    budget = torch.from_numpy(np.repeat(np.arange(b), k * p))
    distance = torch.from_numpy(np.tile(scene['distance'], b * k))
    visible = torch.from_numpy(np.tile(scene['visible'], b * k))
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            sl = slice(i, i + chunk)
            y = model(x[sl].to(device), counts[sl].to(device), budget[sl].to(device), distance[sl].to(device),
                      visible[sl].to(device)).cpu().numpy()
            outs.append(y[..., 0] + 1j * y[..., 1])
    return np.concatenate(outs).reshape(b, k, p, q).astype(np.complex64)


def evaluate(model, scenes, device='cpu'):
    return summarise([scene_metrics(predict(model, s, device), s) for s in scenes])


def train(spec, train_scenes, val_scenes, device='cpu', log=print, guard=None, eval_every=2):
    """Train one spec; keep the state with the best validation selection score."""
    torch.manual_seed(spec['seed'])
    data = as_pairs(train_scenes, shuffle=spec['shuffle'], seed=spec['seed'])
    n = data['x'].shape[0]
    model = Denoiser(spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec['lr'], weight_decay=spec['decay'])
    steps = spec['epochs'] * math.ceil(n / spec['batch'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=spec['lr'], total_steps=steps, pct_start=.1)
    gen = torch.Generator().manual_seed(spec['seed'])
    history, best = [], None
    started = time.time()
    for epoch in range(1, spec['epochs'] + 1):
        model.train()
        order = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, spec['batch']):
            idx = order[i:i + spec['batch']]
            batch = {k: v[idx].to(device) for k, v in data.items()}
            pred = model(batch['x'], batch['counts'], batch['budget'], batch['distance'], batch['visible'])
            loss = loss_fn(pred, batch, spec)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
            if guard:
                guard()
        row = dict(epoch=epoch, loss=total / n, seconds=time.time() - started)
        if epoch % eval_every == 0 or epoch == spec['epochs']:
            summary = evaluate(model, val_scenes, device)
            row['val'] = {v: summary['all']['budgets'][v]['error'] for v in VARIANTS}
            row['score'] = selection_score(summary)
            if best is None or row['score'] < best[0]:
                best = (row['score'], epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            log(f'  epoch {epoch:3d} loss {row["loss"]:.4f} val ' +
                ' / '.join(f'{row["val"][v]:.0f}%' for v in VARIANTS) + f' ({row["seconds"]:.0f} s)')
        history.append(row)
    model.load_state_dict(best[2])
    return model.cpu(), dict(bestEpoch=best[1], bestScore=best[0], history=history,
                             trainSeconds=time.time() - started, pairs=n)


def save_run(folder, model, info, extra=None):
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(dict(spec=model.spec, state=model.state_dict()), folder / 'model.pt')
    record = dict(spec=model.spec, parameters=parameters(model), macs=macs(model), **info, **(extra or {}))
    (folder / 'metrics.json').write_text(json.dumps(record, indent=1) + '\n')
    return record


def load_run(folder):
    blob = torch.load(folder / 'model.pt', weights_only=False)
    model = Denoiser(blob['spec'])
    model.load_state_dict(blob['state'])
    return model.eval()


def run_one(name, spec, scenes, out, args, guard):
    folder = out / 'runs' / name
    if (folder / 'metrics.json').exists() and not args.force:
        print(f'{name}: exists, skipped')
        return json.loads((folder / 'metrics.json').read_text())
    print(f'{name}: {json.dumps({k: v for k, v in spec.items() if BASE.get(k) != v})}')
    model, info = train(spec, split(scenes, 'train'), split(scenes, 'validation'), guard=guard)
    record = save_run(folder, model, info, dict(
        validation=evaluate(model, split(scenes, 'validation')), train=evaluate(model, split(scenes, 'train'))))
    print(f'{name}: best epoch {info["bestEpoch"]}, score {info["bestScore"]:.4f}, {info["trainSeconds"]:.0f} s')
    return record


# ----------------------------------------------------------------------------- timing

def time_inference(model, sample, device, batches=(1, 4, 16, 64), reps=200, warmup=20):
    """Median wall time per call: host tensors in, features + forward, host tensors out."""
    model = model.to(device).eval()
    rows = {}
    for nb in batches:
        args = [v[:nb] for v in sample]
        times = []
        with torch.no_grad():
            for i in range(warmup + reps):
                t0 = time.perf_counter()
                y = model(*[a.to(device) for a in args])
                y = y.cpu()
                if i >= warmup:
                    times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        rows[str(nb)] = dict(callMs=1e3 * med, perPairMs=1e3 * med / nb, p90CallMs=1e3 * float(np.percentile(times, 90)))
    model.cpu()
    return rows


# ----------------------------------------------------------------------------- commands

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['baselines', 'sweep', 'ablate', 'test', 'timing'])
    p.add_argument('--dataset', type=Path, default=DATASET)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--epochs', type=int, default=None, help='override epochs (smoke runs)')
    p.add_argument('--only', nargs='*', default=None, help='subset of candidate/ablation names')
    p.add_argument('--seeds', type=int, default=3, help='initialisation seeds for the selected recipe')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    report = dd.verify(a.dataset, deep=True)
    manifest, scenes = load(a.dataset)
    print(f'dataset {report["datasetSha256"][:12]}: {len(scenes)} accepted scenes '
          f'({", ".join(f"{s} {len(split(scenes, s))}" for s in ["train", "validation", "test"])})')
    guard = Guard()
    gains = fit_linear_gains(split(scenes, 'train'))
    summed_gains = fit_linear_gains(split(scenes, 'train'), 'summed-early')
    slot_gains = fit_linear_gains(split(scenes, 'train'), 'slot-ranges')

    if a.command == 'baselines':
        out = dict(datasetSha256=report['datasetSha256'], gains=dict(families=FAMILIES, variants=VARIANTS, values=gains.tolist(),
                                                                     summedEarly=summed_gains.tolist(), slotRanges=slot_gains.tolist()),
                   seconds={sp: solve_seconds(split(scenes, sp)) for sp in ['train', 'validation', 'test']},
                   splits={sp: baselines(split(scenes, sp), gains, summed_gains, slot_gains) for sp in ['train', 'validation']})
        (a.out / 'baselines.json').write_text(json.dumps(out, indent=1) + '\n')
        for sp, b in out['splits'].items():
            print(sp, {k: ' / '.join(f'{v["all"]["budgets"][x]["error"]:.0f}%' for x in VARIANTS) for k, v in b.items()})
        return

    if a.command == 'sweep':
        names = a.only or list(CANDIDATES)
        records = {}
        for name in names:
            spec = spec_of(**CANDIDATES[name], **({'epochs': a.epochs} if a.epochs else {}))
            records[name] = run_one(name, spec, scenes, a.out, a, guard)
        allr = {n: json.loads((a.out / 'runs' / n / 'metrics.json').read_text())
                for n in CANDIDATES if (a.out / 'runs' / n / 'metrics.json').exists()}
        scores = {n: selection_score(r['validation']) for n, r in allr.items()}
        top = min(scores.values())
        best = min((n for n in allr if scores[n] <= top + SELECTION_TIE), key=lambda n: allr[n]['macs'])
        (a.out / 'selection.json').write_text(json.dumps(dict(
            rule=f'fewest MACs among candidates within {SELECTION_TIE} of the lowest mean log validation error '
                 '(scene-RMS, early field, mean over budgets)', bestScoring=min(scores, key=scores.get), selected=best,
            spec=allr[best]['spec'], scores={n: selection_score(r['validation']) for n, r in allr.items()},
            validation={n: {v: r['validation']['all']['budgets'][v]['error'] for v in VARIANTS} for n, r in allr.items()}),
            indent=1) + '\n')
        print('selected', best)
        return

    selection = json.loads((a.out / 'selection.json').read_text())
    base_spec = selection['spec']

    if a.command == 'ablate':
        names = a.only or (['selected'] + list(ABLATIONS))
        for name in names:
            changes = {} if name == 'selected' else ABLATIONS[name]
            for seed in range(a.seeds):
                spec = dict(base_spec, **changes, seed=seed, **({'epochs': a.epochs} if a.epochs else {}))
                run_one(f'{name}-s{seed}', spec, scenes, a.out, a, guard)
        return

    if a.command == 'test':
        target = a.out / 'test.json'
        if target.exists() and not a.force:
            raise SystemExit('test.json exists: the test split is evaluated once (use --force to record a re-run)')
        test = split(scenes, 'test')
        out = dict(datasetSha256=report['datasetSha256'], selected=selection['selected'],
                   evaluatedAt=time.strftime('%Y-%m-%d %H:%M:%S'), baselines=baselines(test, gains, summed_gains, slot_gains), models={})
        for name in ['selected'] + list(ABLATIONS):
            for seed in range(a.seeds):
                folder = a.out / 'runs' / f'{name}-s{seed}'
                if folder.exists():
                    out['models'][f'{name}-s{seed}'] = evaluate(load_run(folder), test)
        target.write_text(json.dumps(out, indent=1) + '\n')
        print({k: ' / '.join(f'{v["all"]["budgets"][x]["error"]:.0f}%' for x in VARIANTS) for k, v in out['models'].items()})
        return

    if a.command == 'timing':
        test = split(scenes, 'test')
        data = as_pairs(test[:2])
        order = torch.randperm(data['x'].shape[0], generator=torch.Generator().manual_seed(0))[:64]
        sample = [data[k][order] for k in ['x', 'counts', 'budget', 'distance', 'visible']]
        out = dict(machine='Apple M5, 16 GB, shared desktop, nice 19', torch=torch.__version__, models={})
        for name in ['selected-s0'] + [f'{n}-s0' for n in ABLATIONS if n != 'shuffled-labels']:
            folder = a.out / 'runs' / name
            if not folder.exists():
                continue
            model = load_run(folder)
            row = dict(parameters=parameters(model), macs=macs(model))
            torch.set_num_threads(1)
            row['cpu1'] = time_inference(model, sample, 'cpu')
            torch.set_num_threads(a.threads)
            if torch.backends.mps.is_available():
                row['mps'] = time_inference(model, sample, 'mps')
                torch.mps.empty_cache()
            out['models'][name] = row
            print(name, json.dumps({k: {b: round(v['perPairMs'], 4) for b, v in row[k].items()} for k in ['cpu1', 'mps'] if k in row}))
        (a.out / 'timing.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
