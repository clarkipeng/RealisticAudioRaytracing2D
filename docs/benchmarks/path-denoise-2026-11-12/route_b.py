"""README parts 28-29. Noise after the denoiser at lower populations (Church standing and walking, updates 32-63, 2 seeds) against 8N."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / 'tools'))
import path_persistent_denoise as ppd, path_denoise as pdn, torch
D = ROOT / '.context/path-dataset-persistent/raw/church'; W = ROOT / '.context/path-optimise/lottery'
def load(d):
    r = json.loads((d / 'run.json').read_text()); B = r['bins']
    return (np.fromfile(d / 'histogram.f32', '<f4').reshape(r['updates'], 48, B), np.fromfile(d / 'counts.u32', '<u4').reshape(r['updates'], B))
pdn.EDGES = np.fromfile(D / 'standing/low-0/bin-edges.f32', '<f4')
net = ppd.model(); net.load_state_dict(torch.load(ppd.WORK / 'denoiser-persistent.pt', map_location='cpu')); net.eval()
configs = {'11000 (part 25 runs, no cap)': D}
configs.update({c.name: c for c in sorted(W.glob('cap16n*'))})
for name, base in configs.items():
    rows = {'noisy': [], 'denoised': []}
    for track in ('standing', 'walking'):
        if not (base / track / 'low-1' / 'run.json').exists(): continue
        lows = [load(base / track / f'low-{s}') for s in (0, 1)]
        target = np.mean([load(h)[0].astype(np.float64) for h in sorted((D / track).glob('high-*'))], 0)
        t = dict(low=np.stack([l[0] for l in lows]), counts=np.stack([l[1] for l in lows]), target=target.astype(np.float32))
        t['emas'] = [[ppd.ema(t['low'][s].astype(np.float64), d).astype(np.float32) for d in ppd.DECAYS] for s in range(2)]
        S = ppd.segment(t['low'].shape[-1])
        for tr, s, u, x, ms, n, scale in ppd.scored([t]):
            rows['noisy'].append(pdn.metrics(t['low'][s][u], target[u])); rows['denoised'].append(pdn.metrics(ppd.run(net, S, x, ms, n) * scale, target[u]))
    if rows['noisy']:
        print(f"{name:30s} " + '  '.join(f"{k}: " + '/'.join(f"{np.mean([r[m] for r in v]):.2f}" for m in ('window', 'level', 'edc')) for k, v in rows.items()) + f"  ({len(rows['noisy'])} updates)")
