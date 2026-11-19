"""README parts 28-30. Lottery noise: Church tracks, runs under .context/path-optimise/lottery/<config>/<track>/low-{0,1}, scored against
the persistent dataset's 8N targets (raw/church/<track>/high-*). plan CONFIGS... | score
(a config: name=population,cap[,edge slice updates[,resampling (part 32's systematic, since removed)]]; parts 31-34 write to .context/path-optimise/r31 with LOTTERY_DIR)"""
import json, sys, copy
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import os
D = ROOT / '.context/path-dataset-persistent'; W = Path(os.environ.get('LOTTERY_DIR', ROOT / '.context/path-optimise/lottery'))
TRACKS = ['standing', 'walking']
def plan(configs):
    items = json.loads((D / 'items.json').read_text())['items']
    out = []
    for cfg in configs:                       # name=population,cap
        name, v = cfg.split('='); vals = list(map(int, v.split(','))); n, cap = vals[0], vals[1]; sl = vals[2] if len(vals) > 2 else 1
        resampling = vals[3] if len(vals) > 3 else 0
        for it in items:
            if it['kind'] == 'take' and Path(it['output']).name in TRACKS:
                x = copy.deepcopy(it); x.update(output=str(W / name / Path(it['output']).name), population=n, iterationCap=cap, edgeSliceUpdates=sl, resampling=resampling, highSeeds=[], seeds=[0, 1])
                out.append(x)
    W.mkdir(parents=True, exist_ok=True); (W / 'items.json').write_text(json.dumps(dict(items=out)))
    print(len(out), 'items')
def load(d):
    r = json.loads((d / 'run.json').read_text()); B = r['bins']
    return np.fromfile(d / 'histogram.f32', '<f4').reshape(r['updates'], 48, B).astype(np.float64), r
def score():
    import path_denoise as pdn
    pdn.EDGES = np.fromfile(D / 'raw/church/standing/low-0/bin-edges.f32', '<f4')
    configs = {'base11000 (dataset)': {t: D / 'raw/church' / t for t in TRACKS}}
    for c in sorted(W.iterdir()):
        if c.is_dir(): configs[c.name] = {t: c / t for t in TRACKS}
    for name, tracks in configs.items():
        rows = []; pair = []
        for t, d in tracks.items():
            if (d / 'low-1' / 'run.json').exists():
                a, _ = load(d / 'low-0'); b, _ = load(d / 'low-1')
                pair += [pdn.metrics(a[u], b[u]) for u in range(len(a))]
            target = np.mean([load(h)[0] for h in sorted((D / 'raw/church' / t).glob('high-*'))], 0)
            for s in ('low-0', 'low-1'):
                if not (d / s / 'run.json').exists(): continue
                h, r = load(d / s)
                rows += [pdn.metrics(h[u], target[u]) for u in range(len(h))]
                ms = r['updateMs']
        if pair: print(f"{name:22s} seed against seed: window {np.mean([x['window'] for x in pair]):.2f} level {np.mean([x['level'] for x in pair]):.2f} edc {np.mean([x['edc'] for x in pair]):.2f}")
        if rows: print(f"{name:22s} against 8N: window {np.mean([x['window'] for x in rows]):.2f} level {np.mean([x['level'] for x in rows]):.2f} edc {np.mean([x['edc'] for x in rows]):.2f}  ({len(rows)} updates, {ms:.1f} ms/update in the dataset run)")
if __name__ == '__main__':
    plan(sys.argv[2:]) if sys.argv[1] == 'plan' else score()
