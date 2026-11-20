"""README parts 33-34. Noise in the Church at each population of the auto-budget's range, before and after each
denoiser: standing and walking, two seeds, updates 32-63, against part 25's 8 N targets (window / band level /
decay curve, dB). Inputs: tools/path_persistent_dataset.py plan-populations (the live settings).

    python populations.py persistent p64 [...]     # denoiser tags in .context/path-denoise/persistent
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import path_denoise as pdn  # noqa: E402
import path_persistent_dataset as pds  # noqa: E402
import path_persistent_denoise as ppd  # noqa: E402

TARGETS = ROOT / '.context/path-dataset-persistent/raw/church'
INPUTS = pds.POPULATION_WORK / 'raw/church'


def load(d):
    r = json.loads((d / 'run.json').read_text()); B = r['bins']
    return (np.fromfile(d / 'histogram.f32', '<f4').reshape(r['updates'], 48, B),
            np.fromfile(d / 'counts.u32', '<u4').reshape(r['updates'], B), r)


def main(tags):
    pdn.EDGES = np.fromfile(TARGETS / 'standing/low-0/bin-edges.f32', '<f4')
    nets = {t: ppd.load_model(t) for t in tags}
    for n in pds.CHURCH_POPULATIONS:
        rows = {k: [] for k in ['noisy'] + tags}
        ms = []
        for track in ('standing', 'walking'):
            runs = [load(INPUTS / track / f'n{n}-s{s}' / f'low-{s}') for s in pds.SEEDS]
            ms += [r[2]['updateMs'] for r in runs]
            target = np.mean([load(h)[0].astype(np.float64) for h in sorted((TARGETS / track).glob('high-*'))], 0)
            t = dict(low=np.stack([r[0] for r in runs]), counts=np.stack([r[1] for r in runs]), target=target.astype(np.float32))
            t['emas'] = [[ppd.ema(t['low'][s].astype(np.float64), d).astype(np.float32) for d in ppd.DECAYS] for s in range(len(runs))]
            S = ppd.segment(t['low'].shape[-1])
            for tr, s, u, x, m, nn, scale in ppd.scored([t]):
                rows['noisy'].append(pdn.metrics(t['low'][s][u], target[u]))
                for tag, net in nets.items():
                    rows[tag].append(pdn.metrics(ppd.run(net, S, x, m, nn) * scale, target[u]))
        print(f"N {n:6d} ({np.mean(ms):.1f} ms per update in the render)  " +
              '  '.join(f"{k}: " + '/'.join(f"{np.mean([r[q] for r in v]):.2f}" for q in ('window', 'level', 'edc')) for k, v in rows.items()), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
