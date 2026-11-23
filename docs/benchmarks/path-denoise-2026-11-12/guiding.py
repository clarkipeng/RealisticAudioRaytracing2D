"""README part 50. Noise per update in the Church's standing, walking (down the nave) and outside tracks, with and without
path guiding: rendered by MeasurePathPersistentSequences from an items file (the live settings, `guiding` 0 or 1), scored
against part 25's 8 N targets on updates 32-63 of both seeds.

    python guiding.py RAWROOT [denoiser tag]       # RAWROOT/<track>/n<N>-g<0|1>/low-<seed>; default tag p64

(The `guiding` item field and the setting come with guiding.patch beside this file; without it every run is unguided.)

Printed per track and configuration:
  - noisy input and denoised: window / band level / decay curve errors in dB (path_denoise.metrics, as parts 25-33);
  - the late field (0.2-2 s) per update at 125 / 500 / 2,000 Hz: median over mean and standard deviation over mean of its
    level across the 64 updates of both seeds, divided by the target's level at the same update (so a walking listener's
    own change of level cancels), as part 20's table.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import path_denoise as pdn  # noqa: E402
import path_persistent_denoise as ppd  # noqa: E402

TARGETS = ROOT / '.context/path-dataset-persistent/raw/church'
BANDS = (0, 2, 4)   # 125, 500, 2,000 Hz


def load(d):
    r = json.loads((d / 'run.json').read_text()); B = r['bins']
    return (np.fromfile(d / 'histogram.f32', '<f4').reshape(r['updates'], 48, B),
            np.fromfile(d / 'counts.u32', '<u4').reshape(r['updates'], B), r)


def late(h, edges):
    """Band energies of the late field (0.2-2 s) per update: [update][band]."""
    keep = (edges[:-1] >= .2) & (edges[1:] <= 2.0)
    return h[:, :, keep].sum(-1).reshape(h.shape[0], 6, 8).sum(1)


def main(root, tag='p64'):
    root = Path(root)
    edges = np.fromfile(TARGETS / 'standing/low-0/bin-edges.f32', '<f4')
    pdn.EDGES = edges
    net = ppd.load_model(tag)
    for track in ('standing', 'walking', 'outside'):
        target = np.mean([load(h)[0].astype(np.float64) for h in sorted((TARGETS / track).glob('high-*'))], 0)
        target_late = late(target, edges)
        for config in sorted(p.name for p in (root / track).glob('n*-g*')):
            runs = [load(d) for d in sorted((root / track / config).glob('low-*'))]
            t = dict(low=np.stack([r[0] for r in runs]), counts=np.stack([r[1] for r in runs]), target=target.astype(np.float32))
            t['emas'] = [[ppd.ema(t['low'][s].astype(np.float64), d).astype(np.float32) for d in ppd.DECAYS] for s in range(len(runs))]
            S = ppd.segment(t['low'].shape[-1])
            noisy, clean = [], []
            for tr, s, u, x, m, nn, scale in ppd.scored([t]):
                noisy.append(pdn.metrics(t['low'][s][u], target[u]))
                clean.append(pdn.metrics(ppd.run(net, S, x, m, nn) * scale, target[u]))
            ratio = np.concatenate([late(r[0].astype(np.float64), edges) / np.maximum(target_late, 1e-30) for r in runs])[:, list(BANDS)]
            stats = ' '.join(f"{np.median(ratio[:, b]) / ratio[:, b].mean():.2f}|{ratio[:, b].std() / ratio[:, b].mean():.2f}" for b in range(len(BANDS)))
            ms = np.mean([r[2]['updateMs'] for r in runs])
            fmt = lambda rows: '/'.join(f"{np.mean([r[q] for r in rows]):.2f}" for q in ('window', 'level', 'edc'))
            print(f"{track:8s} {config:10s} {ms:5.1f} ms  noisy {fmt(noisy)}  denoised {fmt(clean)}  late median|sd over mean {stats}", flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
