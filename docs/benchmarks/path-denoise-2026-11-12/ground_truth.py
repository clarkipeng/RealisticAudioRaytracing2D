"""The denoiser against the wave ground truth of the method-comparison suite.

    PY=.context/fdtd-teacher/venv/bin/python
    B=docs/benchmarks/path-denoise-2026-11-12
    nice -n 19 $PY $B/ground_truth.py render      # path at 512 rays, four seeds, on every suite scene (guarded Unity)
    nice -n 19 $PY $B/ground_truth.py score       # write the variants beside the ground truth and score them

Variants, all from the same four 512-ray solves per receiver (independent seeds; a static listener's
last four frames): path-512 (each solve alone), path-512-k4 (their mean), path-512-denoised (the
denoiser of tools/path_denoise.py on the four), and path-2048 (the suite's own 2,048-ray solves: the
same rays in one solve, i.e. more rays at equal compute for the four frames).
They are written to .context/path-denoise/suite (scene.json and gt linked from the suite) and scored by
tools/method_compare.py score into .context/path-denoise/suite-score.
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import method_compare as mc  # noqa: E402
import method_runs as mr  # noqa: E402
import path_denoise as pdn  # noqa: E402

OUT = ROOT / '.context/path-denoise/suite'
SEEDS = [0, 1, 2, 3]
MODEL = 'all'


def scenes():
    root = mc.DEFAULT_ROOT
    return [e['id'] for e in json.loads((root / 'suite.json').read_text())['scenes'] if (root / e['id'] / 'gt' / 'ref' / '0').exists()]


def render():
    mr.make_jobs('denoise-suite-512', scenes(), ['path'], [512], SEEDS)
    failures = mr.render('denoise-suite-512')
    if failures:
        raise SystemExit(f'{len(failures)} failures')


def to_ms(h, edges):
    """[R, 48, B] native histogram -> [R, 8, 2000] 1 ms band energies (each bin spread uniformly over its span,
    as MeasureMethods resamples)."""
    hb = h.reshape(h.shape[0], 6, 8, -1).sum(1)
    out = np.zeros((h.shape[0], 8, mc.TIME_BINS))
    for k in range(hb.shape[-1]):
        a, b = edges[k] * 1000, edges[k + 1] * 1000
        for i in range(int(np.floor(a)), min(int(np.ceil(b)), mc.TIME_BINS)):
            overlap = max(0., min(b, i + 1) - max(a, i))
            out[:, :, i] += hb[:, :, k] * overlap / (b - a)
    return out


def score():
    import torch
    net = pdn.model(); net.load_state_dict(torch.load(pdn.WORK / f'denoiser-{MODEL}.pt', map_location='cpu')); net.eval()
    for sid in scenes():
        src = mc.DEFAULT_ROOT / sid
        dst = OUT / sid
        dst.mkdir(parents=True, exist_ok=True)
        for name in ('scene.json', 'gt'):
            if not (dst / name).exists():
                (dst / name).symlink_to(src / name)
        if not (dst / 'path-2048').exists():
            (dst / 'path-2048').symlink_to(src / 'path')
        runs = [mr.RAW / sid / 'path' / 'r00512' / str(s) for s in SEEDS]
        meta_run = json.loads((runs[0] / 'run.json').read_text())
        edges = np.fromfile(runs[0] / 'bin-edges.f32', '<f4')
        R, B = meta_run['receivers'], len(edges) - 1
        hist = np.stack([np.fromfile(r / 'histogram.f32', '<f4').reshape(R, 48, B) for r in runs])        # [seed, R, 48, B]
        counts = np.stack([np.fromfile(r / 'counts.u32', '<u4').reshape(R, B).astype(np.float32) for r in runs])
        S = torch.from_numpy(pdn.segment_matrix(B))
        den = np.zeros(hist.shape[1:])
        with torch.no_grad():
            for i in range(R):
                frames = [hist[s, i] for s in range(len(SEEDS))]
                if not any(f.sum() > 0 for f in frames):
                    continue
                x, m, scale = pdn.features(frames, [counts[s, i] for s in range(len(SEEDS))])
                den[i] = pdn.predict(net, *(torch.from_numpy(v)[None] for v in (x, m, pdn.normaliser(m))), S)[0].numpy() * scale
        scene_sha = json.loads((src / 'path' / 'r02048' / '0' / 'meta.json').read_text())['sceneSha256']
        base = dict(sceneId=sid, sceneSha256=scene_sha, code=dict(name='MethodPath.hlsl + tools/path_denoise.py', sha256=meta_run['codeSha256']),
                    wallSeconds=meta_run.get('wallSeconds', 0), validSeconds=2.0, budgetUnit='rays (all bands per ray)',
                    notes='docs/benchmarks/path-denoise-2026-11-12/ground_truth.py')
        for s in range(len(SEEDS)):
            mc.write_output(dst / 'path-512' / 'r00512' / str(s), dict(base, method='path-512', budget='r00512', seed=s, budgetValue=512), to_ms(hist[s], edges))
        mc.write_output(dst / 'path-512-k4' / 'r00512' / '0', dict(base, method='path-512-k4', budget='r00512', seed=0, budgetValue=512), to_ms(hist.mean(0), edges))
        mc.write_output(dst / 'path-512-denoised' / 'r00512' / '0', dict(base, method='path-512-denoised', budget='r00512', seed=0, budgetValue=512), to_ms(den, edges))
        print(sid, R, 'receivers', flush=True)
    report = mc.score(OUT, OUT.parent / 'suite-score')
    rows = report['rows']
    table = {}
    for method in ('path-512', 'path-512-k4', 'path-512-denoised', 'path-2048'):
        sel = [r for r in rows if r['method'] == method]
        table[method] = {g: {k: mc.aggregate([r for r in sel if r['group'] == g], ('method',), k).get((method,))
                             for k in ('level', 'EDT', 'T30', 'C50', 'C80', 'etc')} for g in ('all', 'view', 'shadow')}
    (OUT.parent / 'suite-table.json').write_text(json.dumps(table, indent=1) + '\n')
    for method, groups in table.items():
        print(f'{method:18s}', ' | '.join(f"{g}: " + ' '.join(f"{k} {v:.2f}" if v is not None else f'{k} -' for k, v in m.items()) for g, m in groups.items()))


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'render':
        render()
    elif command == 'score':
        score()
    else:
        raise SystemExit(__doc__)
