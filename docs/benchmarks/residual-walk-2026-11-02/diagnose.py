"""Residual-walk root-cause diagnosis from GPU-captured walk paths.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY docs/benchmarks/residual-walk-2026-11-02/diagnose.py jobs > jobs.json
    nice -n 19 $PY docs/benchmarks/residual-walk-2026-11-02/unity_jobs.py jobs.json
    $PY docs/benchmarks/residual-walk-2026-11-02/diagnose.py analyze

The Unity test TakeChecks.MeasureResidualWalk records every walk arrival with its
ray and bounce slot, the captured event segments (estimator weight after each
interaction) and the scene's edge/triangle records. walkmodel.py re-evaluates
each edge vertex in float64 (kernel, leg, selection probability, height density)
and must reproduce the captured GPU weights before any factor is interpreted.
Writes diagnosis.json and fig2-step-factors.png next to this file.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from walkmodel import Scene  # noqa: E402

PILOT = ROOT / '.context/audio-dlss-data/scenes'
OUT = ROOT / '.context/residual-walk/diagnosis'
KINDS = ['none', 'trans', 'refl', 'scat', 'diff', 'listen', 'escape', 'reject']
# (name, scene, variant, receivers): pilot rigid and preset scenes with the largest walk peaks.
CASES = [('barriers-rigid', 'barriers-0001', 'tracer-rigid', [20, 45]),
         ('canopy-rigid', 'canopy-0004', 'tracer-rigid', [30, 22]),
         ('barriers-preset', 'barriers-0001', 'tracer', [20, 45]),
         ('canopy-preset', 'canopy-0004', 'tracer', [30, 22])]
SEED, RAYS, BOUNCES = 1234567, 16384, 8


def jobs():
    # All bins of a ray share its path (the walk's random stream ignores the bin), so
    # a 4-point FFT at 500 Hz keeps readback small; the 125 Hz bin is captured.
    return [dict(scene=str(PILOT / scene), variant=variant, output=str(OUT / name), receivers=receivers,
                 seeds=[SEED], raysPerBin=RAYS, completeRouteSamples=64, captureBin=1, topRays=0,
                 maxBounces=BOUNCES, fftSize=4, sampleRate=500) for name, scene, variant, receivers in CASES]


def load(folder, receiver):
    raw = np.fromfile(folder / 'edges.f32', '<f4').reshape(-1, 20)
    edges = dict(a=raw[:, 0:3], b=raw[:, 4:7], face=raw[:, 8:11], angle=raw[:, 11], normal=raw[:, 12:15],
                 ids=raw[:, 16:].copy().view('<u4'))
    scene = Scene(edges, np.fromfile(folder / 'triangles.f32', '<f4').reshape(-1, 11))
    per = 2 + 2 * BOUNCES
    x = np.fromfile(folder / f'r{receiver}-s{SEED}-capture.f32', '<f4')
    x = x.reshape(-1, 2 + per * 12)
    return scene, x[:, 2:].reshape(len(x), per, 12)


def transitions(case):
    """Every consecutive pair of interactions in every captured ray, with its weight ratio.

    For edge-to-edge steps the shader divides by the *incoming* leg one step later;
    `g` (measured GPU ratio times incoming/outgoing leg) re-pairs each leg with the step
    that created it: K * cw / leg times opacity over the sampling density of the new vertex.
    """
    name, scene_id, variant, receivers = case
    folder = OUT / name
    config = json.loads((PILOT / scene_id / 'scene.json').read_text())
    source, points = np.array(config['source']), np.array(config['points']).reshape(-1, 3)
    rows, other = [], []
    for receiver in receivers:
        scene, seg = load(folder, receiver)
        listener = points[receiver]
        for ray in seg:
            verts, prev_w, prev_kind = [(None, source)], None, 'source'
            for b in range(BOUNCES):
                g = ray[2 + 2 * b]
                kind = KINDS[int(g[8])]
                if kind == 'none':
                    break
                if kind in ('escape', 'reject'):
                    other.append((prev_kind, kind, 0.0))
                    break
                w = float(g[11])
                if prev_w:
                    other.append((prev_kind, kind, w / prev_w))
                if kind == 'diff' and prev_kind == 'diff' and prev_w:
                    p = g[4:7].astype(float)
                    e, (e0, z0), zm = scene.edge_at(p), verts[-1], verts[-2][1]
                    sel, _ = scene.selection(e, z0, listener)
                    dens = scene.density(e, scene.height(e, p), z0, listener)
                    K = scene.kernel(e0, zm, z0, p)
                    cw = scene.connection_weight(e0, e)
                    inleg, leg = np.linalg.norm(z0 - zm), np.linalg.norm(p - z0)
                    model = cw * K / inleg * 2 / (sel * dens)
                    rows.append(dict(f=w / prev_w, model=model, g=w / prev_w * inleg / leg, K=K, cw=cw,
                                     inleg=inleg, leg=leg, pdf=sel * dens))
                verts.append((scene.edge_at(g[4:7].astype(float)) if kind == 'diff' else None, g[4:7].astype(float)))
                prev_w, prev_kind = w, kind
    return rows, other


def hill(values, k=50):
    a = np.sort(np.abs(values))[::-1]
    return float(1 / np.mean(np.log(a[:k] / a[k]))) if len(a) > k else None


def analyze():
    report = dict(seed=SEED, raysPerBin=RAYS, maxBounces=BOUNCES, cases=[])
    curves = {}
    for case in CASES:
        rows, other = transitions(case)
        f = np.array([r['f'] for r in rows])
        model = np.array([r['model'] for r in rows])
        g = np.array([r['g'] for r in rows])
        cw = np.array([r['cw'] for r in rows])
        after_edge = [o for o in other if o[0] == 'diff']
        continued = sum(1 for o in after_edge if o[1] == 'diff') / max(1, len(after_edge))
        big = np.abs(f) > 1e-3
        surface = np.array([o[2] for o in other if o[0] in ('refl', 'scat', 'source') and o[1] in ('refl', 'scat', 'trans')])
        entry = dict(case=case[0], scene=case[1], variant=case[2], receivers=case[3], edgeToEdgeSteps=len(rows),
                     modelAgreementWithin1Percent=float(np.mean(np.abs(model[big] - f[big]) <= .01 * np.abs(f[big]))),
                     probabilityEdgeAfterEdge=continued,
                     meanAbsTransfer=float(np.mean(np.abs(g))), meanSignedTransfer=float(np.mean(g)),
                     secondMoment=float(np.mean(g ** 2)), medianAbs=float(np.median(np.abs(g))),
                     absTransferPerStep=float(np.mean(np.abs(g)) * continued),
                     signedTransferPerStep=float(np.mean(g) * continued),
                     secondMomentPerStep=float(np.mean(g ** 2) * continued),
                     sharedFaceSteps=int(np.sum(cw < 1)), sharedFaceMeanAbs=float(np.mean(np.abs(g[cw < 1]))),
                     otherMeanAbs=float(np.mean(np.abs(g[cw == 1]))), tailIndexTop50=hill(g),
                     rawRatioMax=float(np.max(np.abs(f))), rawRatioMinIncomingLeg=float(np.min([r['inleg'] for r in rows])),
                     surfaceSteps=int(len(surface)),
                     surfaceMeanAbs=float(np.mean(np.abs(surface))) if len(surface) else None,
                     surfaceSecondMoment=float(np.mean(surface ** 2)) if len(surface) else None,
                     surfaceMax=float(np.max(np.abs(surface))) if len(surface) else None)
        report['cases'].append(entry)
        curves[case[0]] = (np.sort(np.abs(g))[::-1], surface)
    (HERE / 'diagnosis.json').write_text(json.dumps(report, indent=1) + '\n')
    figure(curves, report)
    for c in report['cases']:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})


def figure(curves, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parent / 'denoise-pilot-2026-10-29'))
    from analyze import COLORS, INK, MUTED, style
    style(plt)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, (a, _) in curves.items():
        family, variant = name.split('-')
        ccdf = np.arange(1, len(a) + 1) / len(a)
        ax.loglog(a, ccdf, color=COLORS[family], ls='-' if variant == 'rigid' else '--', lw=1.6,
                  label=f'{family}, {"rigid" if variant == "rigid" else "presets"}')
    surface = np.sort(np.abs(np.concatenate([s for _, s in curves.values()])))[::-1]
    ax.loglog(surface, np.arange(1, len(surface) + 1) / len(surface), color=MUTED, lw=1.2, label='surface events (presets)')
    ax.axvline(1, color=MUTED, lw=.8, ls=':')
    mean = np.mean([c['absTransferPerStep'] for c in report['cases']])
    ax.text(1.3e-3, 1.3e-4, f'Mean |factor| x P(another edge) = {mean:.1f} per extra edge (> 1: weights grow\n'
            'geometrically with chain length). Steps with factor 0 (inside a wedge) sit below the plateau.',
            color=MUTED, fontsize=7.5)
    ax.set_xlim(1e-3, 1e4); ax.set_ylim(8e-5, 1.5)
    ax.set_xlabel('|weight factor| of one extra interaction (edge: K·w/(leg·pdf))')
    ax.set_ylabel('Fraction of steps above')
    ax.set_title('Each edge-to-edge step multiplies the path weight by a heavy-tailed factor', color=INK, loc='left', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(HERE / 'fig2-step-factors.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['jobs', 'analyze'])
    a = p.parse_args()
    if a.command == 'jobs':
        print(json.dumps(jobs(), indent=1))
    else:
        analyze()


if __name__ == '__main__':
    main()
