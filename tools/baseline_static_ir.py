"""The "static IR" baseline for the method comparison: a typical game convolution reverb.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY tools/baseline_static_ir.py [--root R]       # writes <scene>/baseline-static-ir/ref/0 for every scene
    $PY tools/method_compare.py score                 # scores it beside the methods

Per scene, the ground truth's response at ONE reference receiver (the in-view receiver
nearest the centroid of the in-view receivers) is used at every receiver, with only the
direct sound adjusted: the reference's direct sound is removed and each receiver gets
its own, at the true source-receiver delay with energy 1/d^2 and no occlusion (shadowed
receivers hear it too). Everything else (reflections, reverberation) is the reference's.

The ground truth's histograms are band-filtered (FORMAT.md), so the direct sounds are
the band's energy kernel (method_compare.energy_kernel) scaled by 1/d^2: the reference's
is subtracted (clipped at zero where interference with early reflections made the band
energy smaller) and each receiver's added; the output is therefore band limited too.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import method_compare as mc  # noqa: E402

METHOD = 'baseline-static-ir'


def reference(points, shadow):
    view = np.nonzero(~shadow)[0] if (~shadow).any() else np.arange(len(points))
    centre = points[view].mean(0)
    return int(view[np.argmin(np.linalg.norm(points[view] - centre, axis=1))])


def add_direct(e, b, distance, c, sign=1.0):
    """Add (sign +1) or remove (-1) a direct sound of energy 1/d^2 to band b of histogram e [T]."""
    k = mc.energy_kernel(mc.BANDS[b])
    h = len(k) // 2
    at = int(np.floor(distance / c / mc.BIN_SECONDS))
    for i, w in enumerate(k):
        j = at + i - h
        if 0 <= j < len(e):
            e[j] += sign * w / distance ** 2
        elif j < 0:
            e[0] += sign * w / distance ** 2


def build(root, scene_id):
    root = Path(root)
    scene_path = root / scene_id / 'scene.json'
    raw = scene_path.read_bytes()
    scene = json.loads(raw)
    gt = mc.read_output(root / scene_id / 'gt' / 'ref' / '0')
    gmeta, ge = gt[0], gt[1]
    points = np.asarray(scene['points'], float).reshape(-1, 3)
    source = np.asarray(scene['source'], float)
    shadow = np.asarray(scene.get('shadow', np.zeros(len(points), int)), bool)
    c = float(scene.get('speedOfSound', 343.0))
    ref = reference(points, shadow)
    d = np.maximum(np.linalg.norm(points - source, axis=1), .1)
    out = np.full(ge.shape, np.nan, np.float32)
    for b in range(len(mc.BANDS)):
        if not np.isfinite(ge[ref, b]).all():
            continue
        reverb = np.array(ge[ref, b], float)
        add_direct(reverb, b, d[ref], c, -1.0)
        reverb = np.maximum(reverb, 0)
        for r in range(len(points)):
            e = reverb.copy()
            add_direct(e, b, d[r], c, 1.0)
            out[r, b] = e
    meta = dict(method=METHOD, budget='ref', seed=0, budgetValue=1, budgetUnit='one measured response',
                sceneId=scene_id, sceneSha256=mc.sha256_bytes(raw),
                code=dict(name='tools/baseline_static_ir.py', sha256=mc.sha256_file(Path(__file__))),
                wallSeconds=0.0, validSeconds=gmeta['validSeconds'], bandLimited=True,
                bandsProvided=[float(f) for b, f in enumerate(mc.BANDS) if np.isfinite(out[:, b]).any()],
                referenceReceiver=ref,
                notes=f'ground truth of receiver {ref} everywhere; direct sound replaced by each receiver\'s 1/d^2 at d/c, no occlusion')
    mc.write_output(root / scene_id / METHOD / 'ref' / '0', meta, out)
    return ref


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=str(mc.DEFAULT_ROOT))
    a = p.parse_args()
    suite = json.loads((Path(a.root) / 'suite.json').read_text())
    for entry in suite['scenes']:
        if (Path(a.root) / entry['id'] / 'gt' / 'ref' / '0').exists():
            print(entry['id'], 'reference receiver', build(a.root, entry['id']))


if __name__ == '__main__':
    main()
