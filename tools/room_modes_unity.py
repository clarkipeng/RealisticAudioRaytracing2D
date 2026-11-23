"""Room mode banks for the renderer (Assets/Scripts/Simulation/RoomModes.cs) and the Unity checks' references.

    PY=.context/fdtd-teacher/venv/bin/python
    $PY tools/room_modes_unity.py reference SCENE [--seed 0]   # export the suite room's bank; save room_modes.combine for
                                                              # the path method's run (TakeChecks.RoomModesMatchThePython...),
                                                              # with the probe as documented (see probe below)
    $PY tools/room_modes_unity.py church                       # export the Church's bank with its sidecar (geometry hash,
                                                              # every geometry object's pose at frame 0 of the take)

Outputs go to .context/bass-and-diffraction (banks are large: they stay out of git; a build ships one in StreamingAssets).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import method_compare as mc  # noqa: E402
import room_modes as rm  # noqa: E402

WORK = ROOT / '.context/bass-and-diffraction'
CHURCH_TAKE = ROOT / '.context/energy-audio/church/speech-walk/take'
STATE_FLOATS = 20          # per object and frame: position, quaternion, four motion vectors, flags (AcousticTake.State)
GEOMETRY, SOURCE, LISTENER, ACTIVE = 2, 4, 8, 1


def probe(bank, points):
    """room_modes.probe as its docstring states it (fourth order), which RoomModes.Probe implements.

    room_modes.probe reads its second differences from `parts[a][rows].tocsc()` with columns `np.unique(sub.indices)`:
    a CSC matrix's indices are row numbers, so the columns it multiplies are the wrong cells and the correction is
    about zero (trilinear in effect; on the analytic box at h = 0.125 m to 400 Hz its median shape error is 6 %, against
    0.24 % with the stencil's own neighbours). Here the neighbours are those of each corner cell along each axis.
    """
    edges = [np.asarray(bank[f'edges{a}'], float) for a in range(3)]
    solid = bank['solid']
    centres = [0.5 * (e[1:] + e[:-1]) for e in edges]
    widths = [np.diff(e) for e in edges]
    index = -np.ones(solid.shape, np.int64)
    cells = np.argwhere(solid < 0)
    index[tuple(cells.T)] = np.arange(len(cells))
    psi = bank['psi']

    def air(c):
        return index[tuple(c)] if all(0 <= c[a] < solid.shape[a] for a in range(3)) else -1

    out = np.zeros((len(points), psi.shape[1]))
    for p, x in enumerate(np.atleast_2d(points)):
        c0 = [int(np.clip(np.searchsorted(edges[a], x[a]) - 1, 0, len(centres[a]) - 1)) for a in range(3)]
        if air(c0) < 0:
            raise ValueError(f'point {x} lies in a solid cell')
        t, span, step = [0.0] * 3, [0.0] * 3, [0] * 3
        for a in range(3):
            d = x[a] - centres[a][c0[a]]
            nb = list(c0)
            nb[a] += 1 if d >= 0 else -1
            if air(nb) >= 0:
                span[a], step[a] = abs(centres[a][nb[a]] - centres[a][c0[a]]), nb[a] - c0[a]
            else:
                span[a] = widths[a][c0[a]]
            t[a] = abs(d) / span[a]
        for delta in np.ndindex(2, 2, 2):
            w = np.prod([t[a] if delta[a] else 1 - t[a] for a in range(3)])
            if w == 0:
                continue
            target = [c0[a] + delta[a] * step[a] for a in range(3)]
            for a in range(3):                  # a solid diagonal corner: step back along an axis
                if air(target) >= 0:
                    break
                target[a] = c0[a]
            j = air(target)
            value = np.asarray(psi[j], np.float64).copy()
            for a in range(3):
                factor = t[a] * (1 - t[a]) * span[a] ** 2 / 2
                for side in (-1, 1):
                    nb = list(target)
                    nb[a] += side
                    o = air(nb)
                    if o >= 0 and factor:
                        value -= factor / (widths[a][target[a]] * 0.5 * (widths[a][target[a]] + widths[a][nb[a]])) * (
                            np.asarray(psi[o], np.float64) - psi[j])
            out[p] += w * value
    return out


def export(bank_dir, out):
    if not out.exists():
        rm.export(rm.load_bank(bank_dir), out)
    return out


def reference(scene_id, seed=0, hmax=0.125, budget='r02048'):
    root = mc.DEFAULT_ROOT
    scene = json.loads((root / scene_id / 'scene.json').read_text())
    bank_dir = WORK / f'bank-{scene_id}-{hmax:g}'
    bytes_path = export(bank_dir, WORK / f'{scene_id}.roommodes.bytes')
    bank = rm.load_bank(bank_dir)
    a = probe(bank, np.vstack([scene['source'], np.asarray(scene['points'], float).reshape(-1, 3)]))
    k, S, a_s, a_r = bank['k'], bank['surface'], a[0], a[1:]
    path_dir = root / scene_id / 'path' / budget / str(seed)
    _, pe, _, _ = mc.read_output(path_dir)
    out, _, info = rm.combine(scene, k, S, a_s, a_r, pe.astype(float))
    names = [str(x) for x in bank['materials']]
    mats = scene['materials']
    np.asarray(out, '<f4').tofile(WORK / f'{scene_id}.reference.f32')
    spec = dict(sceneId=scene_id, bank=str(bytes_path), pathEnergy=str(path_dir / 'energy.f32'),
                reference=str(WORK / f'{scene_id}.reference.f32'), source=scene['source'], points=scene['points'],
                lineOfSight=[float(v) for v in scene.get('lineOfSight', [1] * (len(scene['points']) // 3))],
                materialNames=names, absorption=[v for n in names for v in mats[n]['absorption']],
                transmission=[v for n in names for v in mats[n]['transmission']],
                admittance=[v for n in names for v in mats[n]['admittance']], speedOfSound=rm.C,
                schroederHz=info['schroederHz'], pathShare=info['pathShare'], receivers=len(scene['points']) // 3)
    (WORK / f'{scene_id}.reference.json').write_text(json.dumps(spec, indent=1) + '\n')
    print(json.dumps({key: spec[key] for key in ('bank', 'reference', 'schroederHz', 'receivers')}))


def church(h=0.75):
    out = export(WORK / f'bank-church-{h:g}', WORK / 'church.roommodes.bytes')
    meta = json.loads((CHURCH_TAKE / 'take.json').read_text())
    scene = meta['scene']
    objects = len(scene['objects'])
    frames = np.fromfile(CHURCH_TAKE / 'states.bin', dtype=np.dtype([('sample', '<i8'), ('state', '<f4', (objects, STATE_FLOATS))]), count=1)
    state = frames['state'][0]
    flags = state[:, 19].view('<u4')
    poses = []
    for i in range(objects):   # every reflecting object, at its pose when the bank was baked (the doors closed)
        if flags[i] & GEOMETRY and not flags[i] & (SOURCE | LISTENER):
            poses.append(dict(index=i, position=[float(v) for v in state[i, :3]], rotation=[float(v) for v in state[i, 3:7]],
                              active=bool(flags[i] & ACTIVE)))
    sidecar = dict(geometryHash=scene['geometryHash'], poses=poses)
    Path(str(out) + '.json').write_text(json.dumps(sidecar, indent=1) + '\n')
    report = json.loads((WORK / f'church-{h:g}.json').read_text())   # docs/benchmarks/.../church_modes.py bake
    names = [str(x) for x in rm.load_bank(WORK / f'bank-church-{h:g}')['materials']]
    spec = dict(sceneId='church', bank=str(out), take=str(CHURCH_TAKE), materialNames=names,
                admittance=[float(v) for n in names for v in report['admittance'][n]], speedOfSound=rm.C,
                schroederHz=report['schroederHz'])
    (WORK / 'church.reference.json').write_text(json.dumps(spec, indent=1) + '\n')
    print(json.dumps(dict(bank=str(out), geometryHash=scene['geometryHash'], poses=len(poses))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['reference', 'church'])
    parser.add_argument('scene', nargs='?')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'reference':
        reference(args.scene, args.seed)
    else:
        church()


if __name__ == '__main__':
    main()
