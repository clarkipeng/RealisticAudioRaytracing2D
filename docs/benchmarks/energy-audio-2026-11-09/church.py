"""Realistic Church acoustics: published materials, interior proxies, and the speech walk re-capture.

    PY=.context/fdtd-teacher/venv/bin/python
    B=docs/benchmarks/energy-audio-2026-11-09
    nice -n 19 $PY $B/church.py inspect      # guarded Unity: inventory of acoustic and visual objects
    $PY $B/church.py plan                     # materials, assignments and proxies -> .context/energy-audio/church/plan.json
    nice -n 19 $PY $B/church.py apply        # guarded Unity: edits the runner's Church scene/materials, copies them back
    nice -n 19 $PY $B/church.py capture      # guarded Unity: the speech walk as a new take

All Unity work runs in the isolated runner (tools/dlss_jobs.py, tools/heavy_guard.py), never in the
user's editor. The runner first receives the repository's Church scene and materials; after
`apply` the changed scene and material assets (with .meta) are copied back into the repository.
"""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / 'tools'))
import listen  # noqa: E402
import church_transmission as ct  # noqa: E402
import dlss_jobs as dj  # noqa: E402

WORK = ROOT / '.context/energy-audio/church'
SPEECH = ROOT / '.context/church-motion/speech.f32'
SCENE = 'Assets/Scenes/Church.unity'
FREQUENCIES = [125, 250, 500, 1000, 2000, 4000, 8000]

# Materials (11 November). Main source: the calibrated church model of Planells, Segura, Barba,
# Cerda, Cibrian & Gimenez, "Analysis of acoustic variations due to different use configurations in
# the Basilica de Santa Maria de Elche", ICA 2016, paper 331, Table 2 (standardised absorption with the
# stone masonry's absorption and the diffusion coefficients calibrated against measured T30, EDT, C80
# in a stone basilica with barrel vaults, marble floor and wooden pews; acusvirt.upv.es). Their 4 kHz
# value is used at 8 kHz. Their window absorption is the classic ordinary-window-glass table
# (0.35, 0.25, 0.18, 0.12, 0.07, 0.04; e.g. Everest, Master Handbook of Acoustics), which also fits
# leaded panes: single glazing vibrates and absorbs at low frequencies. The copper roof's timber
# lining keeps M. Vorlaender, Auralization (Springer 2008), [wooden_lining] (no counterpart there).
# Until 10 November: Vorlaender's limestone/plaster/marble/3 mm glass tables and estimated scattering.
MATERIALS = {
    'Church stone': dict(absorption=[.038, .038, .04, .048, .05, .028, .028],      # Planells 2016: stone masonry (calibrated)
                         scattering=[.12, .13, .14, .15, .16, .17, .17],
                         note='Stone masonry walls: Planells et al. 2016 (calibrated basilica model)'),
    'Church plaster vault': dict(absorption=[.038, .038, .04, .048, .05, .028, .028],  # Planells 2016: vaults
                                 scattering=[.20, .24, .28, .32, .36, .40, .40],
                                 note='Vaults (ribs, mouldings): Planells et al. 2016'),
    'Church floor': dict(absorption=[.01, .01, .01, .01, .02, .02, .02],           # Planells 2016: marble floor
                         scattering=[.12, .13, .14, .15, .16, .17, .17],
                         note='Marble floor (joints, wear): Planells et al. 2016'),
    'Church wood': dict(absorption=[.12, .14, .16, .16, .16, .16, .16],            # Planells 2016: organ
                        scattering=[.20, .24, .28, .32, .36, .40, .40],
                        note='Organ case and tower: Planells et al. 2016 (organ)'),
    'Church door': dict(absorption=[.14, .10, .06, .08, .10, .10, .10],            # Planells 2016: wooden doors
                        scattering=[.12, .13, .14, .15, .16, .17, .17],
                        note='Wooden doors: Planells et al. 2016; transmission of the leaf with its 4 cm side gaps (church_transmission.py)'),
    'Church pews': dict(absorption=[.09, .14, .16, .16, .15, .13, .13],            # Planells 2016: wooden pews
                        scattering=[.30, .40, .50, .60, .70, .70, .70],
                        note='Wooden pews (and the floor area of the pew blocks): Planells et al. 2016'),
    'Church pews occupied': dict(absorption=[.57, .61, .75, .86, .91, .86, .86],   # occupied wooden pews (classic table)
                                 scattering=[.30, .50, .60, .60, .70, .70, .70],
                                 note='Occupied wooden pews (classic table; 8 kHz = 4 kHz); not assigned (optional)'),
    'Church glass': dict(absorption=[.35, .25, .18, .12, .07, .04, .04],           # Planells 2016 / ordinary window glass
                         scattering=[.10, .10, .10, .10, .10, .10, .10],
                         note='Leaded windows: ordinary window glass (Planells et al. 2016; classic table)'),
    'Church copper roof': dict(absorption=[.27, .23, .22, .15, .10, .07, .06],     # [wooden_lining]
                               scattering=[.10, .15, .20, .30, .40, .40, .40],
                               note='Interior side of the copper roof: 12 mm wooden lining on frame [wooden_lining]; rafters and trusses (estimate)'),
}
ASSIGNMENTS = {
    'StoneBrickWalls': 'Church stone', 'StoneBeams': 'Church stone', 'SideStairs': 'Church stone', 'FrontDoorFrame': 'Church stone',
    'Altar body': 'Church stone', 'StoneRoof': 'Church plaster vault', 'CopperRoof': 'Church copper roof',
    'Stone floor and courtyard': 'Church floor', 'Windows': 'Church glass',
    'FrontDoorLeft': 'Church door', 'FrontDoorRight': 'Church door', 'SideDoor': 'Church door', 'Organ case': 'Church wood', 'Organ tower': 'Church wood',
    'Seat': 'Church pews', 'Back': 'Church pews',
}
# The walk: the centre aisle (x = 0) from the doors to z = -18, and outside to x = 12, z = -46.
ROUTE = [((-1.0, -50.0), (1.0, -16.0)), ((-1.0, -48.0), (13.0, -44.0))]


def sync_church():
    """Give the runner the repository's Church scene and materials (the runner only syncs code)."""
    for rel in [SCENE, SCENE + '.meta'] + [str(p.relative_to(ROOT)) for p in (ROOT / 'Assets/Materials').glob('Church *.asset*')]:
        target = dj.RUNNER / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)


def unity(name, method, *args):
    WORK.mkdir(parents=True, exist_ok=True)
    sync_church()
    logs, resources = listen.launch(name, ['-executeMethod', method, *args])
    if resources['exitCode'] != 0:
        raise SystemExit(f'{method} failed; see {logs}/unity.log')
    return logs


def inside(box, rect):
    (x0, z0), (x1, z1) = rect
    cx, cz, sx, sz = box['center']['x'], box['center']['z'], box['size']['x'] / 2, box['size']['z'] / 2
    return cx + sx > x0 and cx - sx < x1 and cz + sz > z0 and cz - sz < z1


def plan():
    inv = json.loads((WORK / 'inventory.json').read_text())
    acoustic, visual = inv['acoustic'], inv['visual']
    seats = [a for a in acoustic if a['name'] in ('Seat', 'Back')]
    proxies, report = [], []
    # Pews: every visual pew without acoustic seat/back boxes gets the existing pattern (seat slab + back).
    pews = [v for v in visual if v['name'] == 'Pew' or v['path'].endswith('/Pew')]
    for i, p in enumerate(pews):
        c, s = p['center'], p['size']
        covered = any(abs(a['center']['x'] - c['x']) < s['x'] / 2 and abs(a['center']['z'] - c['z']) < s['z'] / 2 + .3 for a in seats)
        if covered:
            continue
        floor = c['y'] - s['y'] / 2
        long_x = s['x'] >= s['z']
        width, depth = (s['x'], s['z']) if long_x else (s['z'], s['x'])
        seat = dict(name=f'Pew seat {i}', material='Church pews', yaw=0 if long_x else 90,
                    center=dict(x=c['x'], y=floor + .48, z=c['z']), size=dict(x=width, y=.16, z=min(.65, .6 * depth)))
        back = dict(name=f'Pew back {i}', material='Church pews', yaw=seat['yaw'],
                    center=dict(x=c['x'] + (0 if long_x else depth / 2 - .1), y=floor + 1.0, z=c['z'] - (depth / 2 - .1 if long_x else 0)),
                    size=dict(x=width, y=.8, z=.08))
        proxies += [seat, back]
        report.append(f"pew {p['path']} at ({c['x']:.1f}, {c['z']:.1f}), {width:.1f} x {depth:.1f} m")
    # Columns and piers of the combined "Nave arcades" mesh: sections of the mesh at ear height,
    # each followed upwards until it widens into the arches; only when the arcades are not already
    # acoustic geometry (they are in the current scene: a stone mesh collider).
    acoustic_names = {a['name'] for a in acoustic}
    if 'Nave arcades' in acoustic_names:
        report.append('Nave arcades are already acoustic geometry (mesh collider): no pier proxies')
    for i, box in enumerate([] if 'Nave arcades' in acoustic_names else arcade_piers()):
        if any(inside(box, r) for r in ROUTE):
            report.append(f"skipped pier at ({box['center']['x']:.1f}, {box['center']['z']:.1f}) on the walk route")
            continue
        proxies.append(box)
        report.append(f"pier {i}: ({box['center']['x']:.1f}, {box['center']['z']:.1f}), {box['size']['x']:.2f} x {box['size']['z']:.2f} m, "
                      f"{box['center']['y'] - box['size']['y'] / 2:.1f}-{box['center']['y'] + box['size']['y'] / 2:.1f} m high")
    # Columns and piers that are separate visual parts.
    for i, v in enumerate(visual):
        if not any(k in v['path'] for k in ('arcade', 'Arcade', 'Column', 'Pillar', 'Pier')):
            continue
        s, c = v['size'], v['center']
        if s['y'] < 3 or max(s['x'], s['z']) > 2.0 or c['y'] - s['y'] / 2 > 1.5:
            continue
        side = .886 * min(s['x'], s['z'])  # square of the same area as a round shaft
        box = dict(name=f'Pier {i}', material='Church stone', yaw=0, center=dict(x=c['x'], y=c['y'], z=c['z']), size=dict(x=side, y=s['y'], z=side))
        if any(inside(box, r) for r in ROUTE):
            report.append(f"skipped pier {v['path']} on the walk route")
            continue
        proxies.append(box)
        report.append(f"pier {v['path']} at ({c['x']:.1f}, {c['z']:.1f}), {side:.2f} m square, {s['y']:.1f} m tall")
    # Pew blocks: the floor under and between the pews is part of the seating block (a pew block
    # absorbs per floor area it covers, Martellotta & Cirillo, Applied Acoustics 70, 2009), so a thin
    # slab with the pews' material covers each block's footprint instead of bare marble.
    seat_boxes = [a for a in acoustic if a['name'] in ('Seat', 'Back')]
    for side in (-1, 1):
        block = [a for a in seat_boxes if a['center']['x'] * side > 0]
        if not block:
            continue
        x0 = min(a['center']['x'] - a['size']['x'] / 2 for a in block); x1 = max(a['center']['x'] + a['size']['x'] / 2 for a in block)
        z0 = min(a['center']['z'] - a['size']['z'] / 2 for a in block); z1 = max(a['center']['z'] + a['size']['z'] / 2 for a in block)
        proxies.append(dict(name=f'Pew block {"west" if side < 0 else "east"}', material='Church pews', yaw=0,
                            center=dict(x=(x0 + x1) / 2, y=1.01, z=(z0 + z1) / 2), size=dict(x=x1 - x0, y=.02, z=z1 - z0)))
        report.append(f'pew block {side:+d}: x {x0:.1f}..{x1:.1f}, z {z0:.1f}..{z1:.1f} ({(x1 - x0) * (z1 - z0):.0f} m2 of floor)')
    for p in proxies:
        if any(inside(p, r) for r in ROUTE):
            raise SystemExit(f"proxy {p['name']} blocks the walk route")
    # Transmission per octave band 125 Hz - 16 kHz (16 kHz = 8 kHz) from the assemblies' sound
    # reduction (church_transmission.py); occupied pews are unassigned and keep the pews' value.
    def transmission(name, absorption):
        t = ct.transmission(name if name in ct.ASSEMBLY else 'Church pews', absorption)
        return t + [t[-1]]
    # The windows are thin closed slabs: acoustic index 1 makes them partitions (their transmission is
    # the pane's sound reduction, applied once) instead of a refracting glass medium.
    data = dict(materials=[dict(name=n, frequencies=FREQUENCIES, absorption=m['absorption'], scattering=m['scattering'],
                                transmission=transmission(n, m['absorption']), **({'ior': [1.0] * 8} if n == 'Church glass' else {}))
                           for n, m in MATERIALS.items()],
                assignments=[dict(objectName=o, material=m) for o, m in ASSIGNMENTS.items()],
                proxies=[dict(name=p['name'], material=p['material'], yaw=p['yaw'], center=p['center'], size=p['size']) for p in proxies])
    (WORK / 'plan.json').write_text(json.dumps(data, indent=1) + '\n')
    (WORK / 'plan.txt').write_text('\n'.join(report) + '\n')
    print('\n'.join(report)); print(f'{len(proxies)} proxies -> {WORK / "plan.json"}')


def sections(tri, y, cell=.1):
    """Clusters of the mesh's cross-section at height y: list of (xmin, xmax, zmin, zmax)."""
    import numpy as np
    from scipy import ndimage
    d = tri[:, :, 1] - y
    points = []
    for i in range(3):
        a, b, da, db = tri[:, i], tri[:, (i + 1) % 3], d[:, i], d[:, (i + 1) % 3]
        cross = (da <= 0) != (db <= 0)
        t = da[cross] / (da[cross] - db[cross])
        points.append(a[cross] + (b[cross] - a[cross]) * t[:, None])
    p = np.concatenate(points)[:, [0, 2]] if points else np.zeros((0, 2))
    if not len(p):
        return []
    lo = p.min(0) - cell
    idx = np.floor((p - lo) / cell).astype(int)
    grid = np.zeros(idx.max(0) + 2, bool); grid[idx[:, 0], idx[:, 1]] = True
    labels, n = ndimage.label(ndimage.binary_dilation(grid, iterations=2))
    out = []
    for k in range(1, n + 1):
        q = p[labels[idx[:, 0], idx[:, 1]] == k]
        out.append((q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max()))
    return out


def arcade_piers(floor=1.0, ear=2.5, widest=2.5):
    import numpy as np
    path = WORK / 'Nave arcades.tris.f32'
    if not path.exists():
        return []
    tri = np.fromfile(path, '<f4').reshape(-1, 3, 3).astype(float)
    boxes = []
    for x0, x1, z0, z1 in sections(tri, ear):
        if max(x1 - x0, z1 - z0) > widest:
            continue
        top = ear
        for y in np.arange(ear + .25, tri[:, :, 1].max(), .25):
            match = [c for c in sections(tri, y) if c[0] < x1 + .3 and c[1] > x0 - .3 and c[2] < z1 + .3 and c[3] > z0 - .3]
            if not match or any((c[1] - c[0]) > 1.5 * (x1 - x0) + .2 or (c[3] - c[2]) > 1.5 * (z1 - z0) + .2 for c in match):
                break
            top = y
        side_x, side_z = .886 * (x1 - x0), .886 * (z1 - z0)   # a round shaft's equal-area square (exact for squares would be 1)
        boxes.append(dict(name=f'Pier {len(boxes)}', material='Church stone', yaw=0,
                          center=dict(x=(x0 + x1) / 2, y=(floor + top) / 2, z=(z0 + z1) / 2), size=dict(x=side_x, y=top - floor, z=side_z)))
    return boxes


def copy_back():
    changed = [SCENE, SCENE + '.meta'] + [str(p.relative_to(dj.RUNNER)) for p in (dj.RUNNER / 'Assets/Materials').glob('Church *.asset*')]
    for rel in changed:
        shutil.copy2(dj.RUNNER / rel, ROOT / rel)
    print('copied back:', changed)


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'inspect':
        unity('church-inspect', 'ChurchAcoustics.Inspect', '-churchOutput', str(WORK))
    elif command == 'plan':
        plan()
    elif command == 'apply':
        unity('church-apply', 'ChurchAcoustics.Apply', '-churchPlan', str(WORK / 'plan.json'))
        copy_back()
    elif command == 'capture':
        target = WORK / 'speech-walk'
        if target.exists():
            raise SystemExit(f'{target} exists')
        unity('church-capture', 'ChurchAcoustics.CaptureSpeechWalk', '-churchOutput', str(target), '-churchSpeech', str(SPEECH))
    else:
        raise SystemExit(__doc__)
