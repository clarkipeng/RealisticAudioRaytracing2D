"""Top-down sound-energy maps of the Church (energy method, organ source).

    PY=.context/fdtd-teacher/venv/bin/python
    B=docs/benchmarks/energy-audio-2026-11-09
    $PY $B/church_map.py grid 1.0                  # receivers at ear height -> .context/energy-audio/maps/receivers.f32
    nice -n 19 $PY $B/church_map.py run 0 650     # guarded Unity: one solve per receiver at take frames 0 (doors closed), 650 (open)
    nice -n 19 $PY $B/church_map.py run 0 --every 64   # cost probe on every 64th receiver
    $PY $B/church_map.py page                      # PNGs and index.html in .context/energy-audio/maps/

Receivers: a square grid inside the enclosed air of the take (church_room.py voxels) and up to
10 m in front of the doors, 1.65 m above the local floor, at least 0.4 m from any surface. Each
receiver is one measurement solve of the energy method with the listening render's settings
(listen.py, 2 m diffuse receiver sphere). Quantities: direct sound, early (to 80 ms after the
direct sound), late and total energy per octave band, dB re the free-field direct sound at 1 m.
The theory line along the nave is the direct sound 1/r^2 plus the diffuse reverberant level
16*pi/A (the classic 4/A in this convention), A from the room estimate or from a mean absorption of 0.12.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
import church_room as cr  # noqa: E402
import listen  # noqa: E402

OUT = ROOT / '.context/energy-audio/maps'
EAR = 1.65
FRONT = ((-14.0, 14.0), (-48.0, -34.9))      # x, z range in front of the doors (covers the walk's end)
QUANTITIES = ['direct', 'early', 'late', 'total']
PANELS = [3, 0, 1, 2]                          # total, direct, early, late
BANDS = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def voxels(cell=.2, seal=5):
    scene, tri, owner, mat = cr.world_triangles(Path(listen.TAKE))
    names = [o['name'] for o in scene['objects']]
    keep = np.array([names[o] != 'Meadow' for o in owner])
    tri, owner = tri[keep], owner[keep]
    lo = tri.reshape(-1, 3).min(0) - 1
    shape = np.ceil((tri.reshape(-1, 3).max(0) + 1 - lo) / cell).astype(int) + 1
    solid = np.zeros(shape, bool)
    for t in tri:
        e1, e2 = t[1] - t[0], t[2] - t[0]
        n = int(np.ceil(max(np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(t[2] - t[1])) / (cell / 4))) + 1
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        ok = u + v <= 1
        idx = np.floor((t[0] + u[ok, None] * e1 + v[ok, None] * e2 - lo) / cell).astype(int)
        solid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    from scipy import ndimage
    lab, _ = ndimage.label(~ndimage.binary_dilation(solid, iterations=seal))
    probe = tuple(np.floor((np.array([0, 3, -20]) - lo) / cell).astype(int))
    interior = ndimage.binary_dilation(lab == lab[probe], iterations=seal) & ~solid
    return scene, tri, owner, names, solid, interior, lo, cell


def grid(spacing=1.0):
    scene, tri, owner, names, solid, interior, lo, cell = voxels()
    # Floor: the highest upward-facing surface of the building fabric below 2.2 m at each point.
    fabric = {'Stone floor and courtyard', 'StoneBrickWalls', 'SideStairs', 'FrontDoorFrame', 'Chapel portals', 'Baptistery ribs'}
    up = []
    for t, o in zip(tri, owner):
        n = np.cross(t[1] - t[0], t[2] - t[0])
        if names[o] in fabric and n[1] > .9 * np.linalg.norm(n) and t[:, 1].max() < 2.2:
            up.append(t)
    up = np.array(up)
    from scipy import ndimage
    clear = ~ndimage.binary_dilation(solid, iterations=int(math.ceil(.4 / cell)))
    xs = np.arange(-24, 24 + 1e-6, spacing); zs = np.arange(FRONT[1][0], 30 + 1e-6, spacing)
    points = []
    for x in xs:
        for z in zs:
            floor = 1.0
            if len(up):
                # point-in-triangle (xz) for upward triangles
                a, b, c = up[:, 0], up[:, 1], up[:, 2]
                v0, v1 = c - a, b - a
                p = np.array([x, 0, z]) - a
                d00 = v0[:, 0] ** 2 + v0[:, 2] ** 2; d01 = v0[:, 0] * v1[:, 0] + v0[:, 2] * v1[:, 2]; d11 = v1[:, 0] ** 2 + v1[:, 2] ** 2
                d20 = p[:, 0] * v0[:, 0] + p[:, 2] * v0[:, 2]; d21 = p[:, 0] * v1[:, 0] + p[:, 2] * v1[:, 2]
                den = d00 * d11 - d01 * d01
                with np.errstate(divide='ignore', invalid='ignore'):
                    u = (d11 * d20 - d01 * d21) / den; v = (d00 * d21 - d01 * d20) / den
                hit = (u >= 0) & (v >= 0) & (u + v <= 1)
                if hit.any():
                    ys = (a[:, 1] + u * v0[:, 1] + v * v1[:, 1])[hit]
                    floor = float(ys.max())
            y = floor + EAR
            idx = tuple(np.floor((np.array([x, y, z]) - lo) / cell).astype(int))
            if min(idx) < 0 or any(i >= s for i, s in zip(idx, solid.shape)) or not clear[idx]:
                continue
            outside = FRONT[0][0] <= x <= FRONT[0][1] and FRONT[1][0] <= z <= FRONT[1][1]
            if interior[idx] or outside:
                points.append((x, y, z, 1.0 if outside and not interior[idx] else 0.0))
    p = np.array(points, np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    p[:, :3].astype('<f4').tofile(OUT / 'receivers.f32')
    np.save(OUT / 'receivers-outside.npy', p[:, 3])
    (OUT / 'grid.json').write_text(json.dumps(dict(spacing=spacing, receivers=len(p), ear=EAR, take=str(listen.TAKE))) + '\n')
    print(f'{len(p)} receivers at {spacing} m ({int((p[:, 3] > 0).sum())} outside)')


def run(frames, every=1):
    """Guarded Unity measurement; every > 1 measures every n-th receiver into maps/timing (cost probe)."""
    out = OUT if every == 1 else OUT / 'timing'
    out.mkdir(parents=True, exist_ok=True)
    receivers = OUT / 'receivers.f32'
    if every > 1:
        receivers = out / 'receivers.f32'
        np.fromfile(OUT / 'receivers.f32', '<f4').reshape(-1, 3)[::every].tofile(receivers)
    job = dict(take=str(listen.TAKE), output=str(out), settings=listen.METHODS['energy'])
    path = out / 'map.job.json'
    path.write_text(json.dumps(job, indent=1) + '\n')
    sys.path.insert(0, str(ROOT / 'tools'))
    import method_runs as mr
    mr.tests('TakeChecks.MeasureEnergyMap', dict(ACOUSTIC_ENERGY_MAP=str(path), ACOUSTIC_ENERGY_MAP_RECEIVERS=str(receivers),
                                                ACOUSTIC_ENERGY_MAP_FRAMES=','.join(map(str, frames))))


def theory(distance, alpha_mean, room, horizon=2.0):
    """Direct 1/r^2 plus the diffuse level 16 pi / A (energy re the free field at 1 m), the diffuse part
    cut at the solver's horizon like the simulation: times (1 - 10^(-6 horizon / T)), T = 0.161 V / A."""
    V, S = room['volume'], room['surface']
    out = {}
    for b, f in enumerate([125, 250, 500, 1000, 2000, 4000, 8000]):
        alpha = room['bands'][b]['meanAbsorption'] if alpha_mean is None else alpha_mean
        A = -S * math.log(1 - alpha) + 4 * cr.AIR_M[b] * V
        out[f] = 1 / np.maximum(distance, .1) ** 2 + 16 * math.pi / A * (1 - 10 ** (-6 * horizon / (.161 * V / A)))
    return out


def page():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xyz = np.fromfile(OUT / 'receivers.f32', '<f4').reshape(-1, 3)
    grid_meta = json.loads((OUT / 'grid.json').read_text())
    h = grid_meta['spacing']
    meta = json.loads((listen.TAKE / 'take.json').read_text())
    scene = meta['scene']
    states = np.fromfile(listen.TAKE / 'states.bin', dtype=np.dtype([('sample', '<i8'), ('state', '<f4', (len(scene['objects']), 20))]))
    source = states['state'][0, scene['sources'][0]['objectIndex'], :3]
    route = states['state'][:, scene['listeners'][0], :3]
    room = json.loads((ROOT / '.context/energy-audio/church/room-after.json').read_text())
    g = np.frombuffer((listen.TAKE / 'geometry.bin').read_bytes(), dtype=[('v', '<f4', (3, 3)), ('o', '<i4'), ('m', '<i4')])
    tri, owner = g['v'].astype(float), g['o']          # object space; posed per frame below
    names = [o['name'] for o in scene['objects']]
    maps = sorted(OUT.glob('map-*.f32'))
    pages = []
    for path in maps:
        frame = int(path.stem.split('-')[1])
        v = np.fromfile(path, '<f4').reshape(-1, 4, 8).astype(float)
        mm = json.loads(path.with_suffix('.json').read_text())
        pose = states['state'][min(frame, len(states) - 1)]
        world = np.empty_like(tri)
        for o in np.unique(owner):
            m = owner == o
            world[m] = pose[o, :3] + cr.rotate(pose[o, 3:7], tri[m].reshape(-1, 3)).reshape(-1, 3, 3)
        walls = cut_segments(world, owner, names, 2.65)
        xs = np.unique(xyz[:, 0]).astype(float); zs = np.unique(xyz[:, 2]).astype(float)
        ix = np.round((xyz[:, 0] - xs.min()) / h).astype(int); iz = np.round((xyz[:, 2] - zs.min()) / h).astype(int)
        nx, nz = ix.max() + 1, iz.max() + 1
        extent = (xs.min() - h / 2, xs.min() + (nx - .5) * h, zs.min() - h / 2, zs.min() + (nz - .5) * h)
        broadband = v[:, :, :6].mean(2)         # 125 Hz - 4 kHz, mean so that 0 dB is still the free field at 1 m
        selections = [('broadband 125 Hz-4 kHz', broadband)] + [(f'{f} Hz', v[:, :, b]) for b, f in ((1, 250), (3, 1000), (5, 4000))]
        doors = 'doors open' if frame > 0 else 'doors closed'
        files, boxes = [], {}
        for label, data in selections:
            fig, axes = plt.subplots(1, 4, figsize=(19, 7.2), sharey=True)
            top = math.ceil(np.percentile(10 * np.log10(np.maximum(data[:, 3], 1e-12)), 99.5) / 5) * 5
            for q, ax in zip(PANELS, axes):
                img = np.full((nz, nx), np.nan)
                img[iz, ix] = 10 * np.log10(np.maximum(data[:, q], 1e-12))
                im = ax.imshow(img, origin='lower', extent=extent, cmap='magma', vmin=top - 40, vmax=top, interpolation='nearest')
                for s in walls:
                    ax.plot(s[:, 0], s[:, 1], color='#4FC3F7', lw=.5)
                ax.plot(route[:, 0], route[:, 2], color='white', lw=1, alpha=.8)
                ax.plot(source[0], source[2], '*', color='#00E676', ms=12)
                ax.set_title(QUANTITIES[q] + (' (to 80 ms after the direct sound)' if q == 1 else ' (after 80 ms)' if q == 2 else ''), fontsize=10)
                ax.set_aspect('equal'); ax.set_xlabel('x (m)')
                fig.colorbar(im, ax=ax, shrink=.6, label='dB re free field at 1 m')
            axes[0].set_ylabel('z (m)')
            fig.suptitle(f'Energy method, organ source, {label}, {doors}; receivers every {h:g} m at 1.65 m above the floor; white: the walk; green star: organ', fontsize=10)
            fig.tight_layout()
            fig.canvas.draw()          # axes boxes after the equal aspect, as fractions of the image (top-left origin)
            boxes[label] = [[ax.get_position().x0, 1 - ax.get_position().y1, ax.get_position().x1, 1 - ax.get_position().y0] for ax in axes]
            name = f"map-{frame}-{label.split()[0].replace('-', '')}.png"
            fig.savefig(OUT / name, dpi=90); plt.close(fig)
            files.append((label, name))
        # Nave centreline against theory
        line = np.abs(xyz[:, 0]) < h / 2 + 1e-3
        order = np.argsort(xyz[line, 2])
        z = xyz[line, 2][order]; d = np.linalg.norm(xyz[line][order] - source, axis=1)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        for ax, (b, f) in zip(axes, ((1, 250), (3, 1000), (5, 4000))):
            sim = v[line][order]
            ax.plot(z, 10 * np.log10(np.maximum(sim[:, 3, b], 1e-12)), '-', color='#009E73', label='simulated total')
            ax.plot(z, 10 * np.log10(np.maximum(sim[:, 2, b], 1e-12)), '--', color='#009E73', lw=.8, label='simulated late (>80 ms)')
            ax.plot(z, 10 * np.log10(np.maximum(sim[:, 0, b], 1e-12)), ':', color='#009E73', lw=.8, label='simulated direct')
            ax.plot(z, 10 * np.log10(theory(d, None, room, mm['horizon'])[f]), color='#000000', lw=1, label=f'1/r² + 16π/A, this model (ᾱ {room["bands"][b]["meanAbsorption"]:.3f})')
            ax.plot(z, 10 * np.log10(theory(d, .12, room, mm['horizon'])[f]), color='#D55E00', lw=1, label='1/r² + 16π/A, real church (ᾱ 0.12)')
            ax.plot(z, 10 * np.log10(1 / np.maximum(d, .1) ** 2), color='#999999', lw=.8, label='direct 1/r²')
            ax.axvline(-35.2, color='#999999', lw=.6, ls='--'); ax.text(-35, ax.get_ylim()[0], ' front doors', fontsize=7, va='bottom')
            ax.set_title(f'nave centreline (x = 0), {f} Hz, {doors}', fontsize=9); ax.set_xlabel('z (m); organ at z = 18'); ax.grid(alpha=.3)
        axes[0].set_ylabel('dB re free field at 1 m'); axes[0].legend(fontsize=7)
        fig.tight_layout(); name = f'centreline-{frame}.png'; fig.savefig(OUT / name, dpi=100); plt.close(fig)
        files.append(('centreline vs theory', name))
        data = dict(x=xyz[:, 0].round(2).tolist(), z=xyz[:, 2].round(2).tolist(),
                    db={f'{QUANTITIES[q]}-{label}': (10 * np.log10(np.maximum(sel[:, q], 1e-12))).round(1).tolist() for label, sel in selections for q in range(4)})
        (OUT / f'data-{frame}.js').write_text(f'window.MAP{frame} = ' + json.dumps(dict(data, extent=[float(e) for e in extent], spacing=h, boxes=boxes)) + ';\n')
        pages.append((frame, doors, files, mm))
    html = ['<!doctype html><meta charset="utf-8"><title>Church energy maps</title>',
            '<style>body{font:14px/1.45 -apple-system,sans-serif;margin:1.5em;color:#222}img{max-width:100%;display:block;margin:.4em 0}'
            '#read{position:fixed;right:1em;top:1em;background:#fff;border:1px solid #999;padding:.4em .7em}</style>',
            '<h1>Church: sound energy over the floor plan (energy method, organ source)</h1>',
            '<p>Each pixel is one receiver 1.65 m above the floor: total, direct, early (to 80 ms after the direct sound) and late energy, '
            'dB re the free-field direct sound at 1 m. Hover a map for its value. The line plots compare the nave centreline with '
            'direct 1/r² plus the diffuse level of this model and of a real church (mean absorption 0.12).</p><div id="read">hover a map</div>']
    if (OUT / 'notes.html').exists():          # optional hand-written findings for this run
        html.append((OUT / 'notes.html').read_text())
    for frame, doors, files, mm in pages:
        html.append(f'<h2>{doors} (take frame {frame}): {mm["receivers"]} receivers, {mm["rays"]} rays, {mm["wallSeconds"]:.0f} s</h2><script src="data-{frame}.js"></script>')
        for label, name in files:
            html.append(f'<h3>{label}</h3><img src="{name}" data-frame="{frame}" data-label="{label}">')
    html.append("""<script>
// Hover: find the panel under the cursor, convert to plan coordinates, show the nearest receiver's value.
for (const img of document.querySelectorAll('img[data-frame]')) {
  const m = window['MAP' + img.dataset.frame], label = img.dataset.label;
  if (!m || !m.boxes[label]) continue;
  img.addEventListener('mousemove', e => {
    const r = img.getBoundingClientRect(), fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    const q = m.boxes[label].findIndex(b => fx >= b[0] && fx <= b[2] && fy >= b[1] && fy <= b[3]);
    const out = document.getElementById('read');
    if (q < 0) { out.textContent = 'hover a map'; return; }
    const b = m.boxes[label][q], [x0, x1, z0, z1] = m.extent;
    const x = x0 + (fx - b[0]) / (b[2] - b[0]) * (x1 - x0), z = z1 - (fy - b[1]) / (b[3] - b[1]) * (z1 - z0);
    let best = -1, d = m.spacing * m.spacing / 2;
    for (let i = 0; i < m.x.length; i++) { const dd = (m.x[i] - x) ** 2 + (m.z[i] - z) ** 2; if (dd <= d) { d = dd; best = i; } }
    const name = ['total', 'direct', 'early', 'late'][q];
    out.textContent = best < 0 ? `x ${x.toFixed(1)}, z ${z.toFixed(1)}: no receiver`
      : `${name}, ${label}, x ${m.x[best]}, z ${m.z[best]}: ${m.db[name + '-' + label][best].toFixed(1)} dB`;
  });
}
</script>""")
    (OUT / 'index.html').write_text('\n'.join(html))
    print(OUT / 'index.html')


def cut_segments(world, owner, names, height):
    keep = np.array([names[o] not in ('Meadow', 'CopperRoof', 'StoneRoof') for o in owner])
    w = world[keep]
    d = w[:, :, 1] - height
    segs = []
    for t, dd in zip(w, d):
        pts = [t[i] + (t[(i + 1) % 3] - t[i]) * (dd[i] / (dd[i] - dd[(i + 1) % 3])) for i in range(3) if (dd[i] <= 0) != (dd[(i + 1) % 3] <= 0)]
        if len(pts) == 2:
            segs.append(np.array([[pts[0][0], pts[0][2]], [pts[1][0], pts[1][2]]]))
    return segs


if __name__ == '__main__':
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ('', [])
    if cmd == 'grid':
        grid(float(args[0]) if args else 1.0)
    elif cmd == 'run':
        every = int(args.pop(args.index('--every') + 1)) if '--every' in args else 1
        run([int(f) for f in args if f != '--every'] or [0], every)
    elif cmd == 'page':
        page()
    else:
        raise SystemExit(__doc__)
