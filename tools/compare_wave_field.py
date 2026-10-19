#!/usr/bin/env python3
"""Independent rigid-wall pressure reference; never opens a window or plays audio.

Requires numpy, scipy, matplotlib and either bempp-cl==0.4.2 (finite 3D)
or ngsolve==6.2.2608 (--planar point-source/extruded geometry). Run with
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2.
The wave solvers use exp(-i*w*t); exported fields are conjugated to our renderer's
exp(+i*w*t) convention. Source pressure is exp(-i*k*r)/r, unit amplitude at 1 m.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(output):
    """The staggered finite 3D walls from the earlier sampling experiment."""
    if (output/'scene.json').exists():
        raise FileExistsError('Scene already exists; use another output directory.')
    walls = [dict(center=[-.8, -.9, 0], half=[.3, 1.1, 1.4]),
             dict(center=[.8, .9, 0], half=[.3, 1.1, 1.7])]
    x, y = np.linspace(-2.4, 3.4, 39), np.linspace(-2.6, 2.6, 35)
    xx, yy = np.meshgrid(x, y)
    points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, .2)]
    keep = np.ones(len(points), dtype=bool)
    for wall in walls:
        keep &= np.any(abs(points - wall['center']) > np.array(wall['half']) + .15, axis=1)
    config = dict(source=[-3, -.4, .2], walls=walls, points=points[keep].ravel().tolist(),
                  gridIndices=np.flatnonzero(keep).tolist(), x=x.tolist(), y=y.tolist(),
                  frequencies=[125, 250, 500], sampleRate=1000, fftSize=8,
                  speedOfSound=343, rays=[8, 128], seeds=[17, 65554, 131091, 196628],
                  maxBounces=4, maximumDelaySeconds=.1,
                  meshSizes=[.6, .4, .28], referenceConvention='exp(+i*w*t)')
    save_json(output / 'scene.json', config)


def box_grid(walls, spacing):
    import bempp_cl.api as bem
    vertices, faces, lookup = [], [], {}
    def vertex(p):
        key = tuple(np.round(p, 12))
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append(p.copy())
        return lookup[key]
    for wall in walls:
        c, half = np.array(wall['center'], dtype=float), np.array(wall['half'], dtype=float)
        for axis in range(3):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            for sign in [-1, 1]:
                nu, nv = np.ceil(2 * half[[u, v]] / spacing).astype(int)
                ids = np.empty((nu + 1, nv + 1), dtype=int)
                for i in range(nu + 1):
                    for j in range(nv + 1):
                        p = c.copy(); p[axis] += sign * half[axis]
                        p[u] += half[u] * (2 * i / nu - 1)
                        p[v] += half[v] * (2 * j / nv - 1)
                        ids[i, j] = vertex(p)
                for i in range(nu):
                    for j in range(nv):
                        a, b, d, e = ids[i, j], ids[i+1, j], ids[i, j+1], ids[i+1, j+1]
                        faces.extend([(a, b, e), (a, e, d)] if sign > 0 else [(a, e, b), (a, d, e)])
    return bem.Grid(np.array(vertices).T, np.array(faces).T)


def incident(points, source, k):
    r = np.linalg.norm(points - source, axis=1)
    return np.exp(1j * k * r) / r


def solve(grid, source, points, k):
    """Burton–Miller: (.5 I - K + alpha W) p = p_inc + alpha dn(p_inc).

    W=-dn(D), alpha=i/k. Total Neumann trace is zero (rigid closed walls).
    Exterior field = incident + D p. No ray paths or wedge coefficients enter.
    Formulation: Smigaj et al., Solving Boundary Integral Problems with BEM++, eq.43.
    """
    import bempp_cl.api as bem
    from bempp_cl.api.operators.boundary import helmholtz, sparse
    from bempp_cl.api.operators.potential import helmholtz as potential
    started = time.perf_counter()
    space = bem.function_space(grid, 'P', 1)
    alpha = 1j / k
    source = np.asarray(source, dtype=float)
    @bem.complex_callable
    def rhs_fun(x, normal, domain, result):
        d = x - source
        r = np.sqrt(np.sum(d*d))
        p = np.exp(1j*k*r)/r
        result[0] = p + alpha*p*(1j*k - 1/r)*np.sum(d*normal)/r
    rhs = bem.GridFunction(space, fun=rhs_fun)
    kw = dict(assembler='dense', device_interface='numba')
    identity = sparse.identity(space, space, space)
    double = helmholtz.double_layer(space, space, space, k, **kw)
    hyper = helmholtz.hypersingular(space, space, space, k, **kw)
    lhs = .5*identity - double + alpha*hyper
    pressure, info, residuals = bem.linalg.gmres(lhs, rhs, tol=1e-8, maxiter=500,
                                                use_strong_form=True, return_residuals=True)
    if info != 0:
        raise RuntimeError(f'BEM solve failed: GMRES {info}')
    field = incident(points, source, k)
    # Bound temporary potential assembly memory independently of map size.
    for first in range(0, len(points), 128):
        part = points[first:first+128]
        op = potential.double_layer(space, part.T, k, device_interface='numba')
        field[first:first+len(part)] += np.asarray(op * pressure).ravel()
    if not np.isfinite(field).all():
        raise ValueError('Non-finite reference field')
    return field.conj(), dict(triangles=grid.number_of_elements, dofs=space.global_dof_count,
                             seconds=time.perf_counter()-started, iterations=len(residuals),
                             finalResidual=float(residuals[-1]))


def sphere_exact(points, source, k, radius=1, terms=40):
    """Independent spherical-harmonic solution for a monopole outside a rigid sphere."""
    r, rs = np.linalg.norm(points, axis=1), np.linalg.norm(source)
    cosine = points @ source / (r * rs)
    def hankel(n, x, derivative=False):
        return spherical_jn(n, x, derivative) + 1j*spherical_yn(n, x, derivative)
    scattered = np.zeros(len(points), dtype=complex)
    for n in range(terms):
        coefficient = -1j*k*(2*n+1)*hankel(n, k*rs)*spherical_jn(n, k*radius, True)/hankel(n, k*radius, True)
        scattered += coefficient*hankel(n, k*r)*eval_legendre(n, cosine)
    return (incident(points, source, k)+scattered).conj()


def reference(output, validate=False, only_mesh=None, only_hz=None):
    config = json.loads((output / 'scene.json').read_text())
    if config.get('physicalModel') == 'point-extruded':
        from planar_wave import reference as planar_reference
        return planar_reference(output, validate, only_mesh, only_hz)
    import bempp_cl.api as bem
    points, source = np.array(config['points']).reshape(-1, 3), np.array(config['source'])
    if validate:
        angle = np.linspace(0, 2*np.pi, 100, endpoint=False)
        probes = 1.7*np.c_[np.cos(angle), np.sin(angle), np.zeros(len(angle))]
        checks = []
        for hz in config['frequencies']:
            k = 2*np.pi*hz/config['speedOfSound']
            exact = sphere_exact(probes, source, k)
            longer = sphere_exact(probes, source, k, terms=60)
            series_change = float(np.linalg.norm(exact-longer)/np.linalg.norm(longer))
            if series_change > 1e-12:
                raise ValueError('Analytic sphere expansion needs more terms at this frequency.')
            for level in [2, 3, 4]:
                actual, stats = solve(bem.shapes.regular_sphere(level), source, probes, k)
                stats.update(hz=hz, refinement=level, relativeL2=float(np.linalg.norm(actual-exact)/np.linalg.norm(exact)),
                             maximumAbsoluteError=float(abs(actual-exact).max()), seriesRelativeChange=series_change)
                checks.append(stats); print('sphere', stats, flush=True)
                save_json(output / 'validation.json', checks)
        return
    fingerprint = hashlib.sha256((output/'scene.json').read_bytes()).hexdigest()
    for h in config['meshSizes'] if only_mesh is None else [only_mesh]:
        grid = box_grid(config['walls'], h)
        for hz in config['frequencies'] if only_hz is None else [only_hz]:
            stem = output / f'reference-{h:g}-{hz:g}'
            if Path(str(stem)+'.npz').exists():
                if json.loads(Path(str(stem)+'.json').read_text())['sceneSha256'] != fingerprint:
                    raise ValueError('Cached reference belongs to another scene.')
                continue
            field, stats = solve(grid, source, points, 2*np.pi*hz/config['speedOfSound'])
            np.savez_compressed(str(stem)+'.npz', pressure=field, points=points)
            stats.update(hz=hz, meshSize=h, sceneSha256=fingerprint, bemppVersion=bem.__version__)
            save_json(Path(str(stem)+'.json'), stats); print('walls', stats, flush=True)


def renderer(output, runner, unity, room_control=False):
    root = Path(__file__).resolve().parents[1]
    runner = runner.resolve(); output = output.resolve()
    if runner == root or (runner/'Temp/UnityLockfile').exists():
        raise ValueError('Use a closed, isolated Unity project; never the working editor.')
    hashes = {}
    for folder in ['Assets/Scripts', 'Assets/Shaders', 'Assets/Resources/Acoustics',
                   'Assets/Tests/Editor/TakeChecks.cs', 'Assets/Tests/Editor/RayTransportChecks.cs']:
        parent = root/folder
        for source in [parent] if parent.is_file() else parent.rglob('*'):
            if not source.is_file() or source.suffix == '.meta':
                continue
            relative = source.relative_to(root)
            if source.read_bytes() != (runner/relative).read_bytes():
                raise ValueError(f'Runner runtime differs: {relative}')
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = runner/'Assets/Tests/Editor/WaveFieldMeasurement.cs'
    if destination.exists() or any(output.glob('*.csv')):
        raise ValueError('Preserve existing measurements; use a fresh output directory.')
    measurement = 'MeasureRoomMaterialMotion' if room_control else 'MeasureWaveField'
    save_json(output/'renderer-provenance.json', dict(files=hashes,
        fixtureSha256=hashlib.sha256((root/'Assets/Tests/Editor/WaveFieldMeasurement.cs').read_bytes()).hexdigest(),
        measurement=measurement,
        sceneSha256=None if room_control else hashlib.sha256((output/'scene.json').read_bytes()).hexdigest()))
    shutil.copy2(root/'Assets/Tests/Editor/WaveFieldMeasurement.cs', destination)
    try:
        subprocess.run(['nice', '-n', '15', str(unity), '-batchmode', '-force-metal', '-disable-audio',
                        '-job-worker-count', '2', '-projectPath', str(runner), '-runTests',
                        '-testPlatform', 'EditMode', '-testFilter', 'TakeChecks.'+measurement,
                        '-testResults', str(output/'renderer-tests.xml'), '-logFile', str(output/'renderer.log')],
                       env=dict(os.environ, ACOUSTIC_WAVE_OUTPUT=str(output)), check=True)
        import xml.etree.ElementTree as xml
        result = xml.parse(output/'renderer-tests.xml').getroot()
        if int(result.get('passed', '0')) != 1 or result.get('result') != 'Passed':
            raise RuntimeError('Pressure measurement did not complete successfully.')
    finally:
        destination.unlink(missing_ok=True)
        Path(str(destination)+'.meta').unlink(missing_ok=True)


def shadow_mask(config, points):
    # Independent analytic segment/box occlusion check, not the shader's classification.
    shadow = np.zeros(len(points), dtype=bool)
    direction = points-np.array(config['source'])
    for wall in config['walls']:
        enter, leave, hit = np.zeros(len(points)), np.ones(len(points)), np.ones(len(points), dtype=bool)
        for axis in range(3):
            low, high = np.array(wall['center'])[axis]+np.array([-1, 1])*wall['half'][axis]
            s, d = config['source'][axis], direction[:, axis]
            hit &= (d!=0) | ((s>=low)&(s<=high))
            a = np.divide(low-s, d, out=np.full(len(points), -np.inf), where=d!=0)
            b = np.divide(high-s, d, out=np.full(len(points), np.inf), where=d!=0)
            enter, leave = np.maximum(enter, np.minimum(a, b)), np.minimum(leave, np.maximum(a, b))
        shadow |= hit & (enter<leave)
    return shadow


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    config = json.loads((output/'scene.json').read_text())
    rows = np.genfromtxt(output/'renderer.csv', delimiter=',', names=True)
    points = np.array(config['points']).reshape(-1, 3)
    shadow = shadow_mask(config, points)
    shape = len(config['y']), len(config['x'])
    indices = np.array(config['gridIndices'])
    reference_runs = [json.loads(p.read_text()) for p in output.glob('reference-*.json')]
    fingerprint = hashlib.sha256((output/'scene.json').read_bytes()).hexdigest()
    provenance = json.loads((output/'renderer-provenance.json').read_text())
    if provenance['sceneSha256'] != fingerprint or any(r['sceneSha256'] != fingerprint for r in reference_runs):
        raise ValueError('Cannot compare fields from different scene configurations.')
    planar = config.get('physicalModel') == 'point-extruded'
    validation = json.loads((output/'validation.json').read_text())
    if planar:
        from planar_wave import fingerprint as solver_fingerprint, control_fingerprint
        if (not validation['passed'] or validation['controlSha256'] != control_fingerprint(config)
                or any(r['solverSha256'] != solver_fingerprint(output) for r in reference_runs)):
            raise ValueError('Planar controls/reference use different settings or solver code.')
    report = dict(scene=config, metricVersion=2, analyticValidation=validation,
                  referenceRuns=reference_runs, rendererProvenance=provenance, comparisons=[], ensembles=[], convergence=[],
                  timing='Warm transport submission, synchronous readback and CPU bookkeeping; no DSP. Not GPU-only time.',
                  pressure='Unit free-field amplitude at 1 m; exp(+i*w*t); no HRTF, output limiter or gain matching.',
                  scope=('Stationary infinitely extruded rigid walls; 3D point source.' if planar else
                         'Stationary finite rigid 3D walls in free space.') + ' Numerical reference, not measured real-world truth.')
    if (output/'audit.json').exists():
        report['numericalAudit'] = json.loads((output/'audit.json').read_text())
        if planar and report['numericalAudit']['solverSha256'] != solver_fingerprint(output):
            raise ValueError('Numerical audit uses a different scene or solver.')
    def field(values):
        result = np.full(np.prod(shape), np.nan)
        result[indices] = values
        return result.reshape(shape)
    def db(values):
        return 20*np.log10(np.maximum(abs(values), 1e-12))
    grid_to_point = {index: i for i, index in enumerate(indices)}
    ix = int(np.argmin(abs(np.array(config['x']))))
    hi, lo = [int(np.argmin(abs(np.array(config['y'])-y))) for y in [.6, -.6]]
    nx = shape[1]
    route_grid = ([hi*nx+x for x in range(ix+1)] + [y*nx+ix for y in range(hi-1, lo-1, -1)] +
                  [lo*nx+x for x in range(ix+1, nx)])
    route = np.array(config.get('routeIndices', [grid_to_point[i] for i in route_grid if i in grid_to_point]), dtype=int)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points[route], axis=0), axis=1))]
    for hz in config['frequencies']:
        meshes = sorted((r['meshSize'] for r in reference_runs if r['hz'] == hz), reverse=True)
        refs = [np.load(output/f'reference-{h:g}-{hz:g}.npz')['pressure'] for h in meshes]
        reference_field = refs[-1]
        for coarse, fine, a, b in zip(meshes, meshes[1:], refs, refs[1:]):
            report['convergence'].append(dict(hz=hz, coarse=coarse, fine=fine,
                relativeL2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                shadowRelativeL2=float(np.linalg.norm((a-b)[shadow])/np.linalg.norm(b[shadow])) if shadow.any() else None,
                maxAbsoluteChange=float(abs(a-b).max())))
        change = report['convergence'][-1]
        display = [('Wave reference', reference_field, None)]
        for diffraction, rays in [(0, max(config['rays']))]+[(1, r) for r in config['rays']]:
            group = rows[(rows['hz']==hz)&(rows['diffraction']==diffraction)&(rows['rays']==rays)]
            samples = []
            for seed in config['seeds']:
                run = np.sort(group[group['seed']==seed], order='point')
                if not np.array_equal(run['point'], np.arange(len(points))):
                    raise ValueError('Incomplete or duplicate renderer measurements.')
                if not np.array_equal(run['direct_visible']==0, shadow):
                    raise ValueError('Shader occlusion disagrees with analytic segment/box intersection.')
                pressure = run['real']+1j*run['imag']; samples.append(pressure)
                for region, mask in [('all', np.ones(len(points), dtype=bool)), ('lit', ~shadow), ('shadow', shadow)]:
                    if not mask.any():
                        continue
                    ref, value = reference_field[mask], pressure[mask]
                    report['comparisons'].append(dict(hz=hz, diffraction=diffraction, rays=rays, seed=seed,
                        region=region, points=int(mask.sum()), complexRelativeL2=float(np.linalg.norm(value-ref)/np.linalg.norm(ref)),
                        amplitudeRelativeL2=float(np.linalg.norm(abs(value)-abs(ref))/np.linalg.norm(ref)),
                        maximumPressure=float(abs(value).max()), zeroPoints=int(np.count_nonzero(value==0)),
                        medianSolveMs=float(np.median(run['solve_seconds'])*1000)))
            ensemble = np.array(samples)
            for region, mask in [('all', np.ones(len(points), dtype=bool)), ('lit', ~shadow), ('shadow', shadow)]:
                if not mask.any():
                    continue
                values, ref = ensemble[:, mask], reference_field[mask]
                mean = values.mean(axis=0); norm = np.linalg.norm(ref)
                report['ensembles'].append(dict(hz=hz, diffraction=diffraction, rays=rays, region=region,
                    meanComplexRelativeL2=float(np.linalg.norm(mean-ref)/norm),
                    seedRmsRelativeL2=float(np.sqrt(np.mean(np.sum(abs(values-mean)**2, axis=1)))/norm)))
            title = f'Diffraction: {rays} rays/bin' if diffraction else 'Diffraction disabled'
            display.append((title, samples[0], np.array(samples)))
        fig = plt.figure(figsize=(16, 10), layout='constrained')
        grid = fig.add_gridspec(3 if len(route) else 2, len(display),
                                 height_ratios=[1, 1, .9] if len(route) else [1, 1])
        map_axes, images = [[], []], [None, None]
        for column, (title, value, _) in enumerate(display):
            for row in range(2):
                ax = fig.add_subplot(grid[row, column])
                map_axes[row].append(ax)
                # Same source normalization and color scale across every method.
                if row == 0:
                    pixels, limits, cmap = db(value), (-45, 6), 'magma'
                elif column == 0:
                    pixels = db(refs[-2])-db(value)
                    limits, cmap = (-15, 15), 'RdBu_r'
                else:
                    pixels = db(value)-db(reference_field)
                    limits, cmap = (-15, 15), 'RdBu_r'
                im = ax.pcolormesh(config['x'], config['y'], field(pixels), shading='nearest',
                                   cmap=cmap, vmin=limits[0], vmax=limits[1], rasterized=True)
                for wall in config['walls']:
                    c, h = np.array(wall['center']), np.array(wall['half'])
                    ax.add_patch(Rectangle(c[:2]-h[:2], *(2*h[:2]), facecolor='#626c7a', edgecolor='white'))
                ax.plot(*points[route, :2].T, '--', color='#35c6bd', lw=1, alpha=.8)
                ax.scatter(*config['source'][:2], marker='*', s=75, color='#35c6bd', zorder=5)
                ax.set(xlabel='x (m)', ylabel='y (m)', aspect='equal', xlim=(-3.3, 3.5), ylim=(-2.7, 2.7),
                       title=title if row==0 else ('Reference refinement check' if column==0 else 'Level error'))
                images[row] = im
        for row, label in enumerate(['Pressure (dB re free-field amplitude at 1 m)', 'Level difference from final reference (dB)']):
            fig.colorbar(images[row], ax=map_axes[row], label=label, shrink=.8, extend='both')
        if len(route):
            ax = fig.add_subplot(grid[2, :])
            for title, value, samples in display:
                line, = ax.plot(distance, db(value[route]), label=title, lw=1.5)
                if samples is not None:
                    levels = db(samples[:, route])
                    ax.fill_between(distance, levels.min(0), levels.max(0), color=line.get_color(), alpha=.12)
            ax.set(xlabel='Distance along dashed route (m)', ylabel='Pressure level (dB)', ylim=(-60, 12))
            ax.grid(alpha=.2); ax.legend(loc='lower left', ncol=2)
        shadow_change = f'{change["shadowRelativeL2"]:.2%}' if shadow.any() else 'n/a'
        title = f'{config.get("family", "walls")} · infinitely extruded walls, point source' if planar else 'two finite rigid 3D walls · slice at z=0.2 m'
        fig.suptitle(f'{hz:g} Hz · {title}\n'
                     f'Maps: seed {config["seeds"][0]}; {len(config["seeds"])} seeds measured. '
                     'Color limits clip extremes; raw values retained.\n'
                     f'Last reference mesh change: {change["relativeL2"]:.2%} overall, '
                     f'{shadow_change} in shadow (not an error bound).', fontsize=12)
        fig.savefig(output/f'pressure-{hz:g}.png', dpi=150); plt.close(fig)
    report['dataHashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [output/'scene.json', output/'renderer.csv']+list(output.glob('reference-*.npz'))}
    save_json(output/'report.json', report)


def prepare_suite(output, profile):
    """Generate data-only fixtures; never launch a solver or Unity implicitly."""
    from planar_wave import prepare as prepare_planar
    if (output/'suite.json').exists():
        raise FileExistsError('Suite already exists; preserve its geometry/split manifest.')
    families = dict(barrier='development', staggered='development', slit='development',
                    corridor='development', double_slit='validation', offset_doors='validation',
                    triple='test', pillars='test')
    cases = []
    def box(x, y, hx, hy):
        return dict(center=[x, y, 0], half=[hx, hy, 1])
    def screen(x, intervals):
        return [box(x, (a+b)/2, .15, (b-a)/2) for a, b in intervals]
    for family, split in [('free', 'control')]+list(families.items()):
        for variant in [1] if profile == 'pilot' or family == 'free' else range(3):
            path = output/f'{family}-{variant}'; path.mkdir()
            prepare_planar(path, family if family in ['free', 'barrier', 'staggered'] else 'free')
            config = json.loads((path/'scene.json').read_text())
            walls = config['walls']
            if family == 'barrier':
                walls = [box(-.8, -.9, [.12, .3, .6][variant], 1.1)]
            elif family == 'staggered':
                distance = [.6, .8, 1.1][variant]
                walls = [box(-distance, -.9, .3, 1.1), box(distance, .9, .3, 1.1)]
            elif family == 'slit':
                gap = [.35, .7, 1.4][variant]
                walls = screen(0, [(-2.5, -gap/2), (gap/2, 2.5)])
            elif family == 'corridor':
                width = [.55, .85, 1.25][variant]
                walls = [box(.2, y*(width+.15), 2.1, .15) for y in [-1, 1]]
            elif family == 'double_slit':
                gap = [.25, .45, .7][variant]
                walls = screen(0, [(-2.5, -.9-gap/2), (-.9+gap/2, .9-gap/2), (.9+gap/2, 2.5)])
            elif family == 'offset_doors':
                gap = [.35, .7, 1.1][variant]
                walls = [w for x, y in [(-.8, -.8), (.8, .8)]
                         for w in screen(x, [(-2.5, y-gap/2), (y+gap/2, 2.5)])]
            elif family == 'triple':
                shift = [.75, 1, 1.25][variant]
                walls = [box(x, y*shift, .18, 1.1) for x, y in [(-1.3, -.9), (0, .9), (1.3, -.9)]]
            elif family == 'pillars':
                half = [.15, .3, .45][variant]
                walls = [box(x, y, half, half) for x in [-.7, .8, 2.3] for y in [-1.5, 0, 1.5]]
            config.update(family=family, walls=walls)
            xx, yy = np.meshgrid(config['x'], config['y'])
            points = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
            keep = np.ones(len(points), dtype=bool)
            for wall in walls:
                keep &= np.any(abs(points[:, :2]-wall['center'][:2]) > np.array(wall['half'][:2])+.15, axis=1)
            config.update(points=points[keep].ravel().tolist(), gridIndices=np.flatnonzero(keep).tolist())
            if family not in ['free', 'barrier', 'staggered']:
                config['routeIndices'] = []  # These are static maps; do not invent a walk through walls.
            if profile == 'study':
                config.update(frequencies=[125, 250, 500], rays=[8, 32, 128, 512],
                              seeds=[17+65537*i for i in range(8)])
            save_json(path/'scene.json', config)
            cases.append(dict(path=path.name, family=family, variant=variant, split=split,
                              sceneSha256=hashlib.sha256((path/'scene.json').read_bytes()).hexdigest()))
    save_json(output/'suite.json', dict(version=1, profile=profile, cases=cases,
              note='Static rigid point-extruded benchmark. Generate first; run explicitly. Test families stay held out.'))


def score_suite(output, split):
    """Equal family/geometry/frequency/seed/region weights; incomplete suites fail."""
    manifest = json.loads((output/'suite.json').read_text()); groups = {}; runtime = None; warnings = []
    cases = [c for c in manifest['cases'] if c['split'] == split]
    if not cases:
        raise ValueError('No cases in the requested split.')
    for case in cases:
        report = json.loads((output/case['path']/'report.json').read_text())
        if report['dataHashes']['scene.json'] != case['sceneSha256']:
            raise ValueError(f'Scene differs from frozen manifest: {case["path"]}')
        provenance = report['rendererProvenance']
        workload = {k: report['scene'][k] for k in ['frequencies', 'sampleRate', 'fftSize', 'rays', 'seeds', 'maxBounces', 'maximumDelaySeconds']}
        identity = (provenance['files'], provenance['fixtureSha256'], workload)
        if runtime is not None and runtime != identity:
            raise ValueError('Cannot pool different renderer implementations or workloads into one score.')
        runtime = identity
        latest = [c for c in report['convergence'] if c['fine'] == min(report['scene']['meshSizes'])]
        if len(latest) != len(report['scene']['frequencies']) or any(
                c['relativeL2'] > .005 or (c['shadowRelativeL2'] or 0) > .005 for c in latest):
            warnings.append(case['path']+': reference mesh change exceeds 0.5% or is incomplete')
        if not report['analyticValidation'].get('passed', False):
            raise ValueError('Analytic reference controls have not passed.')
        audit = report.get('numericalAudit', {})
        if not audit.get('complete') or any(max(c['relativeL2'], c['shadowRelativeL2'] or 0) > .005 for c in audit.get('checks', [])):
            warnings.append(case['path']+': full numerical audit missing or exceeds 0.5%')
        if report.get('metricVersion') != 2:
            raise ValueError('Re-run plot with the current reporter to obtain separate lit/shadow errors.')
        config = report['scene']; expected = len(config['frequencies'])*len(config['seeds'])
        shadow = shadow_mask(config, np.array(config['points']).reshape(-1, 3))
        regions = ['all'] + (['lit'] if (~shadow).any() else []) + (['shadow'] if shadow.any() else [])
        for diffraction, rays in [(0, max(config['rays']))]+[(1, r) for r in config['rays']]:
            selected = [r for r in report['comparisons'] if r['diffraction'] == diffraction and r['rays'] == rays]
            per_region = {}
            for region in regions:
                rows = [r for r in selected if r['region'] == region]
                keys = {(r['hz'], r['seed']) for r in rows}
                wanted = {(hz, seed) for hz in config['frequencies'] for seed in config['seeds']}
                if len(rows) != expected or keys != wanted:
                    raise ValueError('Incomplete/duplicate scene-frequency-seed-region measurements.')
                if region != 'all':
                    per_region[region] = float(np.mean([r['complexRelativeL2']**2 for r in rows]))
            if not per_region:
                raise ValueError('No physical regions measured.')
            key = f'diffraction={diffraction},rays={rays}'
            all_rows = [r for r in selected if r['region'] == 'all']
            groups.setdefault(key, {}).setdefault(case['family'], []).append(dict(
                case=case['path'], meanSquaredError=float(np.mean(list(per_region.values()))),
                shadowSquaredError=per_region.get('shadow'),
                warmSolveMs=float(np.mean([r['medianSolveMs'] for r in all_rows])),
                worstSeedRegionPercent=100*max(r['complexRelativeL2'] for r in selected if r['region'] != 'all')))
    scores = []
    for method, families in groups.items():
        mse = np.mean([np.mean([c['meanSquaredError'] for c in cases]) for cases in families.values()])
        scores.append(dict(method=method, balancedComplexNrmsePercent=float(100*np.sqrt(mse)),
            meanWarmSolveMs=float(np.mean([np.mean([c['warmSolveMs'] for c in cases]) for cases in families.values()])),
            worstSeedRegionPercent=max(c['worstSeedRegionPercent'] for cases in families.values() for c in cases),
            families=families))
    result = dict(profile=manifest['profile'], split=split, cases=len(cases), scores=scores,
        status='provisional' if warnings else 'reference refinement checks passed', warnings=warnings,
        definition='100*sqrt(mean_family(mean_geometry(mean_frequency_seed_region(complex relative L2 squared)))); lit/shadow regions receive equal weight. Lower is better. No clipping.',
        timing='Mean of per-run median transport submission/readback times. Excludes DSP/inference and cold dispatch.')
    save_json(output/f'score-{split}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'scores'}, indent=2))
    for score in scores:
        print({k: v for k, v in score.items() if k != 'families'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'suite', 'score', 'validate', 'reference', 'audit', 'renderer', 'plot'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--planar', choices=['free', 'barrier', 'staggered'], help='prepare a point-source/extruded-wall FEM fixture')
    parser.add_argument('--profile', choices=['pilot', 'study'], default='pilot', help='suite workload; generation does not run it')
    parser.add_argument('--split', choices=['development', 'validation', 'test'], default='development')
    parser.add_argument('--room-control', action='store_true', help='renderer: analytic early room/material/motion control; no scene.json needed')
    parser.add_argument('--mesh', type=float)
    parser.add_argument('--hz', type=float)
    parser.add_argument('--runner', type=Path)
    parser.add_argument('--unity', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'prepare':
        if args.planar:
            from planar_wave import prepare as planar_prepare
            planar_prepare(args.output, args.planar)
        else:
            prepare(args.output)
    elif args.command == 'suite':
        prepare_suite(args.output, args.profile)
    elif args.command == 'score':
        score_suite(args.output, args.split)
    elif args.command == 'audit':
        from planar_wave import audit
        audit(args.output)
    elif args.command == 'renderer':
        if args.runner is None or args.unity is None:
            parser.error('renderer requires --runner and --unity')
        renderer(args.output, args.runner, args.unity, args.room_control)
    elif args.command == 'plot':
        plot(args.output)
    else:
        reference(args.output, args.command == 'validate', args.mesh, args.hz)


if __name__ == '__main__':
    main()
