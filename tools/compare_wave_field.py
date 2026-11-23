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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'suite', 'validate', 'reference', 'audit'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--planar', choices=['free', 'barrier', 'staggered'], help='prepare a point-source/extruded-wall FEM fixture')
    parser.add_argument('--profile', choices=['pilot', 'study'], default='pilot', help='suite workload; generation does not run it')
    parser.add_argument('--mesh', type=float)
    parser.add_argument('--hz', type=float)
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
    elif args.command == 'audit':
        from planar_wave import audit
        audit(args.output)
    else:
        reference(args.output, args.command == 'validate', args.mesh, args.hz)


if __name__ == '__main__':
    main()
