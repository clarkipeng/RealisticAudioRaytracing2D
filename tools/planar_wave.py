"""Optional NGSolve 6.2.2608 reference for rigid, infinitely extruded XY geometry.

Called through compare_wave_field.py. No runtime dependency. Solve 2D Helmholtz
problems and integrate longitudinal wavenumber to recover a 3D point source.
Only exterior, homogeneous, stationary, lossless air and z=0 probes are supported.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import hankel1, h1vp, jvp, roots_legendre


def relative(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def quadrature(count=48, limit=35., angle=35.):
    # Deform the positive kz axis below its outgoing-wave branch point at k.
    # The negative half is identical at dz=0. Validate contour, tail and order.
    x, w = roots_legendre(count)
    scale = .5*limit*np.exp(-1j*np.deg2rad(angle))
    return (x+1)*scale, w*scale


def integrate(solve, k, **settings):
    q, weights = quadrature(**settings)
    # 4*pi source normalization times 1/pi inverse Fourier integral.
    return 4*sum(w*solve(k*k-z*z) for z, w in zip(q, weights))


class Field:
    def __init__(self, source, points, walls=(), h=.5, order=4, radius=5.,
                 thickness=3., strength=2., cylinder=None):
        import ngsolve as ng
        from netgen.occ import WorkPlane, Glue, OCCGeometry
        ng.SetNumThreads(1)
        started = time.perf_counter()
        source, points = np.asarray(source), np.asarray(points)
        if source.shape != (2,) or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('Expected XY coordinates.')
        if max(np.linalg.norm(source), np.linalg.norm(points, axis=1).max()) >= radius:
            raise ValueError('Sources and probes must be inside the physical domain, outside PML.')
        if np.any(np.linalg.norm(points-source, axis=1) < 1e-8):
            raise ValueError('Point-source pressure is singular at the source.')
        inner = WorkPlane().Circle(radius).Face()
        outer = WorkPlane().Circle(radius+thickness).Face(); outer.edges.name = 'outer'
        shell = outer-inner; shell.faces.name = 'pml'
        for wall in walls:
            c, half = np.array(wall['center'][:2]), np.array(wall['half'][:2])
            if np.any(half <= 0) or np.linalg.norm(abs(c)+half) >= radius:
                raise ValueError('Rigid boxes must lie strictly inside the physical domain.')
            if np.any(np.all(abs(np.vstack([source, points])-c) <= half, axis=1)):
                raise ValueError('Source/probe lies inside a rigid wall.')
            hole = WorkPlane().MoveTo(*(c-half)).Rectangle(*(2*half)).Face()
            inner = inner-hole
        if cylinder is not None:
            inner = inner-WorkPlane().Circle(cylinder).Face()
        inner.faces.name = 'air'
        self.mesh = ng.Mesh(OCCGeometry(Glue([inner, shell]), dim=2).GenerateMesh(maxh=h))
        self.mesh.Curve(order)
        self.mesh.SetPML(ng.pml.Radial(rad=radius, alpha=1j*strength, origin=(0, 0)), 'pml')
        self.space = ng.H1(self.mesh, order=order, complex=True, dirichlet='outer')
        u, v = self.space.TnT()
        self.real, self.imag = ng.Parameter(0), ng.Parameter(0)
        self.operator = ng.BilinearForm(self.space, symmetric=True)
        self.operator += (ng.grad(u)*ng.grad(v)-(self.real+1j*self.imag)*u*v)*ng.dx
        self.load = ng.LinearForm(self.space); self.load += v(*source); self.load.Assemble()
        self.field = ng.GridFunction(self.space)
        self.probes = [self.mesh(*p) for p in points]
        self.stats = dict(dofs=self.space.ndof, meshSize=h, order=order, radius=radius,
                          thickness=thickness, strength=strength,
                          meshSeconds=time.perf_counter()-started, ngsolveVersion=ng.__version__)

    def solve(self, transverse_squared):
        self.real.Set(transverse_squared.real); self.imag.Set(transverse_squared.imag)
        self.operator.Assemble()
        self.field.vec.data = self.operator.mat.Inverse(self.space.FreeDofs(), inverse='umfpack')*self.load.vec
        value = np.array([self.field(p) for p in self.probes], dtype=np.complex128)
        if not np.isfinite(value).all():
            raise ValueError('Non-finite FEM pressure.')
        return value


def cylinder_exact(points, source, k, radius=1., terms=40):
    """Independent outgoing Neumann cylinder solution via cylindrical harmonics."""
    r, rs = np.linalg.norm(points, axis=1), np.linalg.norm(source)
    theta = np.arctan2(points[:, 1], points[:, 0])-np.arctan2(source[1], source[0])
    value = .25j*hankel1(0, k*np.linalg.norm(points-source, axis=1))
    for n in range(terms):
        value -= (.25j*(1 if n == 0 else 2)*hankel1(n, k*rs)*jvp(n, k*radius)
                  / h1vp(n, k*radius)*hankel1(n, k*r)*np.cos(n*theta))
    return value


def prepare(output, family):
    from compare_wave_field import save_json
    if (output/'scene.json').exists():
        raise FileExistsError('Scene already exists; use another output directory.')
    walls = [dict(center=[-.8, -.9, 0], half=[.3, 1.1, 1]),
             dict(center=[.8, .9, 0], half=[.3, 1.1, 1])]
    walls = walls[:{'free': 0, 'barrier': 1, 'staggered': 2}[family]]
    x, y = np.linspace(-2.4, 3.4, 25), np.linspace(-2.6, 2.6, 23)
    xx, yy = np.meshgrid(x, y); points = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    keep = np.ones(len(points), dtype=bool)
    for wall in walls:
        keep &= np.any(abs(points[:, :2]-wall['center'][:2]) > np.array(wall['half'][:2])+.15, axis=1)
    config = dict(physicalModel='point-extruded', family=family,
                  source=[-3, -.4, 0], walls=walls, points=points[keep].ravel().tolist(),
                  gridIndices=np.flatnonzero(keep).tolist(), x=x.tolist(), y=y.tolist(),
                  frequencies=[250], sampleRate=1000, fftSize=8, speedOfSound=343,
                  rays=[8, 128], seeds=[17, 65554, 131091, 196628], maxBounces=4,
                  maximumDelaySeconds=.1, meshSizes=[.5, .35, .25],
                  fem=dict(order=4, radius=5., thickness=3., strength=2.),
                  quadrature=dict(count=48, limit=35., angle=35.),
                  referenceConvention='exp(+i*w*t)')
    save_json(output/'scene.json', config)


def fingerprint(output):
    # Invalidate cached fields after solver changes, not just scene edits.
    return hashlib.sha256((output/'scene.json').read_bytes()+Path(__file__).read_bytes()).hexdigest()


def control_fingerprint(config):
    settings = {key: config[key] for key in ['frequencies', 'speedOfSound', 'fem', 'quadrature']}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()


def validate(output, config):
    from compare_wave_field import save_json
    angle = np.linspace(0, 2*np.pi, 48, endpoint=False)
    points = 1.7*np.c_[np.cos(angle), np.sin(angle)]; source = np.array([-3., -.4])
    checks = []
    def record(name, hz, actual, exact, **stats):
        error = relative(actual, exact)
        checks.append(dict(check=name, hz=hz, relativeL2=error, **stats))
        save_json(output/'validation.json', dict(controlSha256=control_fingerprint(config), checks=checks,
                  relativeTolerance=.005, passed=False))
        print(name, hz, error, stats, flush=True)
        if error >= .005:
            raise AssertionError(f'{name}: {error:.3%} exceeds 0.5% control tolerance.')
    for hz in config['frequencies']:
        k = 2*np.pi*hz/config['speedOfSound']; r = np.linalg.norm(points-source, axis=1)
        free = np.exp(1j*k*r)/r
        record('analytic point-source normalization', hz,
               integrate(lambda k2: .25j*hankel1(0, np.sqrt(k2)*r), k, **config['quadrature']), free)
        exact = integrate(lambda k2: cylinder_exact(points, source, np.sqrt(k2)), k, **config['quadrature'])
        for changes in [dict(count=72), dict(limit=45.), dict(angle=25.)]:
            settings = dict(config['quadrature'], **changes)
            audit = integrate(lambda k2: cylinder_exact(points, source, np.sqrt(k2), terms=60), k, **settings)
            record('analytic cylinder integral/series refinement', hz, audit, exact, quadrature=settings)
        for cylinder in [None, 1.]:
            expected = free if cylinder is None else exact
            for h in [.5, .35]:
                field = Field(source, points, cylinder=cylinder, h=h, **config['fem'])
                line = (.25j*hankel1(0, k*r) if cylinder is None else cylinder_exact(points, source, k))
                record('2D free field' if cylinder is None else '2D rigid cylinder', hz,
                       field.solve(complex(k*k)), line, **field.stats)
                started = time.perf_counter()
                value = integrate(field.solve, k, **config['quadrature'])
                record('2.5D free field' if cylinder is None else '2.5D rigid cylinder', hz,
                       value, expected, seconds=time.perf_counter()-started, **field.stats)
            # Independent changes to FEM order and each PML parameter, using the cylinder.
            if cylinder is not None:
                for change in [dict(order=5), dict(radius=6.), dict(thickness=4.), dict(strength=3.)]:
                    field = Field(source, points, cylinder=cylinder, h=.35, **dict(config['fem'], **change))
                    record('FEM/PML refinement', hz, integrate(field.solve, k, **config['quadrature']),
                           exact, **field.stats)
                reverse = Field(points[0], source[None, :], cylinder=cylinder, h=.35, **config['fem'])
                record('reciprocity', hz, integrate(reverse.solve, k, **config['quadrature']), value[:1])

    save_json(output/'validation.json', dict(controlSha256=control_fingerprint(config), checks=checks,
              relativeTolerance=.005, passed=True))

def reference(output, validate_only=False, only_mesh=None, only_hz=None):
    from compare_wave_field import save_json
    config = json.loads((output/'scene.json').read_text())
    if config['physicalModel'] != 'point-extruded':
        raise ValueError('This backend requires point-extruded geometry.')
    if validate_only:
        return validate(output, config)
    points = np.array(config['points']).reshape(-1, 3); source = np.array(config['source'])
    if source[2] != 0 or np.any(points[:, 2] != 0):
        raise ValueError('Initial 2.5D reconstruction requires sources/probes at z=0.')
    for h in config['meshSizes'] if only_mesh is None else [only_mesh]:
        field = None
        for hz in config['frequencies'] if only_hz is None else [only_hz]:
            stem = output/f'reference-{h:g}-{hz:g}'
            archive, metadata = Path(str(stem)+'.npz'), Path(str(stem)+'.json')
            if archive.exists():
                if json.loads(metadata.read_text())['solverSha256'] != fingerprint(output):
                    raise ValueError('Cached field uses different scene/solver code; use a fresh directory.')
                continue
            if field is None:
                field = Field(source[:2], points[:, :2], config['walls'], h=h, **config['fem'])
            started = time.perf_counter(); k = 2*np.pi*hz/config['speedOfSound']
            pressure = integrate(field.solve, k, **config['quadrature']).conj()
            stats = dict(field.stats, hz=hz, seconds=time.perf_counter()-started,
                         quadrature=config['quadrature'], physicalModel=config['physicalModel'],
                         sceneSha256=hashlib.sha256((output/'scene.json').read_bytes()).hexdigest(),
                         solverSha256=fingerprint(output))
            np.savez_compressed(archive, pressure=pressure, points=points)
            save_json(metadata, stats); print('planar', stats, flush=True)


def audit(output):
    """Change one numerical setting at a time on the actual wall field."""
    from compare_wave_field import save_json, shadow_mask
    config = json.loads((output/'scene.json').read_text())
    if config.get('physicalModel') != 'point-extruded':
        raise ValueError('audit currently supports the planar FEM backend.')
    points = np.array(config['points']).reshape(-1, 3); source = np.array(config['source'])
    shadow = shadow_mask(config, points); h = min(config['meshSizes']); checks = []
    for hz in config['frequencies']:
        stem = output/f'reference-{h:g}-{hz:g}'
        if json.loads(Path(str(stem)+'.json').read_text())['solverSha256'] != fingerprint(output):
            raise ValueError('Reference was generated with different solver code.')
        expected = np.load(str(stem)+'.npz')['pressure']
        for fem_change, q_change in [(dict(order=5), {}), (dict(radius=6.), {}),
                (dict(thickness=4.), {}), (dict(strength=3.), {}),
                ({}, dict(count=72)), ({}, dict(limit=45.)), ({}, dict(angle=25.))]:
            fem, quad = dict(config['fem'], **fem_change), dict(config['quadrature'], **q_change)
            field = Field(source[:2], points[:, :2], config['walls'], h=h, **fem)
            started = time.perf_counter()
            value = integrate(field.solve, 2*np.pi*hz/config['speedOfSound'], **quad).conj()
            check = dict(hz=hz, fem=fem, quadrature=quad, seconds=time.perf_counter()-started,
                         relativeL2=relative(value, expected),
                         shadowRelativeL2=relative(value[shadow], expected[shadow]) if shadow.any() else None,
                         maximumAbsoluteChange=float(abs(value-expected).max()))
            checks.append(check); print('audit', check, flush=True)
            save_json(output/'audit.json', dict(solverSha256=fingerprint(output), checks=checks, complete=False))
    save_json(output/'audit.json', dict(solverSha256=fingerprint(output), checks=checks, complete=True))
