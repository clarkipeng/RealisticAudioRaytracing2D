"""Finite 3D room references with passive local admittance; optional NGSolve.

Convention: exp(+i*w*t), free pressure exp(-i*k*r)/r. Material admittance
is dimensionless rho*c/Z. No PML, fitted gains or ray-derived reference.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from compare_wave_field import save_json


def admittance(material, hz):
    frequencies = np.asarray(material['frequencies'], dtype=float)
    real = np.asarray(material['admittanceReal'], dtype=float)
    imag = np.asarray(material['admittanceImag'], dtype=float)
    if (frequencies.ndim != 1 or not len(frequencies) or real.shape != frequencies.shape
            or imag.shape != frequencies.shape or not np.isfinite([frequencies, real, imag]).all()
            or np.any(frequencies < 0) or np.any(np.diff(frequencies) <= 0) or np.any(real < 0)):
        raise ValueError('Admittance requires increasing frequencies, finite values and passive real part.')
    return np.interp(hz, frequencies, real)+1j*np.interp(hz, frequencies, imag)


def reflection(material, hz, cosine):
    y = admittance(material, hz)
    return (cosine-y)/(cosine+y) if abs(cosine+y) else 1+0j


class RoomField:
    def __init__(self, config, h, order):
        import ngsolve as ng
        from netgen.occ import Box, Pnt, OCCGeometry, Z
        ng.SetNumThreads(1)
        self.ng, self.config = ng, config
        started = time.perf_counter()
        def box(spec):
            center, half = np.asarray(spec['center']), np.asarray(spec['half'])
            if center.shape != (3,) or half.shape != (3,) or not np.isfinite([center, half]).all() or np.any(half <= 0):
                raise ValueError('Invalid room box.')
            return Box(Pnt(*(center-half)), Pnt(*(center+half)))
        air = None
        for volume in config['airVolumes']:
            part = box(volume); part.faces.name = 'wall'
            part.faces.Min(Z).name = 'floor'; part.faces.Max(Z).name = 'ceiling'
            air = part if air is None else air+part
        for index, obstacle in enumerate(config.get('obstacles', [])):
            solid = box(obstacle); solid.faces.name = f'obstacle{index}'
            air = air-solid
        self.mesh = ng.Mesh(OCCGeometry(air).GenerateMesh(maxh=h))
        self.space = ng.H1(self.mesh, order=order, complex=True)
        self.k = ng.Parameter(1)
        u, v = self.space.TnT()
        # Solve only the smooth reflected field; retain the exact monopole singularity.
        offset = ng.CF(tuple(coordinate-value for coordinate, value in zip((ng.x, ng.y, ng.z), config['source'])))
        distance = ng.sqrt(offset*offset)
        self.incident = ng.exp(-1j*self.k*distance)/distance
        self.load = ng.LinearForm(self.space)
        self.operator = ng.BilinearForm(self.space, symmetric=True)
        self.operator += (ng.grad(u)*ng.grad(v)-self.k*self.k*u*v)*ng.dx
        self.boundaries = {}
        for name in set(self.mesh.GetBoundaries()):
            real, imag = ng.Parameter(0), ng.Parameter(0)
            self.boundaries[name] = real, imag
            self.operator += 1j*self.k*(real+1j*imag)*u*v*ng.ds(name)
            normal_derivative = self.incident*(-1j*self.k-1/distance)*(offset*ng.specialcf.normal(3))/distance
            self.load += -(normal_derivative+1j*self.k*(real+1j*imag)*self.incident)*v*ng.ds(name)
        self.field = ng.GridFunction(self.space)
        self.stats = dict(meshSize=h, order=order, dofs=self.space.ndof,
                          meshSeconds=time.perf_counter()-started, ngsolveVersion=ng.__version__)

    def solve(self, hz, points=None):
        self.k.Set(2*np.pi*hz/self.config['speedOfSound'])
        for name, (real, imag) in self.boundaries.items():
            index = self.config['boundaryMaterials'].get(name, 0)
            value = admittance(self.config['materials'][index], hz)
            real.Set(value.real); imag.Set(value.imag)
        self.operator.Assemble(); self.load.Assemble()
        self.field.vec.data = self.operator.mat.Inverse(self.space.FreeDofs(), inverse='umfpack')*self.load.vec
        return self.evaluate(self.config['points'] if points is None else points)

    def evaluate(self, points):
        points = np.asarray(points).reshape(-1, 3)
        if np.any(np.linalg.norm(points-self.config['source'], axis=1) < 1e-8):
            raise ValueError('Point-source pressure is singular at the source.')
        value = np.asarray([(self.field+self.incident)(self.mesh(*point)) for point in points], dtype=complex)
        if not np.isfinite(value).all():
            raise ValueError('Non-finite room pressure.')
        return value

    def surface(self):
        """Same exposed planar boundary for the GPU; no overlapping wall boxes."""
        triangles, materials = [], []
        for element in self.mesh.Elements(self.ng.BND):
            if len(element.vertices) != 3:
                raise ValueError('Expected triangular room boundary.')
            for vertex in element.vertices:
                triangles.extend(self.mesh[vertex].point)
            name = self.mesh.GetBoundaries()[element.index]
            materials.append(self.config['boundaryMaterials'].get(name, 0))
        return triangles, materials


def prepare(output):
    if (output/'suite.json').exists():
        raise FileExistsError('Preserve the existing room suite.')
    def box(center, half): return dict(center=center, half=half)
    def material(name, real):
        return dict(name=name, frequencies=[0, 62.5, 125, 250],
                    admittanceReal=[real]*4, admittanceImag=[0]*4)
    hard, resistive = material('low-loss', .02), material('resistive', .5)
    frequencies = np.array([0, 31.25, 62.5, 125, 250, 500, 1000.])
    y = np.zeros(len(frequencies), dtype=complex)
    y[1:] = 1/(.5+1j*(frequencies[1:]/125-125/frequencies[1:]))
    reactive = dict(name='synthetic-mass-spring-damper', frequencies=frequencies.tolist(),
                    admittanceReal=y.real.tolist(), admittanceImag=y.imag.tolist())
    default = [box([0, 0, 0], [3, 2, 1.5])]
    families = {
        'room': (default, [], [hard], {}),
        'resistive': (default, [], [resistive], {}),
        'reactive': (default, [], [reactive], {}),
        'treated': (default, [], [hard, resistive], {'ceiling': 1}),
        'coupled': ([box([-1.6, 0, 0], [1.4, 2, 1.5]), box([1.6, 0, 0], [1.4, 2, 1.5]), box([0, 0, -.4], [.3, .45, 1.1])], [], [hard], {}),
        'elbow': ([box([0, -.8, 0], [3, 1.2, 1.5]), box([1.8, .8, 0], [1.2, 1.2, 1.5])], [], [hard], {}),
        'corridor': ([box([0, 0, 0], [4, .8, 1.5])], [], [hard], {}),
        'church': ([box([0, 0, .4], [3, 1, 1.9]), box([0, 0, -.25], [3, 2, 1.25])],
                   [box([x, y, -.25], [.15, .15, 1.25]) for x in [-.75, .75] for y in [-1.2, 1.2]], [hard], {})}
    cases = []
    for name, (volumes, obstacles, materials, boundaries) in families.items():
        path = output/name; path.mkdir()
        points = np.c_[np.linspace(-1, 1.5, 33), np.full(33, .2), np.zeros(33)]
        if name == 'elbow': points[:, 1] = -.4
        config = dict(physicalModel='point-room-3d', family=name, airVolumes=volumes, obstacles=obstacles,
                      materials=materials, boundaryMaterials=boundaries, source=[-1.5, -.5, 0], points=points.tolist(),
                      frequencies=[62.5, 125], speedOfSound=343, meshSizes=[.9, .65], order=3,
                      sampleRate=4000, fftSize=64, rays=[16, 64], seeds=[17, 65554], maxBounces=8,
                      maximumDelaySeconds=.4, referenceConvention='exp(+i*w*t)',
                      receiverMotion=dict(start=points[0].tolist(), velocity=[1.25, 0, 0], duration=2))
        save_json(path/'scene.json', config)
        cases.append(dict(path=name, sceneSha256=hashlib.sha256((path/'scene.json').read_bytes()).hexdigest()))
    save_json(output/'suite.json', dict(version=1, cases=cases, scope='Finite low-frequency room pilot; references require convergence.'))


def reference(output, only_mesh=None, only_hz=None, order=None):
    config = json.loads((output/'scene.json').read_text())
    identity = hashlib.sha256((output/'scene.json').read_bytes()+Path(__file__).read_bytes()).hexdigest()
    for h in config['meshSizes'] if only_mesh is None else [only_mesh]:
        field = RoomField(config, h, config['order'] if order is None else order)
        for hz in config['frequencies'] if only_hz is None else [only_hz]:
            stem = output/f'reference-{h:g}-p{field.stats["order"]}-{hz:g}'
            if Path(str(stem)+'.npz').exists():
                raise FileExistsError('Preserve existing reference results.')
            started = time.perf_counter(); pressure = field.solve(hz)
            # Evaluate a denser walk directly in the solved FE field, without interpolating probe outputs.
            motion = config['receiverMotion']; times = np.arange(round(motion['duration']*1000)+1)/1000
            points = np.asarray(motion['start'])+times[:, None]*motion['velocity']
            moving = field.evaluate(points)
            np.savez_compressed(str(stem)+'.npz', pressure=pressure, points=config['points'],
                                receiverTimes=times, receiverPressure=moving)
            save_json(Path(str(stem)+'.json'), dict(field.stats, hz=hz, seconds=time.perf_counter()-started,
                      solverSha256=identity, sceneSha256=hashlib.sha256((output/'scene.json').read_bytes()).hexdigest()))
            print(config['family'], h, hz, field.stats, flush=True)
        if not (output/'renderer-scene.json').exists():
            triangles, indices = field.surface()
            save_json(output/'renderer-scene.json', dict(config, points=np.asarray(config['points']).ravel().tolist(),
                                                       triangles=triangles, triangleMaterials=indices))


def rigid_room(config, hz, terms):
    """Independent Neumann Green function: transverse modes, exact longitudinal sum.

    Requires x-separated source/probes and an off-resonance frequency. Evanescent
    terms use decaying exponentials to avoid overflow. No FEM or ray paths.
    """
    box = config['airVolumes'][0]; half = np.asarray(box['half']); length = 2*half
    source = np.asarray(config['source'])-box['center']+half
    points = np.asarray(config['points'])-box['center']+half
    if np.min(abs(points[:, 0]-source[0])) < .1:
        raise ValueError('Modal control needs separated x coordinates.')
    lo, hi = np.minimum(points[:, 0], source[0]), np.maximum(points[:, 0], source[0])
    result = np.zeros(len(points), dtype=complex); k = 2*np.pi*hz/config['speedOfSound']
    for ny in range(terms):
        for nz in range(terms):
            transverse = np.ones(len(points))
            for axis, n in [(1, ny), (2, nz)]:
                transverse *= (1 if n == 0 else 2)/length[axis]*np.cos(n*np.pi*source[axis]/length[axis])*np.cos(n*np.pi*points[:, axis]/length[axis])
            q = k*k-(ny*np.pi/length[1])**2-(nz*np.pi/length[2])**2
            if q > 0:
                q = np.sqrt(q)
                green = -np.cos(q*lo)*np.cos(q*(length[0]-hi))/(q*np.sin(q*length[0]))
            else:
                a = np.sqrt(-q); dx = hi-lo
                green = sum(np.exp(-a*d) for d in [dx, lo+hi, 2*length[0]-lo-hi, 2*length[0]-dx])/(2*a*(-np.expm1(-2*a*length[0])))
            result += transverse*green
    return 4*np.pi*result


def validate(output):
    """Independent plane-wave Robin control: both reflection sign and complex phase."""
    import ngsolve as ng
    from netgen.occ import Box, Pnt, OCCGeometry, X
    ng.SetNumThreads(1); checks = []
    geometry = Box(Pnt(0, 0, 0), Pnt(2, .4, .4)); geometry.faces.name = 'wall'
    geometry.faces.Min(X).name = 'source'; geometry.faces.Max(X).name = 'end'
    for order, h in [(2, .3), (3, .2)]:
        mesh = ng.Mesh(OCCGeometry(geometry).GenerateMesh(maxh=h)); space = ng.H1(mesh, order=order, complex=True)
        u, v = space.TnT()
        for hz in [125, 250]:
            k = 2*np.pi*hz/343
            for y in [0, .5, 2, 8, .4+.3j, .4-.3j]:
                r = (1-y)/(1+y); reflected = r*np.exp(-4j*k)
                operator = ng.BilinearForm(space, symmetric=True)
                operator += (ng.grad(u)*ng.grad(v)-k*k*u*v)*ng.dx+1j*k*y*u*v*ng.ds('end')
                load = ng.LinearForm(space); load += 1j*k*(1-reflected)*v*ng.ds('source')
                operator.Assemble(); load.Assemble(); field = ng.GridFunction(space)
                field.vec.data = operator.mat.Inverse(space.FreeDofs(), inverse='umfpack')*load.vec
                x = np.linspace(.1, 1.9, 31); expected = np.exp(-1j*k*x)+reflected*np.exp(1j*k*x)
                actual = np.array([field(mesh(t, .17, .21)) for t in x])
                error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
                checks.append(dict(order=order, meshSize=h, hz=hz, admittance=[complex(y).real, complex(y).imag], relativeL2=error))
                print('impedance', checks[-1], flush=True)
    config = dict(airVolumes=[dict(center=[0, 0, 0], half=[3, 2, 1.5])], source=[-1.5, -.5, 0],
                  points=np.c_[np.linspace(-.7, 1.5, 29), np.full(29, .2), np.zeros(29)].tolist(),
                  materials=[dict(frequencies=[0], admittanceReal=[0], admittanceImag=[0])], boundaryMaterials={}, speedOfSound=343)
    room_checks = []
    for order, h in [(3, .9), (3, .65), (4, .65)]:
        field = RoomField(config, h, order)
        for hz in [61, 117]:
            expected, shorter = rigid_room(config, hz, 64), rigid_room(config, hz, 32)
            actual = field.solve(hz)
            room_checks.append(dict(field.stats, hz=hz,
                relativeL2=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
                seriesRelativeChange=float(np.linalg.norm(expected-shorter)/np.linalg.norm(expected))))
            print('rigid room', room_checks[-1], flush=True)
    passed = (max(c['relativeL2'] for c in checks if c['order'] == 3) < .005
              and max(c['relativeL2'] for c in room_checks if c['order'] == 4) < .005
              and max(c['seriesRelativeChange'] for c in room_checks) < 1e-7)
    save_json(output/'validation.json', dict(checks=checks, roomChecks=room_checks, passed=passed, relativeTolerance=.005,
              solverSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if not passed: raise AssertionError('Refined impedance/room controls exceed tolerance.')


def report(output):
    config = json.loads((output/'scene.json').read_text())
    identity = hashlib.sha256((output/'scene.json').read_bytes()+Path(__file__).read_bytes()).hexdigest()
    metadata = [(p, json.loads(p.read_text())) for p in output.glob('reference-*.json')]
    if not metadata or any(m['solverSha256'] != identity for _, m in metadata):
        raise ValueError('Missing reference or changed solver/configuration.')
    csv = output/'renderer.csv'
    rows = np.genfromtxt(csv, delimiter=',', names=True) if csv.exists() else None
    if rows is not None:
        provenance = json.loads((output/'renderer-provenance.json').read_text())
        if (provenance['sceneSha256'] != hashlib.sha256((output/'scene.json').read_bytes()).hexdigest()
                or provenance['boundarySha256'] != hashlib.sha256((output/'renderer-scene.json').read_bytes()).hexdigest()):
            raise ValueError('GPU measurements use a different scene/boundary.')
    comparisons, convergence = [], []
    for hz in config['frequencies']:
        runs = sorted(((p, m) for p, m in metadata if m['hz'] == hz), key=lambda r:(r[1]['order'], -r[1]['meshSize']))
        if len(runs) < 2: raise ValueError('At least two reference refinements are required.')
        target = np.load(runs[-1][0].with_suffix('.npz'))
        for p, m in runs[:-1]:
            old = np.load(p.with_suffix('.npz'))
            convergence.append(dict(hz=hz, meshSize=m['meshSize'], order=m['order'],
                comparedTo=runs[-1][1]['meshSize'], comparedToOrder=runs[-1][1]['order'],
                relativeL2=float(np.linalg.norm(old['pressure']-target['pressure'])/np.linalg.norm(target['pressure'])),
                receiverRelativeL2=float(np.linalg.norm(old['receiverPressure']-target['receiverPressure'])/np.linalg.norm(target['receiverPressure']))))
        if rows is None: continue
        for diffraction in [0, 1]:
            for rays in config['rays']:
                for seed in config['seeds']:
                    selected = np.sort(rows[(rows['hz'] == hz)&(rows['diffraction'] == diffraction)&(rows['rays'] == rays)&(rows['seed'] == seed)], order='point')
                    if not np.array_equal(selected['point'], np.arange(len(target['pressure']))):
                        raise ValueError('Missing/duplicate receiver measurements.')
                    value = selected['real']+1j*selected['imag']; expected = target['pressure']; residual = value-expected
                    comparisons.append(dict(hz=hz, diffraction=diffraction, rays=rays, seed=seed,
                        complexRelativeL2=float(np.linalg.norm(residual)/np.linalg.norm(expected)),
                        trajectoryDerivativeRelativeL2=float(np.linalg.norm(np.diff(residual))/np.linalg.norm(np.diff(expected))),
                        maximumAbsoluteError=float(abs(residual).max()), medianSolveMs=float(np.median(selected['solve_seconds'])*1000)))
    validation = json.loads((output.parent/'validation.json').read_text())
    if not validation['passed'] or validation['solverSha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Analytic controls are missing, failed or use another solver.')
    result = dict(scene=config, referenceRuns=[m for _, m in metadata], convergence=convergence, comparisons=comparisons,
        analyticValidation=validation,
        scope='Steady finite-room pressure and fixed-source moving-receiver field. No moving-boundary or broadband decay claim.',
        status='provisional: per-scene mesh/order convergence must be assessed',
        sourceSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_json(output/'report.json', result)
    print(json.dumps(dict(convergence=convergence, comparisons=comparisons), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'reference', 'report'])
    parser.add_argument('output', type=Path); parser.add_argument('--mesh', type=float); parser.add_argument('--hz', type=float)
    parser.add_argument('--order', type=int)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'prepare': prepare(args.output)
    elif args.command == 'validate': validate(args.output)
    elif args.command == 'report': report(args.output)
    else: reference(args.output, args.mesh, args.hz, args.order)


if __name__ == '__main__':
    main()
