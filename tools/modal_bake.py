"""Offline homogeneous solid vibration: primitive fitting, FEM/plate bakes, retuning.

Units are metres, kg, seconds, Pa. Output is mechanical displacement/velocity,
NOT acoustic pressure. Requires numpy/scipy; solid bakes also require NGSolve.
Run `python tools/modal_bake.py --help`. No audio playback or Unity dependency.
"""
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


VERSION = 1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def material(spec):
    result = {key: float(spec[key]) for key in ('density', 'young', 'poisson', 'loss')}
    if (not np.isfinite(list(result.values())).all() or result['density'] <= 0
            or result['young'] <= 0 or not -1 < result['poisson'] < .5
            or not 0 <= result['loss'] < 2):
        raise ValueError('Require density/Young > 0, -1 < Poisson < .5, 0 <= loss < 2.')
    return result


def frame(geometry):
    center = np.asarray(geometry.get('center', [0, 0, 0]), dtype=float)
    basis = np.asarray(geometry.get('basis', np.eye(3)), dtype=float).reshape(3, 3)
    if (center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(basis).all()
            or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6)
            or np.linalg.det(basis) < .999999):
        raise ValueError('Frame requires a finite center and right-handed orthonormal basis.')
    return center, basis


def box_size(geometry):
    size = np.asarray(geometry['size'], dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('Primitive dimensions must be three positive lengths.')
    return size


def box_surface(geometry):
    size = box_size(geometry)
    center, basis = frame(geometry)
    points = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                       for z in (-.5, .5)]) * size
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return points @ basis.T + center, faces


def solid_surface(geometry):
    if geometry['kind'] == 'box':
        vertices, triangles = box_surface(geometry)
    elif geometry['kind'] == 'mesh':
        vertices = np.asarray(geometry['vertices'], dtype=float).reshape(-1, 3)
        raw = np.asarray(geometry['triangles'])
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError('Triangle indices must be integers.')
        triangles = raw.reshape(-1, 3)
    else:
        raise ValueError('Solid geometry must be box or mesh.')
    if (len(vertices) < 4 or len(triangles) < 4 or not np.isfinite(vertices).all()
            or triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError('Invalid solid surface.')
    # Rendering seams often duplicate vertices. Weld exact positions, not gaps.
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    triangles = inverse[triangles]
    used, triangles = np.unique(triangles, return_inverse=True)
    vertices, triangles = vertices[used], triangles.reshape(-1, 3)
    xyz = vertices[triangles]
    cross = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    diagonal = np.linalg.norm(np.ptp(vertices, axis=0))
    if np.any(np.linalg.norm(cross, axis=1) <= diagonal**2 * 1e-14):
        raise ValueError('Degenerate surface triangle.')
    edges = {}
    adjacency = [[] for _ in triangles]
    for face, ids in enumerate(triangles):
        for a, b in zip(ids, np.roll(ids, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append((face, a < b))
    for uses in edges.values():
        if len(uses) != 2 or uses[0][1] == uses[1][1]:
            raise ValueError('Require a closed, consistently oriented manifold solid surface.')
        a, b = uses[0][0], uses[1][0]
        adjacency[a].append(b); adjacency[b].append(a)
    seen, pending = {0}, [0]
    while pending:
        for face in adjacency[pending.pop()]:
            if face not in seen:
                seen.add(face); pending.append(face)
    if len(seen) != len(triangles):
        raise ValueError('Bake one connected physical solid at a time; disconnected shells are ambiguous.')
    relative = xyz - vertices.mean(axis=0)
    volume = np.einsum('ij,ij->i', relative[:, 0],
                       np.cross(relative[:, 1], relative[:, 2])).sum()/6
    if abs(volume) <= diagonal**3 * 1e-12:
        raise ValueError('Solid has zero signed volume.')
    if volume < 0:
        triangles = triangles[:, [0, 2, 1]]
    return vertices, triangles, abs(float(volume))


def load_obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith('#'):
            continue
        if parts[0] == 'v':
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == 'f':
            if len(parts) != 4:
                raise ValueError('OBJ must be triangulated; polygon triangulation is not implicit.')
            ids = [int(p.split('/')[0]) for p in parts[1:]]
            if any(i == 0 for i in ids):
                raise ValueError('OBJ indices cannot be zero.')
            faces.append([i-1 if i > 0 else len(vertices)+i for i in ids])
    return dict(kind='mesh', vertices=vertices, triangles=faces)


def read_spec(path):
    path = Path(path)
    spec = json.loads(path.read_text())
    material(spec['material'])
    geometry = spec['geometry']
    if geometry.get('kind') == 'obj':
        source = path.parent / geometry['file']
        spec['geometry'] = load_obj(source)
        spec['meshFileSha256'] = digest(source)
    return spec


def fit_box(spec):
    """Oriented bounding-box candidates from area moments and dominant face normals.

    The chosen minimum-volume candidate is an enclosing geometric approximation,
    not a minimum-volume OBB guarantee or a vibration-error estimate.
    """
    if spec.get('support', 'free') != 'free':
        raise ValueError('Fit a free object first, then choose support in the fitted frame.')
    vertices, faces, volume = solid_surface(spec['geometry'])
    xyz = vertices[faces]
    crosses = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    areas = np.linalg.norm(crosses, axis=1)/2
    centroid = (xyz.mean(axis=1)*areas[:, None]).sum(axis=0)/areas.sum()
    centered = xyz-centroid
    sums = centered.sum(axis=1)
    # Exact area-integrated triangle second moments; independent of tessellation density.
    moments = (np.einsum('ti,tj->tij', sums, sums)
               + np.einsum('tki,tkj->tij', centered, centered))/12
    covariance = np.einsum('t,tij->ij', areas/areas.sum(), moments)
    _, principal = np.linalg.eigh(covariance)
    candidates = [np.eye(3), principal[:, ::-1]]
    normals = crosses/(2*areas[:, None])
    directions = []
    for index in np.argsort(-areas, kind='stable'):
        normal = normals[index]
        if all(abs(normal @ old) < 1-1e-8 for old in directions):
            directions.append(normal)
            if len(directions) == 16:
                break
    for normal in directions:
        for other in directions:
            tangent = other-normal*(normal @ other)
            if np.linalg.norm(tangent) > 1e-6:
                tangent /= np.linalg.norm(tangent)
                candidates.append(np.column_stack((normal, tangent, np.cross(normal, tangent))))
    best = None
    for axes in candidates:
        coordinates = (vertices-centroid) @ axes
        lo, hi = coordinates.min(axis=0), coordinates.max(axis=0)
        size = hi-lo
        score = np.prod(size)
        if best is None or score < best[0]*(1-1e-12):
            best = score, axes.copy(), (hi+lo)/2, size
    _, axes, local_center, size = best
    center = centroid+axes @ local_center
    order = np.argsort(-size, kind='stable')
    axes, size = axes[:, order], size[order]
    for axis in range(2):
        if axes[np.argmax(abs(axes[:, axis])), axis] < 0:
            axes[:, axis] *= -1
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    # Deterministic area-stratified samples; no dependence on audio/render seed.
    rng = np.random.default_rng(0)
    ids = np.searchsorted(np.cumsum(areas)/areas.sum(), (np.arange(4096)+.5)/4096)
    r = np.sqrt(rng.random(4096)); t = rng.random(4096)
    samples = (1-r[:, None])*xyz[ids, 0]+(r*(1-t))[:, None]*xyz[ids, 1]+(r*t)[:, None]*xyz[ids, 2]
    distances = np.min(np.maximum(size/2-abs((samples-center) @ axes), 0), axis=1)
    diagonal = np.linalg.norm(size)
    report = dict(method='area-moments-and-face-normal-obb-v1', samples=len(samples),
                  distanceDefinition='sampled source surface to enclosing box, one-way',
                  surfaceRmsMeters=float(np.sqrt(np.mean(distances**2))),
                  surfaceP95Meters=float(np.quantile(distances, .95)),
                  surfaceRmsOverDiagonal=float(np.sqrt(np.mean(distances**2))/diagonal),
                  volumeFill=float(volume/np.prod(size)),
                  volumeRelativeError=float(abs(np.prod(size)-volume)/volume),
                  thinness=float(size[2]/size[1]),
                  acousticError='unmeasured; geometric fit does not certify modal accuracy',
                  solidValidation='edge manifold/orientation/connectivity/volume; self intersections not certified')
    return dict(version=VERSION, geometry=dict(kind='box', size=size.tolist(),
                center=center.tolist(), basis=axes.tolist()), material=material(spec['material']),
                support='free', fit=report)


def plate_modes(spec, count, maximum_hz):
    geometry, mat = spec['geometry'], material(spec['material'])
    a, b, h = box_size(geometry)
    center, basis = frame(geometry)
    if spec.get('support') != 'simply-supported':
        raise ValueError('Analytical plate supports simply-supported edges only.')
    if h/min(a, b) > .05:
        raise ValueError('Thin-plate reference requires thickness <= 5% of the smaller span; use solid FEM.')
    rigidity = mat['young']*h**3/(12*(1-mat['poisson']**2))
    factor = math.pi/2*math.sqrt(rigidity/(mat['density']*h))
    def frequency(m, n): return factor*((m/a)**2+(n/b)**2)
    pending, seen, indices, frequencies = [(frequency(1, 1), 1, 1)], {(1, 1)}, [], []
    while pending and len(frequencies) < count:
        hz, m, n = heapq.heappop(pending)
        if hz > maximum_hz:
            break
        # Kirchhoff plate theory also requires thickness small against modal wavelength.
        if h*math.sqrt((m/a)**2+(n/b)**2) > .1:
            break
        indices.append((m, n)); frequencies.append(hz)
        for key in ((m+1, n), (m, n+1)):
            if key not in seen:
                seen.add(key); heapq.heappush(pending, (frequency(*key), *key))
    if not frequencies:
        raise ValueError('No valid plate modes in the requested band.')
    # Surface interpolation grid resolves the highest retained sinusoidal mode.
    nx = max(8, 8*max(m for m, _ in indices)); ny = max(8, 8*max(n for _, n in indices))
    positions = np.array([[x, y, h/2] for x in np.linspace(-a/2, a/2, nx+1)
                         for y in np.linspace(-b/2, b/2, ny+1)])
    faces = []
    for i in range(nx):
        for j in range(ny):
            k = i*(ny+1)+j
            faces.extend([[k, k+ny+1, k+ny+2], [k, k+ny+2, k+1]])
    amplitude = math.sqrt(4/(mat['density']*a*b*h))
    shapes = np.zeros((len(indices), len(positions), 3))
    for i, (m, n) in enumerate(indices):
        shapes[i, :, 2] = amplitude*np.sin(m*np.pi*(positions[:, 0]/a+.5))*np.sin(n*np.pi*(positions[:, 1]/b+.5))
    return np.asarray(frequencies), positions @ basis.T+center, np.asarray(faces), shapes @ basis.T, dict(
        method='kirchhoff-simply-supported-plate-v1', indices=indices,
        volume=a*b*h, cutoffReason='count, frequency or thin-plate modal wavelength limit',
        sampledSurface='upper face; transverse displacement only; no radiation model')


def fem_modes(spec, count, maximum_hz, mesh_size, order):
    import ngsolve as ng
    from netgen.occ import Box, OCCGeometry, Pnt
    from netgen.stl import STLGeometry
    ng.SetNumThreads(1)
    geometry, mat = spec['geometry'], material(spec['material'])
    vertices, faces, volume = solid_surface(geometry)
    origin = (vertices.min(axis=0)+vertices.max(axis=0))/2
    basis = np.eye(3)
    if geometry['kind'] == 'box':
        origin, basis = frame(geometry)
    local = (vertices-origin) @ basis
    length = np.linalg.norm(np.ptp(local, axis=0))
    if not mesh_size > 0 or order not in (1, 2, 3):
        raise ValueError('Require positive mesh size and FEM order 1, 2 or 3.')
    if geometry['kind'] == 'box':
        half = box_size(geometry)/2
        mesh = ng.Mesh(OCCGeometry(Box(Pnt(*-half), Pnt(*half))).GenerateMesh(maxh=mesh_size))
    else:
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder)/'solid.stl'
            with filename.open('w') as stream:
                stream.write('solid object\n')
                for triangle in local[faces]:
                    normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
                    normal /= np.linalg.norm(normal)
                    stream.write('facet normal '+' '.join(map(str, normal))+'\nouter loop\n')
                    for point in triangle:
                        stream.write('vertex '+' '.join(map(str, point))+'\n')
                    stream.write('endloop\nendfacet\n')
                stream.write('endsolid object\n')
            mesh = ng.Mesh(STLGeometry(str(filename)).GenerateMesh(maxh=mesh_size))
    mesh_volume = float(ng.Integrate(1, mesh))
    if abs(mesh_volume-volume)/volume > .01:
        raise ValueError('Tetrahedral volume differs from the input by more than 1%. Inspect the solid.')
    space = ng.VectorH1(mesh, order=order)
    free = np.array(list(space.FreeDofs()), dtype=bool)
    support = spec.get('support', 'free')
    if support != 'free':
        if support not in ('clamp-x-min', 'clamp-y-min', 'clamp-z-min'):
            raise ValueError('Solid supports: free or clamp-x/y/z-min in the geometry frame.')
        axis = 'xyz'.index(support[6]); value = local[:, axis].min()
        for element in mesh.Elements(ng.BND):
            if all(abs(mesh[v].point[axis]-value) < length*1e-7 for v in element.vertices):
                for dof in space.GetDofNrs(element):
                    if dof >= 0:
                        free[dof] = False
        if free.all():
            raise ValueError('No boundary face matched the requested clamp plane.')
    # Geometry matrices at E=rho=1; material scaling is applied after the solve.
    nu = mat['poisson']; mu = 1/(2*(1+nu)); lam = nu/((1+nu)*(1-2*nu))
    u, v = space.TnT()
    stiffness, mass = ng.BilinearForm(space), ng.BilinearForm(space)
    stiffness += (2*mu*ng.InnerProduct(ng.Sym(ng.Grad(u)), ng.Sym(ng.Grad(v)))+lam*ng.div(u)*ng.div(v))*ng.dx
    mass += ng.InnerProduct(u, v)*ng.dx
    stiffness.Assemble(); mass.Assemble()
    def sparse(form):
        row, col, val = form.mat.COO()
        return coo_matrix((val, (row, col)), shape=(space.ndof, space.ndof)).tocsr()[free][:, free]
    K, M = sparse(stiffness), sparse(mass)
    rigid = 6 if support == 'free' else 0
    if count+rigid >= K.shape[0]:
        raise ValueError('Requested modes exceed available mechanical degrees of freedom; refine the mesh.')
    shift = -.01/length**2
    values, vectors = eigsh(K, count+rigid, M=M, sigma=shift, which='LM',
                            v0=np.random.default_rng(0).standard_normal(K.shape[0]), tol=1e-10)
    sorted_ids = np.argsort(values); values, vectors = values[sorted_ids], vectors[:, sorted_ids]
    if rigid and (np.max(abs(values[:6])) > abs(shift)*1e-5 or values[6] <= abs(shift)*1e-5):
        raise ValueError('Expected exactly six rigid-body modes for one connected free solid.')
    values, vectors = values[rigid:], vectors[:, rigid:]
    if np.any(values <= 0):
        raise ValueError('Nonpositive elastic eigenvalue.')
    frequencies = np.sqrt(values*mat['young']/mat['density'])/(2*np.pi)
    keep = frequencies <= maximum_hz
    values, vectors, frequencies = values[keep], vectors[:, keep], frequencies[keep]
    if not len(values):
        raise ValueError('No elastic modes in the requested frequency band.')
    residuals = [np.linalg.norm(K@x-val*(M@x))/(np.linalg.norm(K@x)+abs(val)*np.linalg.norm(M@x))
                 for val, x in zip(values, vectors.T)]
    orthogonality = float(np.max(abs(vectors.T @ (M@vectors)-np.eye(len(values)))))
    if max(residuals) > 1e-5 or orthogonality > 1e-6:
        raise ValueError('Modal eigensolver failed residual or mass-orthogonality checks.')
    # Sample the actual FE field on a subdivided surface for portable impact interpolation.
    points, output_faces, point_ids = [], [], {}
    def point_id(point):
        key = tuple(np.round(point/length, 12))
        if key not in point_ids:
            point_ids[key] = len(points); points.append(point)
        return point_ids[key]
    subdivisions = 2*order
    for element in mesh.Elements(ng.BND):
        triangle = np.array([mesh[v].point for v in element.vertices])
        grid = {}
        for i in range(subdivisions+1):
            for j in range(subdivisions+1-i):
                grid[i, j] = point_id(triangle[0]+i/subdivisions*(triangle[1]-triangle[0])+j/subdivisions*(triangle[2]-triangle[0]))
        for i in range(subdivisions):
            for j in range(subdivisions-i):
                output_faces.append([grid[i, j], grid[i+1, j], grid[i, j+1]])
                if i+j+1 < subdivisions:
                    output_faces.append([grid[i+1, j], grid[i+1, j+1], grid[i, j+1]])
    points = np.asarray(points)
    # Explicit boundary lookup avoids perturbing physical sample locations into the solid.
    locations = [mesh(*point, ng.BND) for point in points]
    field = ng.GridFunction(space)
    shapes = []
    for vector in vectors.T:
        field.vec.FV().NumPy()[:] = 0
        field.vec.FV().NumPy()[free] = vector/math.sqrt(mat['density'])
        shapes.append([field(point) for point in locations])
    return frequencies, points @ basis.T+origin, np.asarray(output_faces), np.asarray(shapes) @ basis.T, dict(
        method='solid-linear-elastic-fem-v1', ngsolveVersion=ng.__version__,
        meshSizeMeters=mesh_size, order=order, dofs=space.ndof, freeDofs=int(free.sum()),
        rigidModesRemoved=rigid, volume=mesh_volume, inputVolume=volume,
        relativeResidualMax=float(max(residuals)), massOrthogonalityMax=orthogonality,
        surfaceSubdivisions=subdivisions,
        convergence='single discretization; compare mesh/order refinements before accepting accuracy',
        sampledSurface='piecewise-linear interpolation of sampled FE displacement; no radiation model')


def write_bank(path, metadata, frequencies, positions, triangles, shapes):
    arrays = dict(frequencies=np.asarray(frequencies, dtype='<f8'), positions=np.asarray(positions, dtype='<f8'),
                  triangles=np.asarray(triangles, dtype='<i4'), shapes=np.asarray(shapes, dtype='<f8'))
    metadata = dict(metadata, version=VERSION, units=dict(frequencies='Hz', positions='m',
        shapes='kg^-1/2', impulse='N s', displacement='m', velocity='m/s'),
        radiation='not included; mechanical surface motion must be converted to acoustic sources')
    arrays['metadata'] = np.array(json.dumps(metadata, allow_nan=False))
    with Path(path).open('wb') as stream:
        np.savez_compressed(stream, **arrays)


def read_bank(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in ('frequencies', 'positions', 'triangles', 'shapes')}
        result['metadata'] = json.loads(str(data['metadata']))
    f, p, t, shapes = (result[key] for key in ('frequencies', 'positions', 'triangles', 'shapes'))
    if (result['metadata']['version'] != VERSION or f.ndim != 1 or not len(f)
            or p.ndim != 2 or p.shape[1] != 3 or t.ndim != 2 or t.shape[1] != 3
            or shapes.shape != (len(f), len(p), 3) or np.any(f <= 0) or np.any(np.diff(f) < 0)
            or t.min() < 0 or t.max() >= len(p)
            or not all(np.isfinite(x).all() for x in (f, p, shapes))):
        raise ValueError('Invalid modal bank.')
    material(result['metadata']['material'])
    return result


def bake(spec, path, count=64, maximum_hz=16000, mesh_size=.1, order=2):
    if count < 1 or not np.isfinite(maximum_hz) or maximum_hz <= 0:
        raise ValueError('Require positive mode count and maximum frequency.')
    started = time.perf_counter()
    if spec['geometry']['kind'] == 'plate':
        frequencies, positions, triangles, shapes, report = plate_modes(spec, count, maximum_hz)
    else:
        frequencies, positions, triangles, shapes, report = fem_modes(spec, count, maximum_hz, mesh_size, order)
    report.update(seconds=time.perf_counter()-started, retainedModes=len(frequencies),
                  requestedModes=count, maximumHz=maximum_hz)
    metadata = dict(material=material(spec['material']), support=spec.get('support', 'free'),
                    input=spec, report=report, uniformScale=1.)
    write_bank(path, metadata, frequencies, positions, triangles, shapes)
    return report


def retune(bank, path, young=None, density=None, loss=None, scale=1.):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Uniform scale must be positive.')
    metadata = dict(bank['metadata']); old = material(metadata['material']); new = dict(old)
    for key, value in (('young', young), ('density', density), ('loss', loss)):
        if value is not None:
            new[key] = value
    new = material(new)
    metadata['material'] = new
    metadata['uniformScale'] *= scale
    metadata['retuning'] = dict(note='Uniform scale about origin; same Poisson ratio/support. No added high modes.',
        previousMaterial=old, scale=scale, frequencyFactor=math.sqrt(new['young']/old['young']*old['density']/new['density'])/scale)
    write_bank(path, metadata, bank['frequencies']*math.sqrt(new['young']/old['young']*old['density']/new['density'])/scale,
               bank['positions']*scale, bank['triangles'],
               bank['shapes']*math.sqrt(old['density']/(new['density']*scale**3)))


def compare(reference, candidate):
    """Compare ordered elastic spectra, not individual mode identity or sound quality."""
    a, b = reference['frequencies'], candidate['frequencies']
    count = min(len(a), len(b))
    errors = b[:count]/a[:count]-1
    return dict(comparedModes=count, referenceModes=len(a), candidateModes=len(b),
                referenceHz=a[:count].tolist(), candidateHz=b[:count].tolist(),
                relativeFrequencyErrors=errors.tolist(),
                maxAbsoluteRelativeError=float(np.max(abs(errors))),
                rmsRelativeError=float(np.sqrt(np.mean(errors**2))),
                interpretation='Ordered spectral comparison only; no mode-shape, damping or acoustic-radiation validation.')


def coupling(bank, triangle, barycentric, direction):
    weights, direction = np.asarray(barycentric, float), np.asarray(direction, float)
    if (not 0 <= triangle < len(bank['triangles']) or weights.shape != (3,) or direction.shape != (3,)
            or not np.isfinite([weights, direction]).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1) or not np.isclose(np.linalg.norm(direction), 1)):
        raise ValueError('Require a surface triangle, barycentric weights and unit force/measurement direction.')
    return np.einsum('mij,i,j->m', bank['shapes'][:, bank['triangles'][triangle]], weights, direction)


class Resonator:
    """Mass-normalized shared modal state; impulses happen before the named sample.

    The loss factor uses the lightly damped approximation zeta=loss/2. Surface
    sampling/retained-mode truncation are explicit approximations. No clipping,
    normalization, propagation or pressure conversion is applied.
    """
    def __init__(self, bank, sample_rate):
        if sample_rate <= 0 or np.any(bank['frequencies'] >= sample_rate/2):
            raise ValueError('All retained modes must lie below Nyquist.')
        self.bank, self.sample_rate, self.sample = bank, sample_rate, 0
        omega = 2*np.pi*bank['frequencies']
        self.alpha = bank['metadata']['material']['loss']*omega/2
        self.beta = np.sqrt(omega**2-self.alpha**2)
        dt = 1/sample_rate; c = np.cos(self.beta*dt); s = np.sin(self.beta*dt)/self.beta
        e = np.exp(-self.alpha*dt)
        self.a, self.b = e*(c+self.alpha*s), e*s
        self.c, self.d = -e*omega**2*s, e*(c-self.alpha*s)
        self.q, self.v = np.zeros_like(omega), np.zeros_like(omega)

    def render(self, count, events, observation):
        if count < 0 or observation.shape != self.q.shape or not np.isfinite(observation).all():
            raise ValueError('Invalid sample count or observation coupling.')
        # Caller supplies only this batch's events; never silently lose an out-of-range event.
        schedule = {}
        for event in events:
            at = event['sample']
            if not isinstance(at, int) or not self.sample <= at < self.sample+count:
                raise ValueError('Impact timestamp must be an integer inside this batch.')
            impulse = float(event['impulse'])
            if not np.isfinite(impulse):
                raise ValueError('Non-finite impulse.')
            vector = impulse*coupling(self.bank, event['triangle'], event['barycentric'], event['direction'])
            schedule[at] = schedule.get(at, 0)+vector
        output = np.empty((count, 2))
        for i in range(count):
            self.v += schedule.get(self.sample+i, 0)
            output[i] = observation @ self.q, observation @ self.v
            self.q, self.v = self.a*self.q+self.b*self.v, self.c*self.q+self.d*self.v
        self.sample += count
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='Propose an oriented solid box with geometric error metrics.')
    fit.add_argument('input', type=Path); fit.add_argument('output', type=Path)
    bake_parser = commands.add_parser('bake', help='Bake plate formulas or solid FEM into a portable NPZ bank.')
    bake_parser.add_argument('input', type=Path); bake_parser.add_argument('output', type=Path)
    bake_parser.add_argument('--modes', type=int, default=64)
    bake_parser.add_argument('--max-hz', type=float, default=16000)
    bake_parser.add_argument('--mesh-size', type=float, default=.1)
    bake_parser.add_argument('--order', type=int, default=2)
    tune = commands.add_parser('retune', help='Reuse a bank for uniform size/stiffness/density/loss changes.')
    tune.add_argument('input', type=Path); tune.add_argument('output', type=Path)
    for option in ('young', 'density', 'loss'):
        tune.add_argument('--'+option, type=float)
    tune.add_argument('--scale', type=float, default=1.)
    render = commands.add_parser('impulse', help='Render mechanical surface displacement/velocity, never play audio.')
    render.add_argument('input', type=Path); render.add_argument('events', type=Path); render.add_argument('output', type=Path)
    render.add_argument('--rate', type=int, default=48000); render.add_argument('--samples', type=int, default=48000)
    render.add_argument('--batch', type=int, default=1024)
    comparison = commands.add_parser('compare', help='Compare spectra for mesh refinement or primitive approximation.')
    comparison.add_argument('reference', type=Path); comparison.add_argument('candidate', type=Path)
    comparison.add_argument('output', type=Path)
    export = commands.add_parser('export', help='Export a bank as JSON for the shared Unity ModalBody runtime.')
    export.add_argument('input', type=Path); export.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Output already exists; choose a new path to preserve evidence.')
    if args.command == 'fit':
        spec = read_spec(args.input)
        result = fit_box(spec); result['sourceSha256'] = digest(args.input)
        if 'meshFileSha256' in spec: result['meshFileSha256'] = spec['meshFileSha256']
        save_json(args.output, result); print(json.dumps(result['fit'], indent=2))
    elif args.command == 'bake':
        spec = read_spec(args.input); spec['sourceSha256'] = digest(args.input)
        print(json.dumps(bake(spec, args.output, args.modes, args.max_hz, args.mesh_size, args.order), indent=2))
    elif args.command == 'retune':
        retune(read_bank(args.input), args.output, args.young, args.density, args.loss, args.scale)
    elif args.command == 'compare':
        result = compare(read_bank(args.reference), read_bank(args.candidate))
        result.update(referenceSha256=digest(args.reference), candidateSha256=digest(args.candidate))
        save_json(args.output, result); print(json.dumps(result, indent=2))
    elif args.command == 'export':
        bank = read_bank(args.input)
        result = {key: bank[key].ravel().tolist() for key in ('frequencies', 'positions', 'triangles', 'shapes')}
        result.update(version=VERSION, loss=bank['metadata']['material']['loss'],
                      bankSha256=digest(args.input), quantity='mechanical-surface-motion',
                      description=bank['metadata']['radiation'])
        save_json(args.output, result)
    else:
        if args.batch < 1 or args.samples < 1:
            raise ValueError('Positive batch and sample counts required.')
        bank = read_bank(args.input); data = json.loads(args.events.read_text())
        observer = data['observation']
        readout = coupling(bank, observer['triangle'], observer['barycentric'], observer['direction'])
        events = sorted(data['impacts'], key=lambda e: e['sample'])
        if any(not isinstance(e['sample'], int) or not 0 <= e['sample'] < args.samples for e in events):
            raise ValueError('Every impact must lie inside the requested output interval.')
        state = Resonator(bank, args.rate); blocks = []; cursor = 0
        while state.sample < args.samples:
            count = min(args.batch, args.samples-state.sample); end = cursor
            while end < len(events) and events[end]['sample'] < state.sample+count:
                end += 1
            blocks.append(state.render(count, events[cursor:end], readout)); cursor = end
        with args.output.open('wb') as stream:
            np.savez_compressed(stream, mechanical=np.concatenate(blocks), sampleRate=args.rate,
                                columns=np.array(['displacement_m', 'velocity_m_per_s']),
                                bankSha256=digest(args.input), eventSha256=digest(args.events))


if __name__ == '__main__':
    main()
