"""Offline planar rigid-box path diagnostic, not a full-wave solver.

Integrates first/second diffraction order and mirror connections in double precision. Same 3D point
source, infinite extrusion and delay horizon as Planar2D. Geometry is tested
independently of Unity. Refined quadrature does not use sampling PDFs;
--samples tests guided complete edge chains at configurable order.
Equations: https://gamma-web.iacs.umd.edu/BTM/FromRegion.pdf (1-3).
Shared-face multiplicity: EDtoolbox EDwedge2nd, revision 67e53c4f.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre


def visible(a, b, walls):
    """Open interior intersection: touching a rigid face is not crossing it."""
    for wall in walls:
        c, h = np.array(wall['center'][:2]), np.array(wall['half'][:2])
        lo, hi = 0., 1.
        for axis in range(2):
            d = b[axis] - a[axis]
            if abs(d) < 1e-12:
                if not c[axis]-h[axis]+1e-10 < a[axis] < c[axis]+h[axis]-1e-10:
                    hi = -1
                    break
            else:
                t = sorted(((c[axis]-h[axis]-a[axis])/d, (c[axis]+h[axis]-a[axis])/d))
                lo, hi = max(lo, t[0]), min(hi, t[1])
        if lo < hi - 1e-10:
            return False
    return True


def corners(walls):
    result = []
    for owner, wall in enumerate(walls):
        c, h = np.array(wall['center'][:2]), np.array(wall['half'][:2])
        for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            # The solid quadrant lies clockwise from the reference face.
            face = np.array([0., -sy]) if sx == sy else np.array([-sx, 0.])
            normal = np.array([-face[1], face[0]])
            result.append((c + h * [sx, sy], face, normal, owner, 1.5*np.pi))
    return result


def angles(edge, source, receiver):
    xy, face, normal, _, exterior = edge
    def angle(p):
        d = p[:2] - xy
        value = np.arctan2(d @ normal, d @ face) % (2*np.pi)
        return 0. if abs(value-2*np.pi) < 1e-12 else value
    ts, tr = angle(source), angle(receiver)
    if max(ts, tr) > exterior + 1e-10:
        return np.zeros(4)
    return (np.pi/exterior) * (np.pi + np.array([ts+tr, ts-tr, -ts+tr, -ts-tr]))


def kernel(phi, rs, rr, zs, zr, nu=2/3):
    # Independent cylindrical expression, with stable cosh-minus-cos form.
    eta = np.arcsinh(zs/rs) + np.arcsinh(zr/rr)
    sinh = np.sinh(nu*eta/2)
    total = np.zeros(np.broadcast_shapes(np.shape(zs), np.shape(zr)))
    for p in phi:
        if abs(np.sin(p/2)) > 1e-12:
            total += np.sin(p) / (2 * (sinh*sinh + np.sin(p/2)**2))
    return -nu*total / (4*np.pi)


def nodes(low, high, order, k, peaks=()):
    """Phase-resolving panels plus narrow angular peaks; no random samples."""
    if high <= low:
        return np.empty(0), np.empty(0)
    ends = list(np.linspace(low, high, max(2, int(np.ceil((high-low)*max(k, 1)/np.pi))+1)))
    for center, width in peaks:
        # Resolve the peak's tails too: a fixed +/-8-width patch misses mass
        # when the next phase panel is many orders of magnitude wider.
        reach = max(abs(low-center), abs(high-center))
        exponent = max(1, int(np.ceil(np.log2(max(1, reach/width)))))
        offsets = width * 2.**np.arange(-1, exponent+1)
        ends.extend(np.r_[center-offsets, center, center+offsets])
    ends = np.unique(np.clip(ends, low, high))
    x, w = roots_legendre(order)
    half = np.diff(ends)/2
    return ((ends[:-1,None]+ends[1:,None])/2 + half[:,None]*x).ravel(), (half[:,None]*w).ravel()


def widths(phi, scale):
    return [max(1e-8, 3*abs(np.sin(p/2))*scale) for p in phi if abs(np.sin(p/2)) > 1e-12]


def first(edge, source, receiver, k, horizon, order):
    rs, rr = np.linalg.norm(source[:2]-edge[0]), np.linalg.norm(receiver[:2]-edge[0])
    if rs+rr >= horizon:
        return 0j
    phi = angles(edge, source, receiver)
    zmax = np.sqrt((horizon*horizon-(rs+rr)**2)*(horizon*horizon-(rs-rr)**2))/(2*horizon)
    z, w = nodes(-zmax, zmax, order, k, [(0, s) for s in widths(phi, rs*rr/(rs+rr))])
    m, l = np.hypot(rs, z), np.hypot(rr, z)
    return np.sum(w*kernel(phi, rs, rr, -z, -z, np.pi/edge[4])/(m*l)*np.exp(-1j*k*(m+l)))


def second(a, b, source, receiver, k, horizon, order):
    rs, re, rr = [np.linalg.norm(x-y) for x, y in
                  [(source[:2], a[0]), (a[0], b[0]), (b[0], receiver[:2])]]
    if rs+re+rr >= horizon:
        return 0j
    pa, pb = angles(a, source, b[0]), angles(b, a[0], receiver)
    # For fixed z1, the remaining two legs form an ellipse in (XY,z).
    zmax = np.sqrt((horizon*horizon-(rs+re+rr)**2)*(horizon*horizon-(rs-re-rr)**2))/(2*horizon)
    scales = widths(pa, rs*(re+rr)/(rs+re+rr)) + widths(pb, (rs+re)*rr/(rs+re+rr))
    z1, w1 = nodes(-zmax, zmax, order, k, [(0, s) for s in scales])
    total = 0j
    for z, weight in zip(z1, w1):
        m = np.hypot(rs, z)
        budget = horizon-m
        major2, focal2 = budget*budget/4, ((re+rr)**2+z*z)/4
        # Intersect ellipse with the fixed XY position of the second edge.
        d = np.hypot(re+rr, z); ux, uy = (re+rr)/d, -z/d
        minor2 = major2-focal2
        x = (re-rr)/2
        qa = uy*uy/major2 + ux*ux/minor2
        qb = 2*x*ux*uy*(1/major2-1/minor2)
        qc = x*x*(ux*ux/major2+uy*uy/minor2)-1
        disc = max(0, qb*qb-4*qa*qc)
        low, high = z/2+(-qb-np.sqrt(disc))/(2*qa), z/2+(-qb+np.sqrt(disc))/(2*qa)
        peaks = [(z*(1+re/rs), s) for s in widths(pa, re*np.hypot(rs,z)/rs)]
        peaks += [(z*rr/(re+rr), s) for s in widths(pb, re*rr/(re+rr))]
        z2, w2 = nodes(low, high, order, k, peaks)
        n, l = np.hypot(re, z-z2), np.hypot(rr, z2)
        value = kernel(pa, rs, re, -z, z2-z, np.pi/a[4])*kernel(pb, re, rr, z-z2, -z2, np.pi/b[4])/(m*n*l)
        total += weight*np.sum(w2*value*np.exp(-1j*k*(m+n+l)))
    shared_face = a[3] == b[3] and np.any(abs(a[0]-b[0]) < 1e-10)
    return total * (.5 if shared_face else 1)


def evaluate(config, points, hz, order):
    source, walls = np.array(config['source']), config['walls']
    edges = corners(walls)
    k, horizon = 2*np.pi*hz/config['speedOfSound'], config['maximumDelaySeconds']*config['speedOfSound']
    values = np.zeros((len(points), 2), complex)
    for pi, receiver in enumerate(points):
        for i, a in enumerate(edges):
            if not visible(source, a[0], walls):
                continue
            if visible(a[0], receiver, walls):
                values[pi, 0] += first(a, source, receiver, k, horizon, order)
            for j, b in enumerate(edges):
                if i != j and visible(a[0], b[0], walls) and visible(b[0], receiver, walls):
                    values[pi, 1] += second(a, b, source, receiver, k, horizon, order)
    return values


def surfaces(walls):
    result = []
    for wall in walls:
        c, h = np.array(wall['center'][:2]), np.array(wall['half'][:2])
        vertices = [c+h*sign for sign in [(-1,-1), (1,-1), (1,1), (-1,1)]]
        result.extend(zip(vertices, vertices[1:]+vertices[:1]))
    return result


def mirror_path(source, receiver, planes, walls):
    """Unfold planar reflections, then check every physical segment."""
    images, normals = [np.asarray(source[:2])], []
    for a, b in planes:
        d = b-a; normal = np.array([-d[1], d[0]])/np.linalg.norm(d)
        images.append(images[-1]-2*normal*np.dot(images[-1]-a, normal)); normals.append(normal)
    current, path = np.asarray(receiver[:2]), [np.asarray(receiver[:2])]
    for index in range(len(planes)-1, -1, -1):
        a, b = planes[index]; normal = normals[index]; d = images[index+1]-current
        denominator = np.dot(d, normal)
        if abs(denominator) < 1e-12:
            return None
        t = np.dot(a-current, normal)/denominator
        hit = current+t*d; u = np.dot(hit-a, b-a)/np.dot(b-a, b-a)
        if not 1e-10 < t < 1-1e-10 or not 1e-10 < u < 1-1e-10:
            return None
        path.append(hit); current = hit
    path.append(np.asarray(source[:2]))
    if not all(visible(a, b, walls) for a, b in zip(path[:-1], path[1:])):
        return None
    return np.r_[images[-1], 0.], np.linalg.norm(images[-1]-receiver[:2])


def connections(config, points, hz, order, integrate=first):
    """Direct, first/second mirrors, reflection-edge and edge-reflection.

    A restricted offline control for missing path classes, not the GPU sampler.
    Every physical leg is validated; no shortest-route selection or fitted gain.
    """
    source, walls = np.array(config['source']), config['walls']
    planes, edges = surfaces(walls), corners(walls)
    k, horizon = 2*np.pi*hz/config['speedOfSound'], config['maximumDelaySeconds']*config['speedOfSound']
    values = np.zeros((len(points), 4), complex)
    for index, receiver in enumerate(points):
        direct = np.linalg.norm(receiver-source)
        if visible(source, receiver, walls) and direct <= horizon:
            values[index, 0] = np.exp(-1j*k*direct)/direct
        for plane in planes:
            for chain, slot in [([plane], 1)]+[([plane, other], 2) for other in planes]:
                path = mirror_path(source, receiver, chain, walls)
                if path is not None and path[1] <= horizon:
                    values[index, slot] += np.exp(-1j*k*path[1])/path[1]
            for edge in edges:
                for start, end in [(source, receiver), (receiver, source)]:
                    path = mirror_path(start, edge[0], [plane], walls)
                    if path is not None and visible(edge[0], end, walls):
                        values[index, 3] += integrate(edge, path[0], end, k, horizon, order)
    return values


def draw_height(centers, scales, random, low, high):
    lower = np.arctan((low[:,None]-centers)/scales)
    upper = np.arctan((high[:,None]-centers)/scales)
    component = (random[:,0]*scales.shape[1]).astype(int)
    rows = np.arange(len(component))
    selected = lower[rows,component] + (upper-lower)[rows,component]*random[:,1]
    z = centers[rows,component] + scales[rows,component]*np.tan(selected)
    pdf = np.mean(scales/((scales**2+(z[:,None]-centers)**2)*(upper-lower)), axis=1)
    return z, pdf


def sample_chain(edges, source, receiver, k, horizon, random):
    """Guide complete routes with outgoing/preceding angular peaks and exact PDFs.

    XY endpoints have z=0. Heights obey the remaining path-length ellipse;
    z -> -z symmetry permits sampling only positive first heights.
    """
    points = [source[:2]] + [edge[0] for edge in edges] + [receiver[:2]]
    radii = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if radii.sum() >= horizon:
        return 0j
    phi = [angles(edge, points[i], points[i+2]) for i, edge in enumerate(edges)]
    phi = [terms[abs(np.sin(terms/2)) > 1e-12] for terms in phi]
    if any(len(terms) == 0 for terms in phi):
        return 0j
    count = len(random)
    heights, density, length = [np.zeros(count)], np.ones(count), np.zeros(count)
    for i, edge in enumerate(edges):
        rs, rr, previous = radii[i], sum(radii[i+1:]), heights[-1]
        # Minimum remaining length follows the whole XY chain, not a shortcut.
        major = (horizon-length)**2/4
        focal = np.hypot(rs+rr, previous)
        minor = major-focal**2/4
        ux, uy, x = (rs+rr)/focal, -previous/focal, (rs-rr)/2
        qa = uy**2/major + ux**2/minor
        qb = 2*x*ux*uy*(1/major-1/minor)
        qc = x*x*(ux**2/major+uy**2/minor)-1
        root = np.sqrt(np.maximum(0, qb*qb-4*qa*qc))
        low, high = previous/2+(-qb-root)/(2*qa), previous/2+(-qb+root)/(2*qa)
        if i == 0:
            low = np.zeros(count)
        center = previous*rr/(rs+rr)
        m, l = np.hypot(rs, previous-center), np.hypot(rr, center)
        scales = np.maximum(1e-8, 2*abs(np.sin(phi[i]/2))[None,:]*(m*l/(m+l))[:,None]/(np.pi/edge[4]))
        centers = np.broadcast_to(center[:,None], scales.shape)
        if i:
            incoming = previous+(previous-heights[-2])*rs/radii[i-1]
            widths = np.maximum(1e-8, 2*abs(np.sin(phi[i-1]/2))[None,:]*
                                np.hypot(rs, incoming-previous)[:,None]/(np.pi/edges[i-1][4]))
            centers = np.concatenate([centers, np.broadcast_to(incoming[:,None], widths.shape)], axis=1)
            scales = np.concatenate([scales, widths], axis=1)
        z, pdf = draw_height(centers, scales, random[:,i*2:i*2+2], low, high)
        density *= pdf
        length += np.hypot(rs, z-previous)
        heights.append(z)
    heights.append(np.zeros(count))
    length += np.hypot(radii[-1], heights[-2])
    gain = np.ones(count)
    for i, edge in enumerate(edges):
        zs, zr = heights[i]-heights[i+1], heights[i+2]-heights[i+1]
        gain *= kernel(phi[i], radii[i], radii[i+1], zs, zr, np.pi/edge[4])/np.hypot(radii[i], zs)
        if i and edges[i-1][3] == edge[3] and np.any(abs(edges[i-1][0]-edge[0]) < 1e-10):
            gain *= .5
    gain /= np.hypot(radii[-1], heights[-2])
    return 2*np.mean(gain*np.exp(-1j*k*length)/density)


def sampled(config, points, hz, count, seed, depth, *, random=None):
    """Offline planar experiment. Budget is samples per visible route, not rays/bin.

    Reuse a scrambled Sobol net across receiver positions. The edge kernel and
    geometric mirror paths are unchanged. Repeated edges are allowed after a
    different edge; the configured order and physical delay bound the walk.
    Pure edge chains reach depth; mixed mirror/edge connections remain order 2.
    """
    from scipy.stats import qmc
    if count < 1 or count & (count-1) or depth < 1:
        raise ValueError('Samples must be a positive power of two; depth must be positive.')
    if random is None:
        # Center the finite Sobol grid so no sample lies on an ellipse endpoint.
        random = qmc.Sobol(2*depth, scramble=True, bits=30, seed=seed).random_base2(count.bit_length()-1) + .5/2**30
    edges, source, walls = corners(config['walls']), np.array(config['source']), config['walls']
    k = 2*np.pi*hz/config['speedOfSound']
    horizon = config['maximumDelaySeconds']*config['speedOfSound']
    neighbors = [[j for j, b in enumerate(edges) if i != j and visible(a[0], b[0], walls)]
                 for i, a in enumerate(edges)]
    values, routes = np.zeros((len(points), depth), complex), np.zeros((len(points), depth), int)
    def visit(path, receiver, index, distance):
        chain = [edges[i] for i in path]
        if distance + np.linalg.norm(chain[-1][0]-receiver[:2]) >= horizon:
            return
        if visible(chain[-1][0], receiver, walls):
            values[index, len(path)-1] += sample_chain(chain, source, receiver, k, horizon, random)
            routes[index, len(path)-1] += 1
        if len(path) < depth:
            for j in neighbors[path[-1]]:
                visit(path+[j], receiver, index, distance+np.linalg.norm(chain[-1][0]-edges[j][0]))
    for index, receiver in enumerate(points):
        for i, edge in enumerate(edges):
            if visible(source, edge[0], walls):
                visit([i], receiver, index, np.linalg.norm(source[:2]-edge[0]))
    def integrate(edge, start, end, k, horizon, order):
        return sample_chain([edge], start, end, k, horizon, random)
    connected = connections(config, points, hz, 0, integrate)
    return values, connected, routes


def validate():
    from scipy.stats import qmc
    samples = qmc.Sobol(4, scramble=True, bits=30, seed=17).random_base2(16) + .5/2**30
    checks = []
    edge = (np.zeros(2), np.array([0., -1]), np.array([1., 0]), 0, 2*np.pi)
    source, receiver = np.array([-2., 1, 0]), np.array([2., -2, 0])
    # Independent Simpson-integral constants retained in RayTransportChecks.
    for hz, target in [(0, .156429101689+0j), (250, .004442267995+.083451251431j)]:
        value = first(edge, source, receiver, 2*np.pi*hz/343, 34.3, 16)
        error = abs(value-target)
        assert error < 1e-8, ('half-plane', hz, error)
        checks.append(dict(check='half-plane', hz=hz, absoluteError=error))
        estimate = sample_chain([edge], source, receiver, 2*np.pi*hz/343, 34.3, samples)
        error = abs(estimate-value)/abs(value)
        assert error < 1e-4, ('sampled half-plane', hz, error)
        checks.append(dict(check='sampled half-plane', hz=hz, relativeError=float(error)))
    # Near a shadow boundary, separately large GO/edge terms must cancel.
    field = []
    for delta in [-1e-6, 1e-6]:
        probe = np.array([2., -1+delta, 0]); distance = np.linalg.norm(probe-source)
        k = 2*np.pi*250/343
        coarse, fine = [first(edge, source, probe, k, 34.3, n) for n in [8,16]]
        assert abs(coarse-fine) < 1e-9, ('boundary refinement', delta)
        field.append(fine + (np.exp(-1j*k*distance)/distance if delta > 0 else 0))
    assert abs(field[1]-field[0]) < 1e-5, 'False jump at the half-plane shadow boundary.'
    checks.append(dict(check='shadow boundary cancellation', absoluteChange=float(abs(field[1]-field[0]))))
    config = dict(source=[-3., 0, 0], walls=[dict(center=[0, 0, 0], half=[1, 1, 1])],
                  speedOfSound=343, maximumDelaySeconds=.1)
    for hz in [0, 125, 250]:
        coarse = evaluate(config, np.array([[3., 0, 0]]), hz, 8)
        fine = evaluate(config, np.array([[3., 0, 0]]), hz, 16)
        change = np.linalg.norm(fine-coarse)/np.linalg.norm(fine)
        assert change < 1e-5, ('two-edge refinement', hz, change)
        if hz == 0:
            assert abs(fine[0, 1]-.199807) < 2e-5  # Earlier independent tensor quadrature.
        checks.append(dict(check='two-edge refinement', hz=hz, relativeChange=float(change)))
        estimate, _, _ = sampled(config, np.array([[3., 0, 0]]), hz, 65536, 17, 2)
        error = np.linalg.norm(estimate-fine)/np.linalg.norm(fine)
        assert error < .01, ('sampled second order', hz, error)
        checks.append(dict(check='sampled second order', hz=hz, relativeError=float(error)))
    config['source'] = [-3., .4, 0]
    receiver = np.array([[3., -.6, 0]])
    forward = evaluate(config, receiver, 250, 16)
    config['source'], receiver = receiver[0].tolist(), np.array([config['source']])
    reverse = evaluate(config, receiver, 250, 16)
    error = np.linalg.norm(forward-reverse)/np.linalg.norm(forward)
    assert error < 1e-5, ('reciprocity', error)
    checks.append(dict(check='two-edge reciprocity', relativeError=float(error)))
    plane = (np.array([-5., 0]), np.array([5., 0]))
    path = mirror_path(np.array([-1., 1, 0]), np.array([1., 1, 0]), [plane], [])
    assert path is not None and abs(path[1]-np.sqrt(8)) < 1e-12
    checks.append(dict(check='analytic mirror distance', absoluteError=float(abs(path[1]-np.sqrt(8)))))
    planes = [(np.array([x, -10.]), np.array([x, 10.])) for x in [0., 2.]]
    path = mirror_path(np.array([1., 0, 0]), np.array([1., 1, 0]), planes, [])
    assert path is not None and abs(path[1]-np.sqrt(17)) < 1e-12
    checks.append(dict(check='analytic two-mirror distance', absoluteError=float(abs(path[1]-np.sqrt(17)))))
    config['source'] = [-3., -.4, 0]
    config['walls'] = [dict(center=[-.8,-.9,0], half=[.3,1.1,1]),
                       dict(center=[.8,.9,0], half=[.3,1.1,1])]
    receiver = np.array([[-2.4, .7, 0]])
    forward = connections(config, receiver, 250, 16)
    assert abs(forward[0, 3]) > .1, 'Reciprocity must exercise nonzero mixed paths.'
    config['source'], receiver = receiver[0].tolist(), np.array([config['source']])
    reverse = connections(config, receiver, 250, 16)
    error = np.linalg.norm(forward-reverse)/max(np.linalg.norm(forward), 1e-12)
    assert error < 1e-6, ('mixed-path reciprocity', error)
    checks.append(dict(check='mixed-path reciprocity', relativeError=float(error)))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--points', type=int, nargs='+')
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--hz', type=float, default=250)
    parser.add_argument('--samples', type=int, help='Use guided samples per visible edge chain instead of refined quadrature.')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--depth', type=int, default=2, help='Pure edge orders for --samples; mixed paths remain order 2.')
    args = parser.parse_args()
    if args.validate:
        if args.output.exists():
            raise FileExistsError(args.output)
        checks = validate()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(checks=checks, passed=True,
            solverSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
        print(checks)
        return
    if args.scene is None or args.points is None:
        parser.error('Field evaluation requires --scene and --points.')
    if any(args.output.with_suffix(suffix).exists() for suffix in ['.npz', '.json']):
        raise FileExistsError('Preserve earlier measurements; choose a fresh output stem.')
    config = json.loads(args.scene.read_text())
    if (config['physicalModel'] != 'point-extruded' or config.get('materials') or
            any(config.get(key) for key in ['sourceMotion', 'receiverMotion', 'wallMotion'])):
        raise ValueError('Only rigid, stationary, disjoint extruded boxes are supported.')
    points = np.array(config['points']).reshape(-1, 3)[args.points]
    locations = np.vstack([config['source'], points])
    if not np.isfinite(locations).all() or np.any(locations[:, 2] != 0):
        raise ValueError('Source and probes must be finite XY positions.')
    scalars = [args.hz, config['maximumDelaySeconds'], config['speedOfSound']]
    if not np.isfinite(scalars).all() or args.order < 1 or args.hz < 0 or min(scalars[1:]) <= 0:
        raise ValueError('Positive quadrature order, horizon and sound speed are required; frequency may be zero.')
    for i, wall in enumerate(config['walls']):
        c, h = np.array(wall['center'][:2]), np.array(wall['half'][:2])
        if not np.isfinite([c, h]).all() or np.any(h <= 0) or np.any(np.all(abs(locations[:,:2]-c) <= h, axis=1)):
            raise ValueError('Invalid box or source/probe on or inside a solid.')
        for other in config['walls'][:i]:
            if np.all(abs(c-other['center'][:2]) <= h+other['half'][:2]):
                raise ValueError('This diagnostic requires disjoint boxes.')
    if np.any(np.linalg.norm(points-config['source'], axis=1) == 0):
        raise ValueError('Pressure is singular at the source.')
    started = time.perf_counter()
    if args.samples is None:
        values = evaluate(config, points, args.hz, args.order)
        connected = connections(config, points, args.hz, args.order)
        routes = np.empty((len(points), 0), int)
    else:
        values, connected, routes = sampled(config, points, args.hz, args.samples, args.seed, args.depth)
    assert np.isfinite(values).all() and np.isfinite(connected).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), pressure=values, connections=connected, points=points, indices=args.points, routes=routes)
    stats = dict(seconds=time.perf_counter()-started, hz=args.hz, quadratureOrder=args.order,
                 orders=list(range(1, values.shape[1]+1)), sceneSha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(),
                 solverSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 connectionSlots=['direct', 'mirror', 'mirror-mirror', 'mirror-edge + edge-mirror'],
                 scope='Offline rigid planar paths. Mixed reflections reach order 2; not a full-wave reference.')
    if args.samples is not None:
        stats.update(samplesPerRoute=args.samples, seed=args.seed, quadratureOrder=None,
                     edgeKernelEvaluationUpperBound=int(args.samples*(routes*np.arange(1,args.depth+1)).sum()),
                     budgetScope='Pure edge kernel evaluations only; excludes visibility and mixed mirror connections.')
    args.output.with_suffix('.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(stats, flush=True)
    print(values, flush=True)


if __name__ == '__main__':
    main()
