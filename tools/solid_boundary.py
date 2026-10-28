"""Bake the exterior of a rigid group of closed meshes, retaining surface materials.

Offline only. Install tools/solid-boundary-requirements.txt in a Python venv.
Input/output positions use the group's local frame. No voxelization or gap fill.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import manifold3d as m
import numpy as np

VERSION = 1


def checked(mesh, label):
    if mesh.status() != m.Error.NoError or mesh.is_empty() or mesh.volume() <= 0:
        raise ValueError(f'{label}: require a nonempty, consistently oriented closed solid ({mesh.status()}).')
    return mesh


def indexed(body):
    v = np.asarray(body['vertices'], dtype=np.float64).reshape(-1, 3)
    raw = np.asarray(body['triangles'])
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('Triangle indices must be integers.')
    t = raw.reshape(-1, 3)
    material = np.asarray(body['materials'])
    if (len(v) < 4 or len(t) < 4 or not np.isfinite(v).all() or
            t.min() < 0 or t.max() >= len(v) or material.shape != (len(t),) or
            not np.issubdtype(material.dtype, np.integer) or material.min() < 0):
        raise ValueError('Invalid vertices, triangle indices or material slots.')
    # Exact position welding removes rendering/submesh seams, never physical gaps.
    v, inverse = np.unique(v, axis=0, return_inverse=True)
    t = inverse[t]
    xyz = v[t] - v.mean(axis=0)
    volume = np.einsum('ij,ij->i', xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])).sum()/6
    if volume < 0:
        t = t[:, [0, 2, 1]]
    return v, t, material.astype(np.int64)


def from_arrays(v, t, materials, ids):
    order = np.argsort(materials, kind='stable')
    labels = materials[order]
    starts = np.r_[0, np.nonzero(np.diff(labels))[0]+1]
    ends = np.r_[starts, len(t)] * 3
    return m.Manifold(m.Mesh64(np.array(v, dtype=np.float64, order='C', copy=True),
        np.ascontiguousarray(t[order], dtype=np.uint64),
        run_index=np.asarray(ends, dtype=np.uint64),
        run_original_id=np.asarray([ids[int(labels[i])] for i in starts], dtype=np.uint32)))


def arrays(mesh, labels):
    q = mesh.to_mesh64()
    material = np.empty(len(q.tri_verts), dtype=np.int64)
    for i, original in enumerate(q.run_original_id):
        material[q.run_index[i]//3:q.run_index[i+1]//3] = labels[int(original)]
    return np.asarray(q.vert_properties)[:, :3], np.asarray(q.tri_verts), material


def canonical(v, t, materials):
    # Stable vertex and triangle ordering makes reproducible bakes independent
    # of the Boolean backend's allocation/visitation order.
    used, t = np.unique(t, return_inverse=True)
    v, inverse = np.unique(v[used], axis=0, return_inverse=True)
    t = inverse[t.reshape(-1, 3)]
    for i in range(len(t)):
        t[i] = np.roll(t[i], -np.argmin(t[i]))
    order = np.lexsort((t[:, 2], t[:, 1], t[:, 0], materials))
    return v, t[order], materials[order]


def bake(spec, input_hash=None):
    if spec.get('version') != VERSION or not spec.get('bodies'):
        raise ValueError('Require version 1 and at least one closed body.')
    start = time.perf_counter()
    bodies = [indexed(body) for body in spec['bodies']]
    slots = sorted({int(x) for _, _, labels in bodies for x in labels})
    first = m.Manifold.reserve_ids(len(slots))
    ids = {slot: first+i for i, slot in enumerate(slots)}
    labels = {value: key for key, value in ids.items()}
    inputs = [checked(from_arrays(v, t, a, ids), f'Body {i}') for i, (v, t, a) in enumerate(bodies)]
    # Earlier members own coincident exterior coatings. Subtract occupied
    # volume before adding each new member, so ambiguous coplanar overlaps do
    # not depend on the backend's choice of a surviving input triangle.
    union = inputs[0]
    for body in inputs[1:]:
        union = union + (body - union)
    union = checked(union, 'Union')
    # Rebuild face adjacency by material, removing input-object face identities.
    # simplify(0) uses the library's numerical tolerance, not an authored shape
    # approximation. Material runs survive; flat material boundaries stay flat.
    v, t, a = arrays(union, labels)
    boundary = checked(from_arrays(v, t, a, ids).simplify(0), 'Boundary')
    v, t, a = canonical(*arrays(boundary, labels))
    return dict(version=VERSION, inputSha256=input_hash or '',
        backend='manifold3d '+importlib.metadata.version('manifold3d'),
        vertices=v.reshape(-1).tolist(), triangles=t.reshape(-1).tolist(), materials=a.tolist(),
        inputTriangles=sum(len(t) for _, t, _ in bodies), outputTriangles=len(t),
        volume=boundary.volume(), area=boundary.surface_area(), tolerance=boundary.get_tolerance(),
        components=len(boundary.decompose()), seconds=time.perf_counter()-start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path); p.add_argument('output', type=Path)
    args = p.parse_args()
    raw = args.input.read_bytes()
    result = bake(json.loads(raw), hashlib.sha256(raw).hexdigest())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f"Baked {result['inputTriangles']} -> {result['outputTriangles']} triangles; "
          f"{result['components']} component(s), volume {result['volume']:.9g}, area {result['area']:.9g}.")


if __name__ == '__main__':
    main()
