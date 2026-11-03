"""Float64 port of the residual walk's edge terms (Assets/Shaders/Transport.hlsl).

Used to decompose captured GPU walk paths into kernel, distance and proposal
factors, and to test estimators on exact chains. Mirrors EdgeAngle, EdgeKernel,
EdgeConnectionWeight, EdgeHeightSpan/SampleEdge's density and NearestHit for
static triangle scenes; it is a diagnostic, not a renderer.
"""
import numpy as np

EPS = 1e-4
PI = np.pi


class Scene:
    def __init__(self, edges, triangles):
        self.a, self.b = edges['a'].astype(float), edges['b'].astype(float)
        self.face, self.normal = edges['face'].astype(float), edges['normal'].astype(float)
        self.angle = edges['angle'].astype(float)
        self.owner, self.first, self.second = edges['ids'][:, 0], edges['ids'][:, 1], edges['ids'][:, 2]
        self.extent = np.linalg.norm(self.b - self.a, axis=1)
        self.axis = (self.b - self.a) / self.extent[:, None]
        t = triangles
        self.ta, self.tb, self.tc = t[:, 0:3].astype(float), t[:, 3:6].astype(float), t[:, 6:9].astype(float)
        self.adjacent = [[] for _ in range(len(t))]
        for i in range(len(self.a)):
            self.adjacent[self.first[i]].append(i)
            if self.second[i] != self.first[i]:
                self.adjacent[self.second[i]].append(i)
        self.count = len(self.a)

    # ------------------------------------------------------------ geometry
    def nearest_hit(self, o, d, limit, minimum=EPS):
        e1, e2 = self.tb - self.ta, self.tc - self.ta
        h = np.cross(d, e2)
        det = (e1 * h).sum(1)
        ok = np.abs(det) > 8 * 1.1920929e-7 * np.linalg.norm(d) * np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
        det = np.where(ok, det, 1)
        s = o - self.ta
        q = np.cross(s, e1)
        u, v, t = (s * h).sum(1) / det, (q * d).sum(1) / det, (e2 * q).sum(1) / det
        hit = ok & (t > minimum) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t < limit)
        if not hit.any():
            return None
        i = np.where(hit)[0][np.argmin(t[hit])]
        return i, t[i]

    def visible(self, p, q):
        dist = np.linalg.norm(q - p)
        return dist <= 2 * EPS or self.nearest_hit(p, (q - p) / dist, dist - EPS) is None

    # ------------------------------------------------------------ edge terms
    def edge_angle(self, i, direction):
        radial = np.array([direction @ self.face[i], direction @ self.normal[i]])
        angle = np.arctan2(radial[1], radial[0])
        angle = angle + 2 * PI if angle < 0 else angle
        r = np.linalg.norm(radial)
        if min(angle, 2 * PI - angle) * r <= EPS:
            angle = 0.0
        if abs(angle - self.angle[i]) * r <= EPS:
            angle = self.angle[i]
        return angle

    def kernel(self, i, source, point, receiver):
        axis = self.axis[i]
        incoming, outgoing = source - point, receiver - point
        m, l = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        zs, zr = incoming @ axis, outgoing @ axis
        rs, rr = np.linalg.norm(incoming - zs * axis), np.linalg.norm(outgoing - zr * axis)
        if rs <= EPS or rr <= EPS:
            return 0.0
        ts, tr = self.edge_angle(i, incoming), self.edge_angle(i, outgoing)
        if ts > self.angle[i] or tr > self.angle[i]:
            return 0.0
        nu = PI / self.angle[i]
        axial = rs * zr + rr * zs
        x = abs(axial) / np.sqrt(2 * rs * rr * (m * l + rs * rr - zs * zr))
        sinh_half = nu * x * (1 + (nu * nu - 1) * x * x / 6) if x < .01 else np.sinh(nu * np.log(x + np.sqrt(1 + x * x)))
        phi = nu * (PI + np.array([ts + tr, ts - tr, -ts + tr, -ts - tr]))
        half = np.sin(phi / 2)
        beta = sum(np.sin(p) / (2 * (sinh_half ** 2 + h * h)) for p, h in zip(phi, half) if abs(h) > 1e-6)
        return -nu * beta / (4 * PI)

    def connection_weight(self, i, j):
        if self.owner[i] != self.owner[j]:
            return 1.0
        def normals(k):
            return [self.normal[k], np.cos(self.angle[k]) * self.normal[k] - np.sin(self.angle[k]) * self.face[k]]
        for an in normals(i):
            for bn in normals(j):
                if abs(an @ bn) > .99999 and abs(an @ (self.a[j] - self.a[i])) <= EPS:
                    return .5
        return 1.0

    def proposal(self, i, source, receiver):
        """SampleEdge's listener-guided Cauchy mixture on edge i: (apex, scales, spans)."""
        s, r = source - self.a[i], receiver - self.a[i]
        axis = self.axis[i]
        zs, zr = s @ axis, r @ axis
        rs, rr = np.linalg.norm(s - zs * axis), np.linalg.norm(r - zr * axis)
        if rs <= EPS or rr <= EPS:
            return None
        apex = (zs * rr + zr * rs) / (rs + rr)
        m, l = np.linalg.norm(s - apex * axis), np.linalg.norm(r - apex * axis)
        nu = PI / self.angle[i]
        phi = nu * (PI + np.array([1, 1, -1, -1]) * self.edge_angle(i, s) + np.array([1, -1, 1, -1]) * self.edge_angle(i, r))
        scale = np.maximum(EPS, 2 * np.abs(np.sin(phi / 2)) * (m * l / (m + l)) / nu)
        y, xx = self.extent[i] * scale, scale * scale + apex * (apex - self.extent[i])
        ratio = y / np.abs(xx)
        angle = np.where(ratio < .01, ratio * (1 - ratio * ratio / 3), np.arctan2(y, np.abs(xx)))
        span = np.where(xx < 0, PI - angle, angle)
        return apex, scale, span

    def selection(self, i, source, receiver):
        """Probability SampleEdge picks edge i from `source` toward `receiver`."""
        dist = np.linalg.norm(receiver - source)
        local = []
        if dist > EPS:
            hit = self.nearest_hit(source, (receiver - source) / dist, dist - EPS)
            if hit is not None:
                local = self.adjacent[hit[0]]
        p = (.5 if local else 1) / self.count
        p += sum(.5 / len(local) for k in local if k == i)
        return p, local

    def density(self, i, z, source, receiver):
        """Height density (per metre) of SampleEdge on edge i at height z."""
        prop = self.proposal(i, source, receiver)
        if prop is None:
            return 0.0
        apex, scale, span = prop
        delta = z - apex
        return float(np.mean(scale / ((scale * scale + delta * delta) * span)))

    def height(self, i, point):
        return (point - self.a[i]) @ self.axis[i]

    def edge_at(self, point, tolerance=1e-3):
        ab = self.b - self.a
        t = np.clip(((point - self.a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
        dist = np.linalg.norm(self.a + t[:, None] * ab - point, axis=1)
        i = int(np.argmin(dist))
        return i if dist[i] < tolerance else None
