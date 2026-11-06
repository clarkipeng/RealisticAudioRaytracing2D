"""3D time-domain acoustic wave solver (FDTD) for low-frequency teacher data.

Staggered pressure/velocity (Yee) scheme on a uniform Cartesian grid, fused
Metal kernels through PyTorch MPS, axis-aligned boxes that are rigid or locally
reacting with a real (frequency-independent) impedance, convolutional PML
(CPML) on all six domain faces, a band-limited monopole source, and trilinear
off-grid receivers. Complex pressure follows the renderer/reference convention:
time dependence exp(+iwt) and free-field pressure exp(-ikr)/r.

Scene JSON uses the wave-reference format: `source` [x,y,z], `walls`
[{center, half}], `points` (flat xyz list), `speedOfSound`.

Install tools/fdtd-requirements.txt in a Python 3.12 venv on an Apple-silicon Mac.
Heavy runs should use `nice -n 19` and the default 20% GPU work pacing.
"""
import argparse
from dataclasses import dataclass
import json
import math
import os
import time

import numpy as np

import heavy_guard as hg

try:
    import torch
except ImportError:  # Geometry helpers stay importable without PyTorch.
    torch = None

SPEED_OF_SOUND = 343.0
# 99% of the 3D Yee stability limit 1/sqrt(3): stable in float32 and close to
# the dispersion-free diagonal, which lowers the direction-averaged phase error.
COURANT = 0.99 / math.sqrt(3)
GIB = 1024 ** 3

KERNELS = r"""
#include <metal_stdlib>
using namespace metal;

// Fields are (nx, ny, nz) row-major, z fastest. u_a[c] is the scaled velocity
// rho*c*v on the +a face of cell c; the outer +a face is never written (rigid).
// CPML memory lives in slabs of 2L layers per axis: low side then high side.
// solid[c] is 0 for air, else a material id; admit[id] is the wall's normalised
// admittance y = rho*c/Z (0 = rigid). On a face between air and a wall the
// boundary pressure is the mean of the air cell and a ghost cell, and equals
// Z times the time-centred normal velocity into the wall; eliminating the ghost
// gives the face update in face_update (unconditionally stable for y >= 0).

inline uint3 cell_of(uint g, int ny, int nz) {
    return uint3(g / uint(ny * nz), (g / uint(nz)) % uint(ny), g % uint(nz));
}

inline uint slab(int i, int n, int L) { return uint(i < L ? i : i - (n - 2 * L)); }

// Normal velocity on a face between cells with materials (lo, hi), lo on the
// low side. Air-air faces take the ordinary update; wall-wall faces stay zero.
inline float face_update(float u, float d, uchar lo, uchar hi, float plo, float phi,
                         device const float* admit, float lam) {
    if (lo == 0 && hi == 0) return u - lam * d;
    if (lo != 0 && hi != 0) return 0.0f;
    const float y = admit[lo != 0 ? lo : hi];
    if (y <= 0.0f) return 0.0f;
    // Wall on the high side: u points into it, u+ (y+lam) = u- (y-lam) + 2 lam y p_air.
    // Wall on the low side: u points out of it, the sign of the pressure term flips.
    const float drive = hi != 0 ? plo : -phi;
    return ((y - lam) * u + 2.0f * lam * y * drive) / (y + lam);
}

kernel void update_velocity(
    device float* ux [[buffer(0)]], device float* uy [[buffer(1)]], device float* uz [[buffer(2)]],
    device const float* p [[buffer(3)]], device const uchar* solid [[buffer(4)]],
    device float* qx [[buffer(5)]], device float* qy [[buffer(6)]], device float* qz [[buffer(7)]],
    device const float* coef [[buffer(8)]], device const int* dims [[buffer(9)]],
    constant float& lam [[buffer(10)]], device const float* admit [[buffer(11)]],
    uint g [[thread_position_in_grid]])
{
    const int nx = dims[0], ny = dims[1], nz = dims[2], L = dims[3];
    if (g >= uint(nx) * uint(ny) * uint(nz)) return;
    const uint3 c = cell_of(g, ny, nz);
    const int i = c.x, j = c.y, k = c.z;
    const uint sy = nz, sx = uint(ny) * uint(nz);
    const float pc = p[g];
    const uchar here = solid[g];
    // Face coefficients: b then a, per axis, packed [x | y | z].
    device const float* bx = coef; device const float* ax = bx + nx;
    device const float* by = ax + nx; device const float* ay = by + ny;
    device const float* bz = ay + ny; device const float* az = bz + nz;
    if (i < nx - 1) {
        float d = p[g + sx] - pc;
        if (i < L || i >= nx - L) {
            const uint s = slab(i, nx, L) * sx + uint(j) * sy + uint(k);
            const float m = bx[i] * qx[s] + ax[i] * d; qx[s] = m; d += m;
        }
        ux[g] = face_update(ux[g], d, here, solid[g + sx], pc, p[g + sx], admit, lam);
    }
    if (j < ny - 1) {
        float d = p[g + sy] - pc;
        if (j < L || j >= ny - L) {
            const uint s = (uint(i) * uint(2 * L) + slab(j, ny, L)) * uint(nz) + uint(k);
            const float m = by[j] * qy[s] + ay[j] * d; qy[s] = m; d += m;
        }
        uy[g] = face_update(uy[g], d, here, solid[g + sy], pc, p[g + sy], admit, lam);
    }
    if (k < nz - 1) {
        float d = p[g + 1] - pc;
        if (k < L || k >= nz - L) {
            const uint s = (uint(i) * uint(ny) + uint(j)) * uint(2 * L) + slab(k, nz, L);
            const float m = bz[k] * qz[s] + az[k] * d; qz[s] = m; d += m;
        }
        uz[g] = face_update(uz[g], d, here, solid[g + 1], pc, p[g + 1], admit, lam);
    }
}

kernel void update_pressure(
    device float* p [[buffer(0)]],
    device const float* ux [[buffer(1)]], device const float* uy [[buffer(2)]], device const float* uz [[buffer(3)]],
    device const uchar* solid [[buffer(4)]],
    device float* qx [[buffer(5)]], device float* qy [[buffer(6)]], device float* qz [[buffer(7)]],
    device const float* coef [[buffer(8)]], device const int* dims [[buffer(9)]],
    constant float& lam [[buffer(10)]], uint g [[thread_position_in_grid]])
{
    const int nx = dims[0], ny = dims[1], nz = dims[2], L = dims[3];
    if (g >= uint(nx) * uint(ny) * uint(nz) || solid[g] != 0) return;
    const uint3 c = cell_of(g, ny, nz);
    const int i = c.x, j = c.y, k = c.z;
    const uint sy = nz, sx = uint(ny) * uint(nz);
    device const float* bx = coef; device const float* ax = bx + nx;
    device const float* by = ax + nx; device const float* ay = by + ny;
    device const float* bz = ay + ny; device const float* az = bz + nz;
    float dx = ux[g] - (i > 0 ? ux[g - sx] : 0.0f);
    float dy = uy[g] - (j > 0 ? uy[g - sy] : 0.0f);
    float dz = uz[g] - (k > 0 ? uz[g - 1] : 0.0f);
    if (i < L || i >= nx - L) {
        const uint s = slab(i, nx, L) * sx + uint(j) * sy + uint(k);
        const float m = bx[i] * qx[s] + ax[i] * dx; qx[s] = m; dx += m;
    }
    if (j < L || j >= ny - L) {
        const uint s = (uint(i) * uint(2 * L) + slab(j, ny, L)) * uint(nz) + uint(k);
        const float m = by[j] * qy[s] + ay[j] * dy; qy[s] = m; dy += m;
    }
    if (k < L || k >= nz - L) {
        const uint s = (uint(i) * uint(ny) + uint(j)) * uint(2 * L) + slab(k, nz, L);
        const float m = bz[k] * qz[s] + az[k] * dz; qz[s] = m; dz += m;
    }
    p[g] -= lam * (dx + dy + dz);
}
"""
_LIBRARY = None


def _library():
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = torch.mps.compile_shader(KERNELS)
    return _LIBRARY


@dataclass(frozen=True)
class Grid:
    """Uniform grid; cell (i,j,k) spans origin + [i,i+1]*spacing per axis."""
    spacing: float
    origin: tuple
    shape: tuple
    pml: int

    @property
    def cells(self):
        return int(np.prod(self.shape))

    def centers(self, axis):
        return self.origin[axis] + (np.arange(self.shape[axis]) + .5) * self.spacing


def make_grid(low, high, spacing, pml):
    """Smallest lattice-aligned grid containing [low, high], plus PML layers.

    Faces sit on integer multiples of `spacing`, so boxes whose faces are
    multiples of it are represented exactly.
    """
    low, high = np.asarray(low, float), np.asarray(high, float)
    first = np.floor(low / spacing + 1e-9).astype(int)
    last = np.ceil(high / spacing - 1e-9).astype(int)
    shape = tuple(int(v) for v in last - first + 2 * pml)
    origin = tuple(float(v) for v in (first - pml) * spacing)
    return Grid(float(spacing), origin, shape, int(pml))


def box_bounds(walls):
    low = np.array([np.subtract(w['center'], w['half']) for w in walls], float).reshape(-1, 3)
    high = np.array([np.add(w['center'], w['half']) for w in walls], float).reshape(-1, 3)
    return low, high


def voxelize(grid, walls, ids=None):
    """Material ids (cell centre inside a box; 0 = air) and the largest face snap in metres.

    `ids` gives each wall's material id (1-255, default 1); a later box wins
    where boxes overlap.
    """
    solid = np.zeros(grid.shape, dtype=np.uint8)
    snap = 0.0
    ids = [1] * len(walls) if ids is None else list(ids)
    if any(not 1 <= v <= 255 for v in ids):
        raise ValueError('Material ids must lie in 1..255.')
    for wall_id, low, high in zip(ids, *box_bounds(walls)):
        ranges = []
        for axis in range(3):
            c = grid.centers(axis)
            inside = np.flatnonzero((c > low[axis]) & (c < high[axis]))
            if inside.size == 0:
                ranges = None
                break
            ranges.append(slice(inside[0], inside[-1] + 1))
            faces = grid.origin[axis] + np.array([inside[0], inside[-1] + 1]) * grid.spacing
            snap = max(snap, float(np.max(np.abs(faces - [low[axis], high[axis]]))))
        if ranges is None:
            snap = max(snap, float(np.min(high - low)))  # Box thinner than a cell vanished.
            continue
        solid[tuple(ranges)] = wall_id
    return solid, snap


def admittance_from_absorption(alpha, incidence='random'):
    """Normalised real admittance y = rho*c/Z of a locally reacting wall.

    `random`: the Paris (London) statistical absorption of a locally reacting
    surface with real impedance zeta = 1/y,
        alpha_st = 8/zeta^2 * (zeta + 1 - 1/(1+zeta) - 2 ln(1+zeta)),
    equals `alpha` (the diffuse-field reading of a tabulated coefficient). It
    peaks at 0.951 for zeta = 1.567; the hard branch zeta >= 1.567 is used.
    `normal`: normal-incidence absorption 4 zeta/(zeta+1)^2 equals `alpha`.
    alpha = 0 returns 0 (rigid).
    """
    alpha = float(alpha)
    if alpha <= 0:
        return 0.0
    if incidence == 'normal':
        if alpha > 1:
            raise ValueError('Absorption must lie in [0, 1].')
        r = math.sqrt(1 - alpha)
        return (1 - r) / (1 + r)
    if incidence != 'random':
        raise ValueError(incidence)
    if alpha > STATISTICAL_ABSORPTION_MAX:
        raise ValueError(f'Statistical absorption {alpha} exceeds {STATISTICAL_ABSORPTION_MAX:.4f} for a real impedance.')
    lo, hi = math.log(ZETA_AT_MAX), math.log(1e9)   # alpha_st falls monotonically on this branch
    for _ in range(200):
        mid = .5 * (lo + hi)
        if statistical_absorption(math.exp(mid)) > alpha:
            lo = mid
        else:
            hi = mid
    return 1 / math.exp(.5 * (lo + hi))


def statistical_absorption(zeta):
    """Paris random-incidence absorption of a real normalised impedance zeta."""
    z = float(zeta)
    return 8 / z ** 2 * (z + 1 - 1 / (1 + z) - 2 * math.log1p(z))


def reflection_coefficient(zeta, theta):
    """Plane-wave pressure reflection of a locally reacting wall at incidence theta."""
    c = np.cos(theta)
    return (zeta * c - 1) / (zeta * c + 1)


ZETA_AT_MAX = 1.56692
STATISTICAL_ABSORPTION_MAX = 0.95122


def pml_coefficients(grid, dt, c=SPEED_OF_SOUND, reflection=1e-6, order=3, alpha=0.0):
    """CPML recursion coefficients (b, a) at velocity faces and cell centres.

    sigma(d) = sigma_max (d/D)^order with sigma_max from the target normal
    reflection of the continuous layer; kappa = 1, complex-frequency shift alpha.
    Recursion: psi = b psi + a * difference; effective difference += psi.
    """
    L, h = grid.pml, grid.spacing
    faces, cells = [], []
    for n in grid.shape:
        out = []
        for offset in (1.0, .5):  # Face i is at index i+1, centre i at i+1/2.
            x = np.arange(n) + offset
            depth = np.maximum(np.maximum(L - x, x - (n - L)), 0) * h if L else np.zeros(n)
            if L:
                sigma_max = -(order + 1) * c * math.log(reflection) / (2 * L * h)
                sigma = sigma_max * (depth / (L * h)) ** order
            else:
                sigma = np.zeros(n)
            b = np.exp(-(sigma + alpha) * dt)
            a = np.divide(sigma * (b - 1), sigma + alpha, out=np.zeros(n), where=sigma > 0)
            out.append((b, a))
        faces.append(out[0])
        cells.append(out[1])
    pack = lambda parts: np.concatenate([np.concatenate(p) for p in parts]).astype(np.float32)
    return pack(faces), pack(cells)


def trilinear_stencil(grid, solid, points):
    """Flat indices (m, 8) and weights of cell centres around each point.

    Solid corners get zero weight and the rest are renormalised, which matches a
    zero-gradient (rigid) ghost value across a face. Used for receivers and, as
    its adjoint, for spreading an off-grid source.
    """
    points = np.atleast_2d(np.asarray(points, float))
    u = (points - np.array(grid.origin)) / grid.spacing - .5
    base = np.floor(u).astype(int)
    frac = u - base
    shape = np.array(grid.shape)
    if np.any(base < 0) or np.any(base + 1 >= shape):
        raise ValueError('Point outside the interpolation domain.')
    index = np.zeros((len(points), 8), dtype=np.int64)
    weight = np.zeros((len(points), 8))
    for n, corner in enumerate(np.ndindex(2, 2, 2)):
        corner = np.array(corner)
        ijk = base + corner
        index[:, n] = np.ravel_multi_index(ijk.T, grid.shape)
        weight[:, n] = np.prod(np.where(corner, frac, 1 - frac), axis=1)
    air = solid.ravel()[index] == 0
    weight *= air
    total = weight.sum(1)
    if np.any(total < 1e-6):
        raise ValueError('Point has no air cell in its interpolation stencil.')
    return index, weight / total[:, None]


def pulse(t, peak_hz):
    """Zero-mean derivative-of-Gaussian volume velocity (m^3/s).

    |Q(f)| peaks at `peak_hz`, so no net volume is injected. The amplitude gives
    roughly unit peak pressure at 1 m in free field.
    """
    tau = 1 / (2 * math.pi * peak_hz)
    s = (t - 7 * tau) / tau
    return 4 * math.pi * tau * (-s) * np.exp(-.5 * s * s)


def dft(series, times, hz):
    """sum x(t) exp(-i w t) dt over the leading axis, for each frequency (float64)."""
    w = 2 * math.pi * np.atleast_1d(hz)
    phase = np.exp(-1j * np.outer(w, times))
    return phase @ np.asarray(series, np.float64) * (times[1] - times[0])


def memory_bytes(grid):
    L = grid.pml
    nx, ny, nz = grid.shape
    slabs = 2 * L * (ny * nz + nx * nz + nx * ny) if L else 0
    return grid.cells * (4 * 4 + 1) + 2 * slabs * 4


def process_footprint():
    return hg.footprint(os.getpid())  # includes MPS allocations, unlike RSS


class Simulation:
    """One FDTD domain on the GPU. Pressure p^n lives at t = n dt."""

    def __init__(self, grid, solid, c=SPEED_OF_SOUND, courant=COURANT, reflection=1e-6,
                 order=3, alpha=0.0, memory_limit=4 * GIB, admittance=None):
        if torch is None or not torch.backends.mps.is_available():
            raise RuntimeError('PyTorch with the MPS backend is required.')
        need = memory_bytes(grid)
        if need > memory_limit:
            raise MemoryError(f'Grid needs {need/GIB:.2f} GiB, above the {memory_limit/GIB:.2f} GiB guard.')
        if np.any(np.array(grid.shape) <= 2 * grid.pml + 1):
            raise ValueError('PML layers leave no interior.')
        self.grid, self.c, self.courant = grid, c, courant
        self.dt = courant * grid.spacing / c
        self.memory_limit = memory_limit
        dev = torch.device('mps')
        self.dev = dev
        nx, ny, nz = grid.shape
        L = grid.pml
        self.p, self.ux, self.uy, self.uz = (torch.zeros(grid.cells, device=dev) for _ in range(4))
        self.solid = torch.from_numpy(np.ascontiguousarray(solid.ravel())).to(dev)
        size = lambda n: max(n, 1)
        self.velocity_memory = [torch.zeros(size(2 * L * ny * nz), device=dev),
                                torch.zeros(size(2 * L * nx * nz), device=dev),
                                torch.zeros(size(2 * L * nx * ny), device=dev)]
        self.pressure_memory = [torch.zeros_like(m) for m in self.velocity_memory]
        faces, cells = pml_coefficients(grid, self.dt, c, reflection, order, alpha)
        self.face_coef = torch.from_numpy(faces).to(dev)
        self.cell_coef = torch.from_numpy(cells).to(dev)
        self.dims = torch.tensor([nx, ny, nz, L], dtype=torch.int32, device=dev)
        # admittance: material id -> normalised admittance (missing ids are rigid).
        table = np.zeros(256, np.float32)
        for key, value in (admittance or {}).items():
            if not 1 <= int(key) <= 255 or not value >= 0:
                raise ValueError(f'Bad admittance {key}: {value}')
            table[int(key)] = value
        self.admittance = table
        self.admit = torch.from_numpy(table).to(dev)
        self.lib = _library()
        self.steps = 0

    def step(self):
        lib, n = _library(), self.grid.cells
        lib.update_velocity(self.ux, self.uy, self.uz, self.p, self.solid, *self.velocity_memory,
                            self.face_coef, self.dims, float(self.courant), self.admit, threads=n, group_size=256)
        lib.update_pressure(self.p, self.ux, self.uy, self.uz, self.solid, *self.pressure_memory,
                            self.cell_coef, self.dims, float(self.courant), threads=n, group_size=256)
        self.steps += 1

    def run(self, source, receivers, duration, peak_hz, plane=None, plane_hz=(),
            work_fraction=hg.WORK_FRACTION, batch_seconds=.1, progress=None, pause=None):
        """Run under the machine-wide heavy-job lock so FDTD and Unity jobs never overlap."""
        with hg.HeavyLock():
            return self._run(source, receivers, duration, peak_hz, plane, plane_hz, work_fraction, batch_seconds, progress, pause)

    def _run(self, source, receivers, duration, peak_hz, plane=None, plane_hz=(),
            work_fraction=hg.WORK_FRACTION, batch_seconds=.1, progress=None, pause=None):
        """Inject a monopole pulse at `source` and record pressure at `receivers`.

        Returns time series p^n (n = 0..N) at the receivers, the source signal
        q^(n+1/2), and optional running DFTs on an axis-aligned plane
        `plane=(axis, value)`. Only `work_fraction` of wall time is GPU work;
        `pause()`, if given, may block between batches (e.g. on battery power).
        """
        grid, dt, dev = self.grid, self.dt, self.dev
        steps = int(math.ceil(duration / dt))
        solid = self.solid.cpu().numpy().reshape(grid.shape)
        s_index, s_weight = trilinear_stencil(grid, np.zeros_like(solid), [source])
        if np.any(solid.ravel()[s_index] != 0):
            raise ValueError('Source stencil touches a solid cell.')
        r_index, r_weight = trilinear_stencil(grid, solid, receivers)
        q = pulse((np.arange(steps) + .5) * dt, peak_hz)
        # dp/dt = -rho c^2 div v + rho c^2 q delta, rho = 1 (cancels in H).
        drive = torch.from_numpy((self.c ** 2 * dt / grid.spacing ** 3 * q).astype(np.float32)).to(dev)
        s_index = torch.from_numpy(s_index[0]).to(dev)
        s_weight = torch.from_numpy(s_weight[0].astype(np.float32)).to(dev)
        r_index = torch.from_numpy(r_index).to(dev)
        r_weight = torch.from_numpy(r_weight.astype(np.float32)).to(dev)
        record = torch.zeros((steps + 1, len(r_index)), device=dev)
        planes = None
        if plane is not None:
            axis, value = plane
            u = (value - grid.origin[axis]) / grid.spacing - .5
            j0 = int(math.floor(u))
            frac = float(u - j0)
            view = self.p.view(grid.shape)
            layer = lambda j: view.select(axis, j)
            w = 2 * math.pi * np.asarray(plane_hz, float)
            planes = torch.zeros((len(w), 2) + tuple(np.delete(grid.shape, axis)), device=dev)
        timing = dict(gpuSeconds=0.0, wallSeconds=0.0, peakDriverBytes=0, peakFootprintBytes=0)
        watch = hg.SystemWatch()
        start = time.time()  # Wall clock; perf_counter stops while the system sleeps.
        n = 0
        while n < steps:
            t0 = time.perf_counter()
            batch_end = steps
            while n < batch_end:
                self.step()
                self.p.index_add_(0, s_index, s_weight * drive[n])
                record[n + 1] = (self.p[r_index] * r_weight).sum(1)
                if planes is not None:
                    slab = (1 - frac) * layer(j0) + frac * layer(j0 + 1)
                    t = (n + 1) * dt
                    for f in range(len(w)):
                        planes[f, 0].add_(slab, alpha=math.cos(w[f] * t) * dt)
                        planes[f, 1].add_(slab, alpha=-math.sin(w[f] * t) * dt)
                n += 1
                if (n & 15) == 0:
                    torch.mps.synchronize()
                    if time.perf_counter() - t0 > batch_seconds:
                        break
            torch.mps.synchronize()
            busy = time.perf_counter() - t0
            timing['gpuSeconds'] += busy
            used = max(torch.mps.driver_allocated_memory(), process_footprint())
            timing['peakDriverBytes'] = max(timing['peakDriverBytes'], torch.mps.driver_allocated_memory())
            timing['peakFootprintBytes'] = max(timing['peakFootprintBytes'], used)
            if used > self.memory_limit:
                raise MemoryError(f'Memory {used/GIB:.2f} GiB exceeded the guard.')
            if problem := watch.problem():
                raise MemoryError(f'Stopped FDTD: {problem}.')
            if progress:
                progress(n, steps)
            if work_fraction < 1 and n < steps:
                time.sleep(busy * (1 - work_fraction) / work_fraction)
            if pause and n < steps:
                pause()
        timing['wallSeconds'] = time.time() - start
        timing['steps'] = steps
        timing['gcellPerSecond'] = grid.cells * steps / timing['gpuSeconds'] / 1e9
        out = dict(pressure=record.cpu().numpy(), source=q, dt=dt, timing=timing)
        if planes is not None:
            out['plane'] = (planes[:, 0] + 1j * planes[:, 1]).cpu().numpy()
        return out


def transfer(pressure, q, dt, hz, c=SPEED_OF_SOUND, rho=1.0):
    """Complex response normalised so free-field pressure is exp(-ikr)/r.

    Continuous theory: P = rho i w Q exp(-ikr) / (4 pi r). The scheme's central
    time difference acts as i w~ with w~ = 2 sin(w dt/2)/dt, used here so the
    remaining error is the spatial/temporal dispersion of the grid alone.
    Pressure samples sit at n dt, source samples at (n+1/2) dt.
    """
    hz = np.atleast_1d(hz)
    P = dft(pressure, np.arange(len(pressure)) * dt, hz)
    Q = dft(q, (np.arange(len(q)) + .5) * dt, hz)
    w_disc = 2 * np.sin(np.pi * hz * dt) / dt
    return 4 * np.pi * P / (rho * 1j * w_disc * Q)[:, None]


def transfer_curve(pressure, q, dt, hz, times, chunk=4096):
    """`transfer` truncated at each time in `times` (one pass, bounded memory)."""
    w = 2 * math.pi * hz
    ends = np.round(np.asarray(times) / dt).astype(int)
    P = np.zeros((len(ends), pressure.shape[1]), complex)
    total = np.zeros(pressure.shape[1], complex)
    k = 0
    for first in range(0, ends[-1] + 1, chunk):
        last = min(first + chunk, ends[-1] + 1)
        n = np.arange(first, last)
        weighted = np.exp(-1j * w * n * dt)[:, None] * pressure[first:last].astype(np.float64)
        partial = np.cumsum(weighted, axis=0) + total
        while k < len(ends) and ends[k] < last:
            P[k] = partial[ends[k] - first]
            k += 1
        total = partial[-1]
    Qn = np.cumsum(q * np.exp(-1j * w * (np.arange(len(q)) + .5) * dt))
    Q = Qn[ends - 1]
    w_disc = 2 * math.sin(w * dt / 2) / dt
    return 4 * np.pi * P / (1j * w_disc * Q)[:, None]


def scene_bounds(scene, margin, free_field=False):
    points = np.array(scene['points'], float).reshape(-1, 3)
    parts = [points, np.atleast_2d(scene['source'])]
    if not free_field and scene['walls']:
        low, high = box_bounds(scene['walls'])
        parts += [low, high]
    every = np.vstack(parts)
    return every.min(0) - margin, every.max(0) + margin


def material_ids(wall_admittance):
    """Per-wall material ids (1..255) and the id -> admittance table."""
    values = sorted(set(float(v) for v in wall_admittance))
    if len(values) > 255:
        raise ValueError('At most 255 distinct wall admittances.')
    table = {n + 1: v for n, v in enumerate(values)}
    lookup = {v: n for n, v in table.items()}
    return [lookup[float(v)] for v in wall_admittance], table


def solve(scene, spacing, hz, duration, peak_hz=None, pml=16, margin=.5, reflection=1e-6,
          order=3, alpha=0.0, free_field=False, plane=None, work_fraction=hg.WORK_FRACTION,
          memory_limit=4 * GIB, progress=None, pause=None, wall_admittance=None, extra_points=None):
    """Complex pressure at the scene's receivers, normalised to exp(-ikr)/r.

    `wall_admittance` gives each wall's normalised admittance rho*c/Z (0 or
    None = rigid). `extra_points` are recorded after the scene's receivers and
    widen the domain (used for image points in the reflection check).
    """
    c = float(scene.get('speedOfSound', SPEED_OF_SOUND))
    hz = np.atleast_1d(hz).astype(float)
    peak_hz = float(peak_hz or hz.max() * 1.2)
    points = np.array(scene['points'], float).reshape(-1, 3)
    if extra_points is not None:
        points = np.vstack([points, np.asarray(extra_points, float).reshape(-1, 3)])
    low, high = scene_bounds(dict(scene, points=points.ravel().tolist()), margin, free_field)
    grid = make_grid(low, high, spacing, pml)
    walls = [] if free_field else scene['walls']
    admit = [0.0] * len(walls) if wall_admittance is None or free_field else \
        [float(v or 0.0) for v in wall_admittance]
    if len(admit) != len(walls):
        raise ValueError('One admittance per wall is required.')
    ids, table = material_ids(admit)
    solid, snap = voxelize(grid, walls, ids)
    sim = Simulation(grid, solid, c=c, reflection=reflection, order=order, alpha=alpha,
                     memory_limit=memory_limit, admittance=table)
    result = sim.run(scene['source'], points, duration, peak_hz, plane=plane, plane_hz=hz,
                     work_fraction=work_fraction, progress=progress, pause=pause)
    H = transfer(result['pressure'], result['source'], result['dt'], hz, c)
    meta = dict(spacing=spacing, hz=hz.tolist(), duration=duration, peakHz=peak_hz, pml=pml,
                margin=margin, reflection=reflection, order=order, alpha=alpha, freeField=free_field,
                shape=list(grid.shape), origin=list(grid.origin), cells=grid.cells, dt=result['dt'],
                courant=sim.courant, speedOfSound=c, maxBoxSnap=snap, solidCells=int(solid.sum()),
                pointsPerWavelength=c / hz.max() / spacing, timing=result['timing'],
                admittances=sorted(set(admit)))
    out = dict(H=H, pressure=result['pressure'], source=result['source'], meta=meta, grid=grid)
    if plane is not None:
        out['plane'] = result['plane']
    return out


def free_field_pressure(points, source, hz, c=SPEED_OF_SOUND):
    r = np.linalg.norm(np.asarray(points, float).reshape(-1, 3) - source, axis=1)
    return np.exp(-1j * 2 * np.pi * hz / c * r) / r


def relative_l2(actual, target, mask=None):
    if mask is not None:
        actual, target = actual[mask], target[mask]
    return float(np.linalg.norm(actual - target) / np.linalg.norm(target))


def benchmark(sizes=(128, 192, 256, 320), steps=200, pml=16):
    """Throughput (Gcell/s) of the fused kernels on cubic grids with CPML."""
    rows = []
    for n in sizes:
        grid = Grid(.05, (0., 0., 0.), (n, n, n), pml)
        sim = Simulation(grid, np.zeros(grid.shape, np.uint8))
        for _ in range(10):
            sim.step()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            sim.step()
        torch.mps.synchronize()
        seconds = time.perf_counter() - t0
        rows.append(dict(n=n, cells=grid.cells, steps=steps, seconds=seconds,
                         gcellPerSecond=grid.cells * steps / seconds / 1e9))
        del sim
        torch.mps.empty_cache()
        time.sleep(4 * seconds)  # 20% duty between sizes.
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='Solve one scene and write an .npz')
    r.add_argument('scene')
    r.add_argument('output')
    r.add_argument('--spacing', type=float, required=True)
    r.add_argument('--hz', type=float, nargs='+', required=True)
    r.add_argument('--duration', type=float, required=True, help='Simulated seconds')
    r.add_argument('--peak-hz', type=float)
    r.add_argument('--pml', type=int, default=16)
    r.add_argument('--margin', type=float, default=.5)
    r.add_argument('--reflection', type=float, default=1e-6)
    r.add_argument('--free-field', action='store_true', help='Ignore walls (analytic control)')
    r.add_argument('--plane', nargs=2, type=float, metavar=('AXIS', 'VALUE'))
    r.add_argument('--work-fraction', type=float, default=hg.WORK_FRACTION)
    r.add_argument('--memory-gib', type=float, default=4)
    b = sub.add_parser('bench', help='Measure throughput')
    b.add_argument('--sizes', type=int, nargs='+', default=[128, 192, 256, 320])
    a = p.parse_args()
    if a.command == 'bench':
        print(json.dumps(benchmark(a.sizes), indent=2))
        return
    scene = json.loads(open(a.scene).read())
    plane = (int(a.plane[0]), a.plane[1]) if a.plane else None
    last = [0.0]

    def progress(n, total):
        if time.time() - last[0] > 30:
            last[0] = time.time()
            print(f'{n}/{total} steps', flush=True)
    out = solve(scene, a.spacing, a.hz, a.duration, a.peak_hz, a.pml, a.margin, a.reflection,
                free_field=a.free_field, plane=plane, work_fraction=a.work_fraction,
                memory_limit=a.memory_gib * GIB, progress=progress)
    extra = dict(plane=out['plane']) if 'plane' in out else {}
    np.savez_compressed(a.output, H=out['H'], pressure=out['pressure'].astype(np.float32),
                        source=out['source'], points=np.array(scene['points']).reshape(-1, 3),
                        meta=json.dumps(out['meta']), **extra)
    print(json.dumps(out['meta'], indent=2))


if __name__ == '__main__':
    main()
