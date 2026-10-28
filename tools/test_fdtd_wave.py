import math
import unittest

import numpy as np

import fdtd_wave as fw

MPS = fw.torch is not None and fw.torch.backends.mps.is_available()


def reference_step(state, solid, faces, cells, lam, L):
    """Plain NumPy transcription of the scheme, for checking the Metal kernels."""
    p, u, qv, qp = state['p'], state['u'], state['qv'], state['qp']
    shape = p.shape
    for axis in range(3):
        n = shape[axis]
        d = np.zeros(shape)
        hi = [slice(None)] * 3; lo = [slice(None)] * 3
        hi[axis] = slice(1, None); lo[axis] = slice(0, -1)
        d[tuple(lo)] = p[tuple(hi)] - p[tuple(lo)]
        b, a = faces[axis]
        expand = [None] * 3; expand[axis] = slice(None)
        qv[axis] = b[tuple(expand)] * qv[axis] + a[tuple(expand)] * d
        d = d + qv[axis]
        blocked = np.zeros(shape, bool)
        blocked[tuple(lo)] = (solid[tuple(lo)] | solid[tuple(hi)]) != 0
        last = [slice(None)] * 3; last[axis] = n - 1
        new = np.where(blocked, 0, u[axis] - lam * d)
        new[tuple(last)] = 0
        u[axis] = new
    div = np.zeros(shape)
    for axis in range(3):
        d = u[axis].copy()
        hi = [slice(None)] * 3; lo = [slice(None)] * 3
        hi[axis] = slice(1, None); lo[axis] = slice(0, -1)
        d[tuple(hi)] -= u[axis][tuple(lo)]
        b, a = cells[axis]
        expand = [None] * 3; expand[axis] = slice(None)
        qp[axis] = b[tuple(expand)] * qp[axis] + a[tuple(expand)] * d
        div += d + qp[axis]
    state['p'] = np.where(solid != 0, p, p - lam * div)


def split(packed, shape):
    out, at = [], 0
    for n in shape:
        out.append((packed[at:at + n].astype(float), packed[at + n:at + 2 * n].astype(float)))
        at += 2 * n
    return out


class Geometry(unittest.TestCase):
    def test_lattice_boxes_are_exact(self):
        grid = fw.make_grid([-1, -1, -1], [1, 1, 1], .05, 4)
        solid, snap = fw.voxelize(grid, [dict(center=[0, 0, 0], half=[.25, .5, .9])])
        self.assertLess(snap, 1e-9)
        self.assertEqual(int(solid.sum()), 10 * 20 * 36)

    def test_off_lattice_box_reports_snap(self):
        grid = fw.make_grid([-1, -1, -1], [1, 1, 1], .1, 0)
        _, snap = fw.voxelize(grid, [dict(center=[0, 0, 0], half=[.27, .3, .3])])
        self.assertAlmostEqual(snap, .03, places=9)

    def test_trilinear_exact_for_linear_field(self):
        grid = fw.make_grid([0, 0, 0], [1, 1, 1], .1, 0)
        solid = np.zeros(grid.shape, np.uint8)
        c = np.meshgrid(*[grid.centers(a) for a in range(3)], indexing='ij')
        field = 2 * c[0] - 3 * c[1] + .5 * c[2] + 1
        pts = np.random.default_rng(1).uniform(.1, .9, (50, 3))
        idx, w = fw.trilinear_stencil(grid, solid, pts)
        got = (field.ravel()[idx] * w).sum(1)
        np.testing.assert_allclose(got, 2 * pts[:, 0] - 3 * pts[:, 1] + .5 * pts[:, 2] + 1, atol=1e-12)

    def test_solid_corners_are_excluded(self):
        grid = fw.make_grid([0, 0, 0], [1, 1, 1], .1, 0)
        solid = np.zeros(grid.shape, np.uint8)
        solid[5:] = 1
        idx, w = fw.trilinear_stencil(grid, solid, [[.5, .5, .5]])
        self.assertTrue(np.all(w[solid.ravel()[idx] != 0] == 0))
        self.assertAlmostEqual(w.sum(), 1.0)

    def test_truncation_curve_matches_transfer(self):
        rng = np.random.default_rng(2)
        dt, pressure, q = 1e-4, rng.standard_normal((900, 3)), rng.standard_normal(899)
        times = [.02, .051, .0899]
        curve = fw.transfer_curve(pressure, q, dt, 125.0, times, chunk=128)
        for t, row in zip(times, curve):
            n = int(round(t / dt))
            np.testing.assert_allclose(row, fw.transfer(pressure[:n + 1], q[:n], dt, 125.0)[0], rtol=1e-10)

    def test_pulse_has_no_net_volume(self):
        t = np.arange(0, .1, 1e-5)
        self.assertLess(abs(fw.pulse(t, 150).sum()) * 1e-5, 1e-9)


@unittest.skipUnless(MPS, 'needs PyTorch MPS')
class Kernels(unittest.TestCase):
    def test_metal_matches_reference(self):
        grid = fw.Grid(.1, (0., 0., 0.), (14, 11, 9), 3)
        solid = np.zeros(grid.shape, np.uint8)
        solid[6:8, 4:9, 2:5] = 1
        sim = fw.Simulation(grid, solid)
        rng = np.random.default_rng(0)
        p0 = rng.standard_normal(grid.shape) * (solid == 0)
        sim.p.copy_(fw.torch.from_numpy(p0.ravel().astype(np.float32)))
        faces, cells = (split(c, grid.shape) for c in fw.pml_coefficients(grid, sim.dt))
        state = dict(p=p0.astype(float), u=[np.zeros(grid.shape) for _ in range(3)],
                     qv=[np.zeros(grid.shape) for _ in range(3)], qp=[np.zeros(grid.shape) for _ in range(3)])
        for _ in range(25):
            sim.step()
            reference_step(state, solid, faces, cells, sim.courant, grid.pml)
        got = sim.p.cpu().numpy().reshape(grid.shape)
        np.testing.assert_allclose(got, state['p'], atol=2e-5 * np.abs(state['p']).max())
        for axis, u in enumerate([sim.ux, sim.uy, sim.uz]):
            np.testing.assert_allclose(u.cpu().numpy().reshape(grid.shape), state['u'][axis], atol=2e-5)

    def test_free_field_matches_analytic(self):
        # 125 Hz at 27 and 55 points per wavelength, receivers 1.5-2.6 m away in
        # axial and oblique directions. No fitted gain: error is the grid's alone
        # and must fall at second order.
        scene = dict(source=[0, 0, 0], walls=[], speedOfSound=343,
                     points=[1.5, 0, 0, 0, 2.2, 0, 1.2, 1.2, 1.2, -2.5, .3, .7])
        exact = fw.free_field_pressure(scene['points'], scene['source'], 125)
        coarse, fine = (fw.relative_l2(fw.solve(scene, h, 125, .06, pml=12, margin=.6,
                                                work_fraction=1)['H'][0], exact) for h in (.1, .05))
        self.assertLess(fine, .004)
        self.assertGreater(coarse / fine, 3.5)

    def test_response_is_independent_of_pulse(self):
        scene = dict(source=[.03, -.02, .01], walls=[dict(center=[.8, 0, 0], half=[.2, .5, .5])],
                     speedOfSound=343, points=[1.5, .1, 0, -1, .5, .2])
        a = fw.solve(scene, .1, 125, .08, peak_hz=120, pml=10, work_fraction=1)['H']
        b = fw.solve(scene, .1, 125, .08, peak_hz=220, pml=10, work_fraction=1)['H']
        self.assertLess(fw.relative_l2(a, b), 1e-3)

    def test_rigid_box_modes(self):
        # Closed 1.2 x 0.9 x 0.7 m box, no PML: first modes from the analytic formula.
        grid = fw.Grid(.05, (0., 0., 0.), (24, 18, 14), 0)
        sim = fw.Simulation(grid, np.zeros(grid.shape, np.uint8))
        out = sim.run([.13, .21, .17], [[1.01, .66, .52]], .5, 300, work_fraction=1)
        x = out['pressure'][:, 0] * np.hanning(len(out['pressure']))
        f = np.fft.rfftfreq(len(x), out['dt'])
        s = np.abs(np.fft.rfft(x))
        band = (f > 100) & (f < 300)
        peaks = [f[i] for i in np.flatnonzero(band) if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > .05 * s[band].max()]
        dims = [1.2, .9, .7]
        modes = sorted(343 / 2 * math.sqrt(sum((m / d) ** 2 for m, d in zip(n, dims)))
                       for n in np.ndindex(4, 4, 4) if any(n))
        self.assertGreaterEqual(len(peaks), 3)
        for peak in peaks:
            self.assertLess(min(abs(peak - m) for m in modes) / peak, .01)

    def test_memory_guard(self):
        with self.assertRaises(MemoryError):
            fw.Simulation(fw.Grid(.01, (0., 0., 0.), (2000, 2000, 2000), 0), None)


if __name__ == '__main__':
    unittest.main()
