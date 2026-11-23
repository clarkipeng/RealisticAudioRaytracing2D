"""Checks of tools/room_modes.py: the grid bake against analytic box modes, the probe, the damping and the
crossover's energy bookkeeping.

    PY=.context/fdtd-teacher/venv/bin/python
    nice -n 19 $PY -m unittest tools/test_room_modes.py
"""
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import method_compare as mc  # noqa: E402
import room_modes as rm  # noqa: E402

L = (1.5, 1.0, 1.25)


def box_walls(L, t=0.25):
    """Six wall boxes around the interior [0, L], materials 0..5 in the order x0, x1, y0, y1, z0, z1."""
    L = np.asarray(L)
    out = []
    for a in range(3):
        for side in range(2):
            lo, hi = -t * np.ones(3), L + t
            if side == 0:
                hi = hi.copy(); hi[a] = 0.0
            else:
                lo = lo.copy(); lo[a] = L[a]
            if a > 0:          # keep each box to its own face; corners belong to the x walls
                lo[0], hi[0] = 0.0, L[0]
            if a > 1:
                lo[1], hi[1] = 0.0, L[1]
            out.append((lo, hi, 2 * a + side))
    return out


class RoomModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fmax = 500.0
        cls.tmp = tempfile.TemporaryDirectory()
        cls.bank = rm.bake_boxes(box_walls(L), [f'm{i}' for i in range(6)], 0.0625, cls.fmax, out=cls.tmp.name, log=lambda m: None)
        cls.k, shape, cls.S = rm.box_modes(L, cls.fmax, [0, 1, 2, 3, 4, 5], 6)
        cls.shape = staticmethod(shape)

    def test_frequencies_match_the_analytic_box(self):
        fa = np.sort(self.k) * rm.C / (2 * math.pi)
        fg = np.sort(self.bank['k']) * rm.C / (2 * math.pi)
        m = min(len(fa), len(fg))
        self.assertLessEqual(abs(len(fa) - len(fg)), 3)
        rel = np.abs(fg[1:m] - fa[1:m]) / fa[1:m]
        self.assertLess(np.max(rel), 2e-3, 'fourth-order corrected eigenvalues within 0.2% to 500 Hz at 6 ppw')

    def test_wall_integrals_and_damping(self):
        # the constant mode: S_m = A_m / V for every face
        V = np.prod(L)
        areas = [L[1] * L[2]] * 2 + [L[0] * L[2]] * 2 + [L[0] * L[1]] * 2
        np.testing.assert_allclose(self.bank['surface'][0], np.array(areas) / V, rtol=1e-6)
        # Morse and Ingard: an oblique mode in uniform admittance decays at delta = 2 c beta sum_a 1 / L_a
        beta = [[0.01] * 8] * 6
        d = rm.damping(self.k, self.S, beta)
        self.assertAlmostEqual(d[0], 0.5 * rm.C * 0.01 * sum(areas) / V, places=9)
        nmax = np.floor(2 * self.fmax * np.asarray(L) / rm.C).astype(int)
        idx = np.stack(np.meshgrid(*[np.arange(n + 1) for n in nmax], indexing='ij'), -1).reshape(-1, 3)
        idx = idx[np.pi * np.sqrt(((idx / np.asarray(L)) ** 2).sum(1)) <= 2 * math.pi * self.fmax / rm.C]
        oblique = np.all(idx > 0, 1)
        np.testing.assert_allclose(d[oblique], 2 * rm.C * 0.01 * sum(1 / x for x in L), rtol=1e-9)
        # and the grid bake's damping of the constant mode agrees
        dg = rm.damping(self.bank['k'], self.bank['surface'], beta)
        self.assertAlmostEqual(dg[np.argmin(self.bank['k'])] / d[0], 1, places=6)

    def test_probe_is_fourth_order(self):
        pts = np.array([[0.11, 0.53, 0.97], [1.47, 0.02, 0.6], [0.75, 0.5, 0.625]])
        g = rm.probe(self.bank, pts)
        a = self.shape(pts)
        # match grid modes to analytic ones by frequency (non-degenerate ones only) and compare |psi|
        fa, fg = self.k, self.bank['k']
        errors = []
        for i in range(1, len(fa)):
            if np.sum(np.abs(fa - fa[i]) < 1e-6) > 1:
                continue
            j = int(np.argmin(np.abs(fg - fa[i])))
            if abs(fg[j] - fa[i]) / fa[i] > 2e-3 or np.sum(np.abs(fg - fg[j]) / fg[j] < 2e-3) > 1:
                continue
            sign = np.sign(np.dot(g[:, j], a[:, i]))
            errors.append(np.max(np.abs(sign * g[:, j] - a[:, i])) / np.max(np.abs(a[:, i])))
        self.assertGreater(len(errors), 20)
        print(f"probe: {len(errors)} isolated modes, median error {np.median(errors):.2e}, max {np.max(errors):.2e}")
        self.assertLess(np.median(errors), 0.01)

    def test_crossover_keeps_energy(self):
        f = np.geomspace(20, 2000, 2000)
        fs = 180.0
        W = rm.crossover(f, fs)
        self.assertTrue(np.all(W[f <= fs / math.sqrt(2)] == 1) and np.all(W[f >= fs * math.sqrt(2)] < 1e-12))
        g = rm.path_share(fs)
        # g_b is the band filter's energy above the crossover: 0 far below, 1 far above, between otherwise
        self.assertAlmostEqual(g[-1], 1.0, places=9)
        self.assertTrue(np.all((g >= 0) & (g <= 1)))
        # the direct sound passes in amplitude: W H + (1 - W) H = H, so its band energies are untouched
        Ed, _, _ = rm.direct_energy(dict(points=[1.0, 0, 0], source=[0, 0, 0]))
        self.assertAlmostEqual(Ed.sum(), 1.0)

    def test_schroeder_frequency_matches_the_classical_formula(self):
        # uniform modal density 4 pi V f^2 / c^3 and bandwidth 2.2 / T give f = 2095 sqrt(T / V)
        V, T = 200.0, 1.5
        f = np.sort(np.random.default_rng(0).uniform(0, 1500, int(4 * math.pi * V * 1500 ** 3 / (3 * rm.C ** 3))))
        f = 1500 * (np.arange(len(f)) / len(f)) ** (1 / 3)
        k = 2 * math.pi * f / rm.C
        delta = np.full(len(k), 3 * math.log(10) / T)
        fs = rm.schroeder_frequency(k, delta)
        self.assertAlmostEqual(fs / (2095 * math.sqrt(T / V)), 1, delta=0.03)


if __name__ == '__main__':
    unittest.main()
