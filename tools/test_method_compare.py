"""Checks for the method-comparison format, band energies and scoring; run with unittest.

No GPU is needed: the harness is exercised on synthetic ground truth and
synthetic perturbed methods.
"""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import method_compare as mc

C = 343.0


def room_histogram(distance, t60, level_db, bins=mc.TIME_BINS, start=None):
    """Direct sound at `distance` plus an exponential tail (energy per 1 ms bin), every band."""
    e = np.zeros((len(mc.BANDS), bins))
    k0 = int(distance / C / mc.BIN_SECONDS)
    e[:, k0] = 1 / distance ** 2
    t = (np.arange(bins) - k0) * mc.BIN_SECONDS
    tail = 10 ** (level_db / 10) * np.exp(-13.8155 * t / t60) * mc.BIN_SECONDS
    e[:, (start or k0 + 3):] += tail[(start or k0 + 3):]
    return e


def synthetic_scene(root, sid, receivers=6):
    rng = np.random.default_rng(5)
    pts = rng.uniform(1, 4, (receivers, 3))
    scene = dict(schema=mc.SCENE_SCHEMA, sceneId=sid, family='shoebox', variant='presets', closed=True,
                 source=[0, 1.5, 0], walls=[], points=pts.ravel().tolist(), speedOfSound=C,
                 shadow=[i % 2 for i in range(receivers)], bands=mc.BANDS.tolist(), surfaceMaterials=[],
                 materials={}, gtBands=[125.0, 250.0, 500.0], viewAxes=[0, 2])
    folder = Path(root) / sid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'scene.json').write_text(json.dumps(scene))
    return scene, folder / 'scene.json'


def meta_for(method, budget, seed, scene_path, **extra):
    meta = dict(method=method, budget=budget, seed=seed, sceneId=Path(scene_path).parent.name,
                sceneSha256=mc.sha256_file(scene_path), code=dict(name='test', sha256='0'), wallSeconds=1.0,
                validSeconds=2.0)
    meta.update(extra)
    return meta


class BandFilters(unittest.TestCase):
    def test_power_complementary(self):
        f = np.linspace(125, 16000, 50001)
        total = sum(mc.band_power(f, fc) for fc in mc.BANDS)
        np.testing.assert_allclose(total, 1, atol=1e-12)

    def test_free_field_normalisation_impulse(self):
        rate, n = 48000, 378
        d = n / rate * C
        h = np.zeros((1, 2 * rate))
        h[0, n] = 1 / d
        e, _ = mc.energy_from_impulse(h, rate)
        provided = np.isfinite(e[0, :, 0])
        np.testing.assert_allclose(e[0, provided].sum(-1) * d * d, 1, atol=.02)
        self.assertTrue(provided.all())           # 16 kHz: nominal band below 24 kHz, skirt renormalised
        peak = np.argmax(e[0, 2])
        self.assertEqual(peak, int(d / C / mc.BIN_SECONDS))

    def test_free_field_normalisation_spectrum(self):
        d, df = 4.0, .25
        f = np.arange(0, 4000 + df, df)
        H = np.exp(-2j * np.pi * f * d / C) / d
        e, beyond = mc.energy_from_spectrum(H[None], df, [125., 250., 500., 1000.])
        np.testing.assert_allclose(e[0].sum(-1) * d * d, 1, rtol=1e-5)

    def test_arrivals_add_coherently(self):
        same = mc.energy_from_arrivals([.01, .01], np.ones((2, 8)), bands=[250.])
        apart = mc.energy_from_arrivals([.01, .5], np.ones((2, 8)), bands=[250.])
        opposite = mc.energy_from_arrivals([.01, .01], np.array([[1.] * 8, [-1.] * 8]), bands=[250.])
        self.assertAlmostEqual(same.sum() / apart.sum(), 2, delta=.02)
        self.assertAlmostEqual(apart.sum(), 2, delta=.03)
        self.assertLess(opposite.sum(), 1e-9)

    def test_levels_from_complex_matches_band_total(self):
        d1, d2, df = 3.0, 5.2, .25
        f = np.arange(0, 2000 + df, df)
        H = np.exp(-2j * np.pi * f * d1 / C) / d1 + .7 * np.exp(-2j * np.pi * f * d2 / C) / d2
        e, _ = mc.energy_from_spectrum(H[None], df, [250.])
        lv = mc.levels_from_complex(H[None], f, [250.])
        self.assertAlmostEqual(e.sum() / lv[0, 0], 1, delta=1e-3)

    def test_kernel_preserves_energy_and_smooth_is_linear(self):
        for fc in mc.BANDS:
            self.assertAlmostEqual(mc.energy_kernel(fc).sum(), 1, delta=1e-3)
        e = room_histogram(3, .8, -10)
        s = mc.smooth(e[None])[0]
        np.testing.assert_allclose(s.sum(-1), e.sum(-1), rtol=1e-3)


class Metrics(unittest.TestCase):
    def test_decay_times_of_exponential(self):
        e = room_histogram(2, .6, 0)
        m = mc.receiver_metrics(e[0], mc.TIME_BINS, mc.onset_bin(e[0]))
        self.assertAlmostEqual(m['T30'], .6, delta=.006)
        self.assertAlmostEqual(m['T20'], .6, delta=.006)
        self.assertTrue(math.isnan(mc.receiver_metrics(e[0], 300, 5)['T30']))   # -35 dB not reached in 0.3 s

    def test_clarity_and_definition(self):
        e = np.zeros(mc.TIME_BINS)
        e[10], e[100] = 3, 1
        m = mc.receiver_metrics(e, mc.TIME_BINS, 10)
        self.assertAlmostEqual(m['C50'], 10 * math.log10(3), places=9)
        self.assertAlmostEqual(m['D50'], .75, places=9)
        self.assertEqual(m['onset'], 10)

    def test_etc_distance(self):
        e = room_histogram(3, .8, -10)[0]
        self.assertEqual(mc.etc_distance(e, e, 1500), 0)
        self.assertAlmostEqual(mc.etc_distance(2 * e, e, 200), 10 * math.log10(2), delta=1e-6)


class Format(unittest.TestCase):
    def test_roundtrip_and_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, path = synthetic_scene(tmp, 's-0')
            R = len(scene['points']) // 3
            e = np.stack([room_histogram(3, .8, -10) for _ in range(R)])
            e[:, :, 1900:] = 0
            e[:, 3:] = np.nan
            hz = [125., 250.]
            cp = np.ones((R, 2)) * (1 + 1j)
            d = np.repeat(e[..., None] / 6, 6, -1)
            out = Path(tmp) / 's-0/energy/r00064/3'
            mc.write_output(out, meta_for('energy', 'r00064', 3, path, frequenciesHz=hz, validSeconds=1.9),
                            e, cp, d)
            self.assertEqual(mc.check_output(out, path), [])
            meta, e2, c2, d2 = mc.read_output(out)
            np.testing.assert_array_equal(np.isnan(e2), np.isnan(e))
            np.testing.assert_allclose(c2, cp)
            self.assertEqual(meta['bandsProvided'], [125., 250., 500.])
            self.assertIn(out.parent.parent.parent.name, mc.discover(tmp))
            # A late nonzero bin past validSeconds and a bad seed are reported / rejected.
            raw = np.fromfile(out / 'energy.f32', '<f4').reshape(e.shape)
            raw[0, 0, 1950] = 1
            raw.astype('<f4').tofile(out / 'energy.f32')
            self.assertTrue(any('after validSeconds' in p for p in mc.check_output(out, path)))
            with self.assertRaises(ValueError):
                mc.write_output(Path(tmp) / 'x', meta_for('energy', 'R1', -1, path), e)


class Scoring(unittest.TestCase):
    """GT against itself scores zero; a perturbed method scores its perturbation."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        scene, path = synthetic_scene(root, 'shoebox-9999-presets')
        R = len(scene['points']) // 3
        pts = np.array(scene['points']).reshape(-1, 3)
        dist = np.linalg.norm(pts - scene['source'], axis=1)
        raw = np.stack([room_histogram(d, .9, -12) for d in dist])
        raw[:, 3:] = np.nan
        cls.raw = raw
        gt = mc.smooth(raw)                                   # band-limited ground truth
        mc.write_output(root / scene['sceneId'] / 'gt/ref/0',
                        meta_for('gt', 'ref', 0, path, bandLimited=True, budgetValue=1), gt)
        rng = np.random.default_rng(7)
        for budget, value in (('r00016', 16), ('r00256', 256)):
            for seed in range(4):
                noise = 1 + rng.standard_normal(raw.shape) * (2 / math.sqrt(value))
                k0 = (dist / C / mc.BIN_SECONDS).astype(int)
                for r in range(R):
                    noise[r, :, :k0[r] + 1] = 1                # deterministic direct sound
                pert = np.clip(raw * 2 * noise, 0, None)       # +3 dB and tail noise falling as 1/sqrt(N)
                pert[:, 3:] = np.nan
                mc.write_output(root / scene['sceneId'] / f'energy/{budget}/{seed}',
                                meta_for('energy', budget, seed, path, budgetValue=value), pert)
        (root / 'suite.json').write_text(json.dumps(dict(schema=mc.SUITE_SCHEMA, scenes=[dict(id=scene['sceneId'])])))
        cls.report = mc.score(root, root / 'score')
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, method, budget=None):
        return [r for r in self.report['rows'] if r['method'] == method and (budget is None or r['budget'] == budget)]

    def test_ground_truth_against_itself_is_zero(self):
        rows = self.rows('gt')
        self.assertTrue(rows)
        for r in rows:
            for k in ('level', 'levelBias', 'onset', 'EDT', 'T20', 'T30', 'C50', 'C80', 'D50', 'etc'):
                if r[k] is not None:
                    self.assertEqual(r[k], 0, k)

    def test_perturbed_method_scores_its_gain(self):
        top = self.rows('energy', 'r00256')
        bias = mc.aggregate(top, ('method',), 'levelBias')[('energy',)]
        self.assertAlmostEqual(bias, 10 * math.log10(2), delta=.15)
        t30 = mc.aggregate(top, ('method',), 'T30')[('energy',)]
        self.assertLess(t30, 5)

    def test_error_and_spread_fall_with_budget(self):
        low = mc.aggregate(self.rows('energy', 'r00016'), ('method',), 'etc')[('energy',)]
        high = mc.aggregate(self.rows('energy', 'r00256'), ('method',), 'etc')[('energy',)]
        self.assertLess(high, low)
        spread = {s['budget']: s['histogramSD'] for s in self.report['spreads']
                  if s['method'] == 'energy' and s['group'] == 'all' and s['band'] == 125.0}
        self.assertAlmostEqual(spread['r00016'] / spread['r00256'], 4, delta=.8)

    def test_summary_and_viewer_build(self):
        text = (self.root / 'score/summary.md').read_text()
        self.assertIn('## energy (budget r00256', text)
        import method_compare_view as mv
        index = mv.build(self.root, self.root / 'view')
        self.assertTrue(index.exists())
        data = (self.root / 'view/data/shoebox-9999-presets.js').read_text()
        self.assertIn('"energy"', data)


if __name__ == '__main__':
    unittest.main()
