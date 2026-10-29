"""Checks for the audio-DLSS scene generator and dataset packer; run with unittest.

No Unity or GPU is needed: tracer payloads are synthesised in the exact byte
layout TakeChecks.MeasureDatasetScenes writes.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import dlss_baselines as bl
import dlss_dataset as dd
import dlss_scenes as ds
import fdtd_wave as fw

FIXTURE = Path(__file__).with_name('fixtures') / 'dlss-scenes-v2.json'


class Generator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = {f: ds.generate(f, 0) for f in ds.FAMILIES}

    def test_regeneration_is_byte_identical_and_index_dependent(self):
        for family in ds.FAMILIES:
            self.assertEqual(ds.dumps(ds.generate(family, 0)), ds.dumps(self.scenes[family]))
            self.assertNotEqual(ds.dumps(ds.generate(family, 1)), ds.dumps(self.scenes[family]))

    def test_frozen_fixture_hashes(self):
        """Any generator change must bump GENERATOR_VERSION and refresh the fixture."""
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(fixture['generator'], ds.GENERATOR_VERSION)
        for family, scene in self.scenes.items():
            self.assertEqual(hashlib.sha256(ds.dumps(scene)).hexdigest(), fixture['scenes'][scene['sceneId']])

    def test_faces_are_on_a_float32_exact_lattice(self):
        for scene in self.scenes.values():
            low, high = ds.box_arrays(scene['walls'])
            for value in np.concatenate([low.ravel(), high.ravel()]):
                self.assertAlmostEqual(value / ds.LATTICE, round(value / ds.LATTICE), places=9)
                self.assertEqual(float(np.float32(value)), value)
            for wall in scene['walls']:
                for v in wall['center'] + wall['half']:
                    self.assertEqual(float(np.float32(v)), v)

    def test_fdtd_voxels_reproduce_the_box_union_exactly(self):
        for scene in self.scenes.values():
            exact, _ = dd.union_volume(scene['walls'], ds.LATTICE)
            for h in (ds.LATTICE, ds.FDTD_SPACING):
                low, high = fw.scene_bounds(scene, ds.FDTD_MARGIN)
                grid = fw.make_grid(low, high, h, 2)
                solid, snap = fw.voxelize(grid, scene['walls'])
                self.assertLess(snap, 1e-9)
                self.assertAlmostEqual(solid.sum() * h ** 3, exact, places=6)

    def test_endpoint_invariants(self):
        for scene in self.scenes.values():
            ds.validate(scene)
            points = np.array(scene['points']).reshape(-1, 3)
            self.assertGreaterEqual(len(points), 16)
            self.assertEqual(len(scene['lineOfSight']), len(points))
            self.assertEqual(len(scene['surfaceMaterials']), len(scene['walls']))
            self.assertTrue(set(scene['surfaceMaterials']) <= {'stone', 'wood', 'glass', 'ground'})

    def test_line_of_sight_matches_dense_segment_sampling(self):
        rng = np.random.default_rng(3)
        for scene in self.scenes.values():
            points = np.array(scene['points']).reshape(-1, 3)
            chosen = points[rng.choice(len(points), min(12, len(points)), replace=False)]
            fast = ds.line_of_sight(scene['source'], chosen, scene['walls'])
            t = np.linspace(0, 1, 4001)[1:-1, None]
            for point, visible in zip(chosen, fast):
                samples = np.asarray(scene['source']) + t * (point - scene['source'])
                self.assertEqual(bool(visible), bool(np.all(ds.clearance(samples, scene['walls']) > 0)))

    def test_split_rule_holds_out_families_and_ranges(self):
        self.assertEqual(ds.split_for('canopy', {}), 'test')
        self.assertEqual(ds.split_for('corridor', {}), 'validation')
        self.assertEqual(ds.split_for('doorway', dict(openingWidth=1.75)), 'test')
        self.assertEqual(ds.split_for('doorway', dict(openingWidth=1.0)), 'train')
        self.assertEqual(ds.split_for('furnished', dict(items=9)), 'validation')
        groups = {}
        for family in ds.FAMILIES:
            for index in range(8):
                scene = ds.generate(family, index)
                self.assertEqual(groups.setdefault(scene['splitGroup'], scene['split']), scene['split'])

    def test_seeds_are_stable_and_disjoint(self):
        self.assertEqual(ds.stable_seed('x', 1), ds.stable_seed('x', 1))
        seeds = [s for b in ds.budgets('shoebox-0000') for s in b['seeds']]
        self.assertEqual(len(seeds), len(set(seeds)))
        fractions = [ds.LABEL_BUDGET['completeRouteSamples'] // b['completeRouteSamples'] for b in ds.budgets('a')[1:]]
        self.assertEqual(fractions, list(ds.INPUT_FRACTIONS))

    def test_neighbours_are_symmetric(self):
        for scene in self.scenes.values():
            n = ds.neighbours(scene)
            for i, row in enumerate(n):
                for j in row:
                    self.assertIn(i, n[j])


def synthesise_tracer(folder, scene, variant='tracer', rng=None, label_noise=1e-3):
    """Write payloads in the C# byte layout with a known field plus seed noise."""
    rng = rng or np.random.default_rng(0)
    out = folder / variant
    out.mkdir()
    families = ['direct', 'R', 'D', 'DD', 'RD', 'DR', 'RR', 'walk']
    P, Q, F = len(scene['points']) // 3, len(scene['frequencies']), len(families)
    truth = rng.normal(size=(P, Q, F)) + 1j * rng.normal(size=(P, Q, F))
    budgets = []
    for b in scene['render']['budgets']:
        S = len(b['seeds'])
        scale = label_noise if b['name'] == 'label' else .1
        values = truth[None] + scale * (rng.normal(size=(S, P, Q, F)) + 1j * rng.normal(size=(S, P, Q, F)))
        np.stack([values.real, values.imag], -1).astype('<f4').tofile(out / f'responses-{b["name"]}.f32')
        np.full((S, P, Q, F), 3, '<i8').tofile(out / f'counts-{b["name"]}.i64')
        np.abs(values).astype('<f4').tofile(out / f'peaks-{b["name"]}.f32')
        np.full((S, P), .01, '<f4').tofile(out / f'seconds-{b["name"]}.f32')
        budgets.append(dict(name=b['name'], raysPerBin=b['raysPerBin'], completeRouteSamples=b['completeRouteSamples'],
                            seeds=b['seeds'], pathsPerSolve=1, busySeconds=1, wallSeconds=5))
    np.array(scene['lineOfSight'], '|u1').tofile(out / 'direct-visible.u8')
    (out / 'boundary.json').write_text('{}')
    run = dict(schema='dlss-tracer-run-v1', sceneId=scene['sceneId'],
               sceneSha256=hashlib.sha256((folder / 'scene.json').read_bytes()).hexdigest(), variant=variant,
               families=families, frequencies=scene['frequencies'], points=P, budgets=budgets)
    (out / 'run.json').write_text(json.dumps(run))
    return truth


class Packer(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.data = Path(self.folder.name)
        rng = np.random.default_rng(1)
        self.truth = {}
        for family in ['shoebox', 'canopy', 'corridor']:
            scene = ds.generate(family, 0)
            scene['points'] = scene['points'][:12]          # 4 receivers keep the fixture small
            scene['lineOfSight'] = scene['lineOfSight'][:4]
            scene['frequencies'] = scene['frequencies'][:3]
            folder = self.data / 'scenes' / scene['sceneId']
            folder.mkdir(parents=True)
            (folder / 'scene.json').write_bytes(ds.dumps(scene))
            self.truth[scene['sceneId']] = synthesise_tracer(folder, scene, rng=rng)

    def tearDown(self):
        self.folder.cleanup()

    def test_pack_verify_and_load_round_trip(self):
        target = dd.pack(self.data, 'fixture')
        report = dd.verify(target)
        self.assertEqual(report['bundles'], 3)
        self.assertEqual(report['heldOutFamilies'], ['canopy'])
        self.assertEqual(report['examples'], 3 * 4 * len(ds.INPUT_FRACTIONS) * ds.INPUT_SEEDS)
        manifest, bundles, base = dd.open_dataset(target)
        for bundle in bundles:
            data = dd.load_bundle(base, bundle, check=True)
            truth = self.truth[bundle['bundle_id']][..., :7].sum(2)   # early-field target excludes the walk
            self.assertLess(np.abs(data['target'] - truth).max(), 1e-2)
            self.assertTrue(data['valid'].all())
            self.assertEqual(data['inputs']['input-1-4'].shape, (ds.INPUT_SEEDS, 4, 3, 8))
            self.assertTrue(bundle['quality']['accepted'])

    def test_identity_volumes(self):
        v = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
        t = [0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5]
        self.assertAlmostEqual(abs(dd.mesh_volume(dict(vertices=v, triangles=t))), 1.0)
        walls = [dict(center=[.5, .5, .5], half=[.5, .5, .5]), dict(center=[1, .5, .5], half=[.5, .5, .5])]
        self.assertAlmostEqual(dd.union_volume(walls, ds.LATTICE)[0], 1.5)

    def test_label_statistics_are_unbiased(self):
        runs = np.zeros((3, 1, 1, 1, 2), np.float32)
        runs[:, 0, 0, 0, 0] = [1, 2, 6]
        mean, variance, k = dd.label_statistics(runs)
        self.assertAlmostEqual(mean[0, 0].real, 3)
        self.assertAlmostEqual(variance[0, 0], 7)   # ((-2)^2 + (-1)^2 + 3^2) / 2
        self.assertEqual(k, 3)

    def test_tampered_payload_fails_verification(self):
        target = dd.pack(self.data, 'fixture')
        payload = next((self.data / 'scenes').glob('*/tracer/responses-input-1-16.f32'))
        raw = bytearray(payload.read_bytes())
        raw[0] ^= 1
        payload.write_bytes(bytes(raw))
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            dd.verify(target)

    def test_changed_scene_bytes_break_identity(self):
        scene = next((self.data / 'scenes').glob('*/scene.json'))
        scene.write_bytes(scene.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'different scene bytes'):
            dd.pack(self.data, 'fixture')

    def test_noisy_labels_are_rejected_and_reported(self):
        folder = next((self.data / 'scenes').glob('shoebox-*'))
        scene = json.loads((folder / 'scene.json').read_text())
        import shutil
        shutil.rmtree(folder / 'tracer')
        synthesise_tracer(folder, scene, rng=np.random.default_rng(5), label_noise=10)
        target = dd.pack(self.data, 'fixture')
        generation = json.loads((target / 'generation.json').read_text())
        self.assertEqual(generation['rejectedByFamily']['shoebox'], 1)
        report = dd.verify(target)
        self.assertEqual(report['rejected'], 1)


class Baselines(unittest.TestCase):
    def test_metrics_and_baselines(self):
        target = np.ones((4, 2), complex)
        self.assertAlmostEqual(bl.relative_error_squared(np.full((3, 4, 2), 2 + 0j), target), 1.0)
        runs = np.arange(8, dtype=float).reshape(4, 2) + 0j
        self.assertTrue(np.allclose(bl.seed_mean(runs, 2), [[1, 2], [5, 6]]))
        totals = np.array([[[1.], [3.], [5.]]], complex)          # S=1, P=3, Q=1
        out = bl.neighbour_mean(totals, [[1], [0, 2], [1]])
        self.assertTrue(np.allclose(out[0, :, 0], [2, 3, 4]))
        spiky = np.ones((1, 5, 1, 2), complex)
        spiky[0, 2, 0, 1] = 100
        clamped = bl.family_clamp(spiky)
        self.assertAlmostEqual(clamped[0, 2, 0].real, 4.0)        # 1 + 3 x median 1
        self.assertTrue(np.allclose(clamped[0, [0, 1, 3, 4], 0], 2))
        masks = bl.band_masks([125, 500, 625, 4000])
        self.assertEqual(masks['low'].tolist(), [True, True, False, False])


if __name__ == '__main__':
    unittest.main()
