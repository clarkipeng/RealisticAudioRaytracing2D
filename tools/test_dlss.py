"""Checks for the procedural box-scene generator (tools/dlss_scenes.py); run with unittest.

No Unity or GPU is needed.
"""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

import dlss_scenes as ds
import fdtd_wave as fw

FIXTURE = Path(__file__).with_name('fixtures') / 'dlss-scenes-v2.json'


def union_volume(walls, spacing):
    """Exact union volume of lattice boxes by rasterising at `spacing`."""
    low, high = ds.box_arrays(walls)
    origin = low.min(0)
    lo = np.round((low - origin) / spacing).astype(int)
    hi = np.round((high - origin) / spacing).astype(int)
    grid = np.zeros(hi.max(0), bool)
    for l, h in zip(lo, hi):
        grid[l[0]:h[0], l[1]:h[1], l[2]:h[2]] = True
    return float(grid.sum() * spacing ** 3)


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
        overlapping = [dict(center=[.5, .5, .5], half=[.5, .5, .5]), dict(center=[1, .5, .5], half=[.5, .5, .5])]
        self.assertAlmostEqual(union_volume(overlapping, ds.LATTICE), 1.5)
        for scene in self.scenes.values():
            exact = union_volume(scene['walls'], ds.LATTICE)
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


if __name__ == '__main__':
    unittest.main()
