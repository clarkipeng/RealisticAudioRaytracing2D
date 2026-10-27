import unittest

import numpy as np
from numpy.testing import assert_allclose

import rain_area as area
import rain_texture


class AreaRainChecks(unittest.TestCase):
    def test_clipped_patches_preserve_area_and_surface(self):
        surfaces = [dict(x=(-3, 1.3), z=(-2, 2), height=3.251, profile='roof'),
                    dict(x=(1.3, 4), z=(-2, 2), height=.001, profile='ground')]
        patches = area.patches(surfaces, 2)
        self.assertAlmostEqual(sum(p.area for p in patches), 28)
        for p in patches:
            self.assertEqual(p.profile, 'roof' if p.x[1] <= 1.3 else 'ground')
            self.assertLessEqual(p.area, 4)

    def test_ray_count_keeps_area_power_and_stereo_sum(self):
        patches = [area.Patch((-1, 1), (-1, 1), 0, 'ground')]
        p = np.array([[[0., 0., 0.]]])
        gain, delay = area.response(patches, p, [0, 2, 0], [])
        assert_allclose(np.sum(gain*gain), 4/2**2)
        assert_allclose(delay, [2/343])
        repeated, _ = area.response(patches, np.repeat(p, 17, axis=1), [0, 2, 0], [])
        assert_allclose(gain, repeated)

    def test_roof_blocks_sources_above_but_not_below(self):
        patches = [area.Patch((-1, 1), (-1, 1), 3.251, 'roof')]
        roof = dict(center=[0, 3.125, 0], half=[2, .125, 2])
        points = area.sites(patches, 4)
        gain, _ = area.response(patches, points, [0, 1.6, 0], [roof])
        assert_allclose(gain, 0)
        visible, _ = area.response(patches, points, [0, 4, 0], [roof])
        self.assertGreater(visible.sum(), 0)

    def test_texture_and_delayed_mix_are_independent_of_batch_partition(self):
        rate = 2000
        recording = np.random.default_rng(5).standard_normal(10*rate)
        full = rain_texture.recording(recording, 4*rate, 23, 9, 0, 10001)
        cuts = [0, 3, 2000, 2001, 7319, 10001]
        split = np.concatenate([rain_texture.recording(recording, 4*rate, 23, 9, a, b)
                                for a, b in zip(cuts, cuts[1:])])
        assert_allclose(full, split, rtol=0, atol=0)
        knots = np.array([0., .2, .6, 1.])
        gain = np.array([[1., .5], [.8, .7], [.1, .2], [0., .1]])
        delay = np.array([.03, .032, .025, .03])
        full = area.mix_patch(recording, 23, 9, rate, knots, gain, delay, 0, rate)
        split = np.concatenate([area.mix_patch(recording, 23, 9, rate, knots, gain, delay, a, b)
                                for a, b in zip([0, 191, 876], [191, 876, rate])])
        assert_allclose(full, split, rtol=0, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
