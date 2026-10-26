"""Analytic controls for the small direct-rain fixture."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import rain_source as rain
import rain_spatial as spatial


class RainSpatialChecks(unittest.TestCase):
    def test_free_field_amplitude_delay_and_causality(self):
        config = spatial.scene()
        config['walls'] = []
        source = np.array([[3.43, 0., 0.], [6.86, 0., 0.]])
        path = spatial.transfer(source, [0., 0., 0.], config)
        assert_allclose(path['gain'], [1/3.43, 1/6.86])
        assert_array_equal(path['delay'], [441, 882])
        events = np.array([(0, 0, 1.)], dtype=rain.EVENT)
        pulse = spatial.arrivals(events, spatial.transfer(source[:1], [0., 0., 0.], config))
        bank = rain.Bank(np.array([[0., 1., -1., 0.]]), np.array([1.]), {})
        audio = rain.compact(bank, pulse, 900)
        assert_allclose(audio[:441], 0, atol=1e-16)
        assert_allclose(audio[441:445], bank.pcm[0]/3.43, atol=1e-16)

    def test_roof_blocks_and_doorway_passes_direct_sound(self):
        config = spatial.scene()
        sources = np.array([[-2., 3.251, 0.], [4., .001, 0.], [4., .001, 6.]])
        path = spatial.transfer(sources, config['receivers']['inside'], config)
        assert_array_equal(path['visible'], [False, True, False])

    def test_positions_survive_query_splits_and_patches_stay_on_surface(self):
        config, bank = spatial.scene(), rain.make_splash_bank()
        full = spatial.field(bank, 23, 0, 97031, config, .2)
        cuts = [0, 17, 44099, 44101, 70123, 97031]
        parts = [spatial.field(bank, 23, a, b, config, .2) for a, b in zip(cuts, cuts[1:])]
        assert_array_equal(np.concatenate(parts), full)
        for width in [2., 1., .5, .25]:
            grouped = spatial.patch_positions(full, config, width)
            assert_array_equal(grouped[:, 1], full['position'][:, 1])
            for index, surface in enumerate(config['surfaces']):
                points = grouped[full['surface'] == index]
                for axis, key in [(0, 'x'), (2, 'z')]:
                    self.assertTrue(np.all((points[:, axis] >= surface[key][0]) &
                                           (points[:, axis] < surface[key][1])))


if __name__ == '__main__':
    unittest.main()
