import unittest

import numpy as np
from numpy.testing import assert_array_equal

import rain_texture


class RainTextureChecks(unittest.TestCase):
    def test_irregular_queries_preserve_samples_and_region_identity(self):
        bank = np.random.default_rng(4).standard_normal((16, 1000))
        full = rain_texture.render(bank, 23, 0, 0, 5001)
        cuts = [0, 1, 499, 500, 1001, 1783, 5001]
        split = np.concatenate([rain_texture.render(bank, 23, 0, a, b)
                                for a, b in zip(cuts, cuts[1:])])
        assert_array_equal(full, split)
        self.assertFalse(np.array_equal(full, rain_texture.render(bank, 23, 1, 0, 5001)))
        self.assertEqual(len(rain_texture.render(bank, 23, 0, 0, 0)), 0)

    def test_overlap_does_not_introduce_periodic_expected_power_dips(self):
        # Stationary independent grains should not pump at the 50% overlap rate.
        # An ensemble checks the renderer, including its initialization and seams.
        rng = np.random.default_rng(93)
        bank = rng.standard_normal((256, 200))
        power = np.mean([rain_texture.render(bank, seed, 0, 0, 2000)**2
                         for seed in range(256)], axis=0)
        phase_power = power.reshape(20, 100).mean(axis=0).reshape(10, 10).mean(axis=1)
        self.assertLess(np.max(np.abs(phase_power-1)), .08)


if __name__ == '__main__':
    unittest.main()
