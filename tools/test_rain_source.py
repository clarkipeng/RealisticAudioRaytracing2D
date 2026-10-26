"""Source-clock and independent mixing checks; no playback or Unity."""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import rain_source as rain


class RainSourceChecks(unittest.TestCase):
    def test_overlap_add_matches_explicit_for_collisions_boundaries_and_tail(self):
        bank = rain.Bank(np.array([[0., 1., -.5, -.5, 0.], [0., -.2, .6, -.4, 0.]]),
                         np.array([.5, .5]), {})
        events = np.array([(0, 0, 1.), (3, 1, .3), (3, 1, -.1), (4, 0, .7),
                           (15, 1, 2.), (18, 0, 1.)], dtype=rain.EVENT)
        reference = rain.explicit(bank, events, 19)
        for block in [1, 3, 4, 7, 32]:
            assert_allclose(rain.compact(bank, events, 19, block), reference, atol=2e-15, rtol=2e-14)
        self.assertNotEqual(reference[-2], 0)  # Final event tail must survive.

    def test_absolute_clock_survives_irregular_partition_queries(self):
        bank = rain.make_bank('hard')
        full = rain.events(bank, 5, 11, 0, 97031, .2)
        for cuts in [[0, 1, 2205, 44100, 50003, 97031], [0, 97031]]:
            pieces = [rain.events(bank, 5, 11, a, b, .2) for a, b in zip(cuts, cuts[1:])]
            assert_array_equal(np.concatenate(pieces), full)
        other = rain.events(bank, 5, 12, 0, 97031, .2)
        self.assertNotEqual(full.tobytes(), other.tobytes())

    def test_authored_bank_renders_across_short_and_long_blocks(self):
        bank = rain.make_bank('water')
        events = rain.events(bank, 7, 0, 0, 4410, .5)
        reference = rain.explicit(bank, events, 4410)
        for block in [511, 2048, 4096]:
            actual = rain.compact(bank, events, 4410, block)
            assert_allclose(actual, reference, atol=2e-16, rtol=1e-12)
        self.assertGreater(np.max(abs(reference[4410:])), 0)


if __name__ == '__main__':
    unittest.main()
