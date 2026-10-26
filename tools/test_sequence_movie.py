"""Geometry-dependent HUD checks; no Unity or audio playback."""
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sequence_movie import SurfaceProjection


class SurfaceProjectionChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        dtype = [('vertices', '<f4', (3, 3)), ('owner', '<i4'), ('material', '<i4')]
        triangles = np.zeros(4, dtype=dtype)
        for owner, (height, radius) in enumerate([(2, 1), (0, 10)]):
            a, b, c, d = [(-radius, height, -radius), (radius, height, radius),
                          (radius, height, -radius), (-radius, height, radius)]
            triangles[owner * 2]['vertices'] = [a, c, b]  # Reversed winding.
            triangles[owner * 2 + 1]['vertices'] = [a, d, b]
            triangles[owner * 2:owner * 2 + 2]['owner'] = owner
        payload = triangles.tobytes()
        (self.directory / 'geometry.bin').write_bytes(payload)
        self.scene = {'geometryHash': hashlib.sha256(payload).hexdigest()}
        self.projection = SurfaceProjection(self.directory, self.scene)
        self.poses = np.zeros((2, 20))
        self.poses[:, 6] = 1
        self.poses[:, 19] = np.array(3, dtype=np.uint32).view(np.float32)

    def test_roof_shields_ground_and_misses_stay_silent(self):
        np.testing.assert_allclose(self.projection.down([.3, 8, .2], self.poses), [.3, 2.001, .2])
        np.testing.assert_allclose(self.projection.down([2, 8, 0], self.poses), [2, .001, 0])
        self.assertIsNone(self.projection.down([20, 8, 0], self.poses))

    def test_birth_pose_and_geometry_flags_invalidate_cached_surface(self):
        self.projection.down([0, 8, 0], self.poses)
        angle = math.pi / 12
        self.poses[0, 1] = 1
        self.poses[0, 3:7] = [0, 0, math.sin(angle / 2), math.cos(angle / 2)]
        normal = np.array([-math.sin(angle), math.cos(angle), 0])
        expected = np.array([0, 1 + 2 / math.cos(angle), 0]) + normal * .001
        np.testing.assert_allclose(self.projection.down([0, 8, 0], self.poses), expected)
        for flag in [1, 2]:
            self.poses[0, 19] = np.array(flag, dtype=np.uint32).view(np.float32)
            np.testing.assert_allclose(self.projection.down([0, 8, 0], self.poses), [0, .001, 0])

    def test_geometry_hash_is_required(self):
        (self.directory / 'geometry.bin').write_bytes(b'')
        with self.assertRaises(ValueError):
            SurfaceProjection(self.directory, self.scene)


if __name__ == '__main__':
    unittest.main()
