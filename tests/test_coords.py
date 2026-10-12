import unittest

import numpy as np

from neuron_ng.core.coords import CoordinateSpace


class TestCoordinateSpace(unittest.TestCase):
    def test_invalid_scale_rejected(self):
        for scale in ((1, 1), (1, 0, 1), (1, -1, 1), (1, float("nan"), 1)):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                CoordinateSpace(scale)

    def test_dimensions_in_meters(self):
        space = CoordinateSpace((0.5, 0.5, 2.0))
        self.assertEqual(space.dimensions_json(), {
            "x": [5e-07, "m"], "y": [5e-07, "m"], "z": [2e-06, "m"],
        })

    def test_identity_skeleton_transform(self):
        space = CoordinateSpace((0.5, 0.5, 2.0))
        self.assertTrue(space.is_identity)
        self.assertEqual(space.skeleton_transform_nm(),
                         [500.0, 0, 0, 0, 0, 500.0, 0, 0, 0, 0, 2000.0, 0])
        self.assertEqual(space.source_transform(), {"outputDimensions": space.dimensions_json()})
        self.assertEqual(space.apply((1, 2, 3)), (1.0, 2.0, 3.0))

    def test_affine_reaches_skeleton_transform_and_positions(self):
        # Flip x about 1000 and shift z, e.g. a hemisphere-mirrored registration.
        affine = [[-1, 0, 0, 1000], [0, 1, 0, 0], [0, 0, 1, 5]]
        space = CoordinateSpace((10, 10, 10), affine=affine)
        self.assertFalse(space.is_identity)
        np.testing.assert_allclose(
            np.reshape(space.skeleton_transform_nm(), (3, 4)),
            [[-1e4, 0, 0, 1e7], [0, 1e4, 0, 0], [0, 0, 1e4, 5e4]],
        )
        self.assertEqual(space.apply((1, 2, 3)), (999.0, 2.0, 8.0))
        self.assertEqual(space.source_transform()["matrix"], [[-1.0, 0.0, 0.0, 1000.0],
                                                              [0.0, 1.0, 0.0, 0.0],
                                                              [0.0, 0.0, 1.0, 5.0]])

    def test_affine_accepts_4x4_and_rejects_bad_shapes(self):
        four = np.eye(4)
        four[0, 3] = 7
        self.assertEqual(CoordinateSpace((1, 1, 1), four).apply((0, 0, 0)), (7.0, 0.0, 0.0))
        for bad in (np.eye(3), np.ones((4, 4)), [[1, 2, 3, float("inf")]] * 3):
            with self.subTest(affine=bad), self.assertRaises(ValueError):
                CoordinateSpace((1, 1, 1), bad)

    def test_coerce(self):
        space = CoordinateSpace((1, 2, 3))
        self.assertIs(CoordinateSpace.coerce(space), space)
        self.assertEqual(CoordinateSpace.coerce((1, 2, 3)), space)


if __name__ == "__main__":
    unittest.main()
