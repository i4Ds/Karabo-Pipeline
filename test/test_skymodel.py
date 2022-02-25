import unittest

import numpy as np

from karabo.simulation.sky_model import SkyModel


class TestSkyModel(unittest.TestCase):

    def test_init(self):
        sky = SkyModel()
        sky_data = np.array([
            [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
            [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
            [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10]])
        sky.add_points_sources(sky_data)
        print(sky.sources)
        # test if sources are inside now
        self.assertEqual(sky_data.shape, sky.sources.shape)

    def test_not_full_array(self):
        sky = SkyModel()
        sky_data = np.array([
            [20.0, -30.0, 1],
            [20.0, -30.5, 3],
            [20.5, -30.5, 3]])
        sky.add_points_sources(sky_data)
        print(sky.sources)
        # test if docs shape were expanded
        self.assertEqual(sky.sources.shape, (3, 12))

