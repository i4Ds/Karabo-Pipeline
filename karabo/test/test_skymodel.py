import unittest

import numpy as np

from karabo.simulation.sky_model import SkyModel, get_GLEAM_Sky


class TestSkyModel(unittest.TestCase):

    def test_init(self):
        sky1 = SkyModel()
        sky_data = np.array([
            [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0, 'source1'],
            [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45, 'source2'],
            [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10, 'source3']])
        sky1.add_point_sources(sky_data)
        sky2 = SkyModel(sky_data)
        # test if sources are inside now
        self.assertEqual(sky_data.shape, sky1.sources.shape)
        self.assertEqual(sky_data.shape, sky2.sources.shape)

    def test_not_full_array(self):
        sky1 = SkyModel()
        sky_data = np.array([
            [20.0, -30.0, 1],
            [20.0, -30.5, 3],
            [20.5, -30.5, 3]])
        sky1.add_point_sources(sky_data)
        sky2 = SkyModel(sky_data)
        # test if doc shape were expanded
        self.assertEqual(sky1.sources.shape, (sky_data.shape[0], 13))
        self.assertEqual(sky2.sources.shape, (sky_data.shape[0], 13))

    def test_plot_gleam(self):
        sky = get_GLEAM_Sky()
        sky.plot_sky([250, -80])
        cartesian_sky = sky.get_cartesian_sky()
        print(cartesian_sky)

    def test_get_cartesian(self):
        sky1 = SkyModel()
        sky_data = np.array([
            [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0, 'source1'],
            [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45, 'source2'],
            [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10, 'source3']])
        sky1.add_point_sources(sky_data)
        cart_sky = sky1.get_cartesian_sky()
        print(cart_sky)

