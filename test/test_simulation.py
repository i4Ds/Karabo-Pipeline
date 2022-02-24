import unittest

import numpy as np

from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.interferometer import InterferometerSimulation


class TestSimulation(unittest.TestCase):

    def test_basic(self):
        sky = SkyModel()
        sky_data = np.array([
            [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
            [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
            [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10]])
        sky.add_points_sources(sky_data)
        telescope = Telescope.get_OSKAR_Example_Telescope()
        simulation = InterferometerSimulation("./test_result.ms")
        observation = Observation(1e6)
        simulation.run_simulation(telescope, sky, observation)
