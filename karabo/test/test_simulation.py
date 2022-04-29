import os
import unittest

import numpy as np

from karabo.util.jupyter import setup_jupyter_env

setup_jupyter_env()
from karabo.Imaging.imager import Imager
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope, get_OSKAR_Example_Telescope, get_SKA1_MID_Telescope
from karabo.simulation.interferometer import InterferometerSimulation


class TestSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists('result/sim'):
            os.makedirs('result/sim')

    def test_basic(self):
        sky = SkyModel()
        sky_data = np.array([
            [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
            [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
            [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10]])
        sky.add_point_sources(sky_data)

        telescope = get_OSKAR_Example_Telescope()
        # telescope.centre_longitude = 3

        simulation = InterferometerSimulation(channel_bandwidth_hz=1e6,
                                              time_average_sec=10)
        observation = Observation(100e6,
                                  phase_centre_ra_deg=20,
                                  phase_centre_dec_deg=-30,
                                  number_of_time_steps=24,
                                  frequency_increment_hz=20e6,
                                  number_of_channels=64)

        visibility = simulation.run_simulation(telescope, sky, observation)

        imager = Imager(visibility, imaging_npixel=2048,
                        imaging_cellsize=3.878509448876288e-05)

        dirty = imager.get_dirty_image()
        dirty.save_as_fits("result/dirty.fits")
        dirty.plot()
