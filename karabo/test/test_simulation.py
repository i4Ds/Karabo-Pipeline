import os
import unittest

import numpy as np

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


class TestSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists("result/sim"):
            os.makedirs("result/sim")

    def test_oskar_simulation_basic(self):
        # Tests oskar simulation. Should use GPU if available and if not, CPU.
        sky = SkyModel()
        sky_data = np.array(
            [
                [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
                [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
                [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10],
            ]
        )
        sky.add_point_sources(sky_data)
        sky = SkyModel.get_random_poisson_disk_sky((220, -60), (260, -80), 1, 1, 1)
        sky.explore_sky([240, -70], s=10)
        telescope = Telescope.get_OSKAR_Example_Telescope()
        telescope.centre_longitude = 3

        simulation = InterferometerSimulation(
            channel_bandwidth_hz=1e6,
            time_average_sec=10,
        )
        observation = Observation(
            start_frequency_hz=100e6,
            phase_centre_ra_deg=240,
            phase_centre_dec_deg=-70,
            number_of_time_steps=24,
            frequency_increment_hz=20e6,
            number_of_channels=64,
        )

        simulation.run_simulation(telescope, sky, observation)

    def test_create_observations_oskar_settings_tree(self):
        CHANNEL_BANDWIDTH_HZ = 1e6
        NUM_SPLITS = 5
        NUM_CHANNELS = 10
        observation = Observation(
            start_frequency_hz=100e6,
            number_of_channels=NUM_CHANNELS,
        )
        observations = Observation.extract_multiple_observations_from_settings(
            observation.get_OSKAR_settings_tree(),
            NUM_SPLITS,
            CHANNEL_BANDWIDTH_HZ,
        )

        for i, observation in enumerate(observations):
            observation = observation["observation"]
            assert (
                float(observation["start_frequency_hz"])
                == 100e6 + i * CHANNEL_BANDWIDTH_HZ * NUM_CHANNELS / NUM_SPLITS + i * 1
            )
            assert float(observation["num_channels"]) == 2

        assert len(observations) == NUM_SPLITS

    def test_parallelization_by_channel(self):
        # TODO: Do this test
        pass
