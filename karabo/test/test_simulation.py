import os
import unittest
from datetime import datetime, timedelta

import numpy as np

from karabo.imaging.imager import Imager
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
        CENTRAL_FREQS = [200, 250, 370, 300, 301]
        CHANNEL_BANDWIDTHS = [1, 2, 3, 4, 5]
        N_CHANNELS = [1, 2, 3, 4, 5]

        obs = Observation()
        settings_tree = obs.get_OSKAR_settings_tree()
        observations = Observation.create_observations_oskar_from_lists(
            settings_tree, CENTRAL_FREQS, CHANNEL_BANDWIDTHS, N_CHANNELS
        )
        for i, observation in enumerate(observations):
            start_frequency_hz = float(observation["observation"]["start_frequency_hz"])
            n_channels = int(observation["observation"]["num_channels"])
            freq_inc_hz = float(observation["observation"]["frequency_inc_hz"])

            assert start_frequency_hz == CENTRAL_FREQS[i]
            assert n_channels == N_CHANNELS[i]
            assert freq_inc_hz == CHANNEL_BANDWIDTHS[i]

        CENTRAL_FREQS = [200, 300, 400]
        CHANNEL_BANDWIDTHS = 5
        N_CHANNELS = 1

        obs = Observation()
        settings_tree = obs.get_OSKAR_settings_tree()
        observations = Observation.create_observations_oskar_from_lists(
            settings_tree, CENTRAL_FREQS, CHANNEL_BANDWIDTHS, N_CHANNELS
        )

        for i, observation in enumerate(observations):
            start_frequency_hz = float(observation["observation"]["start_frequency_hz"])
            n_channels = int(observation["observation"]["num_channels"])
            freq_inc_hz = float(observation["observation"]["frequency_inc_hz"])

            assert start_frequency_hz == CENTRAL_FREQS[i]
            assert n_channels == N_CHANNELS
            assert freq_inc_hz == CHANNEL_BANDWIDTHS

        CENTRAL_FREQ = 200
        CHANNEL_BANDWIDTHS = [1, 2, 3, 4, 5]
        N_CHANNELS = [1, 2, 3, 4, 5]

        obs = Observation()
        settings_tree = obs.get_OSKAR_settings_tree()
        observations = Observation.create_observations_oskar_from_lists(
            settings_tree, CENTRAL_FREQ, CHANNEL_BANDWIDTHS, N_CHANNELS
        )

        for i, observation in enumerate(observations):
            start_frequency_hz = float(observation["observation"]["start_frequency_hz"])
            n_channels = int(observation["observation"]["num_channels"])
            freq_inc_hz = float(observation["observation"]["frequency_inc_hz"])

            assert start_frequency_hz == CENTRAL_FREQ
            assert n_channels == N_CHANNELS[i]
            assert freq_inc_hz == CHANNEL_BANDWIDTHS[i]

    def test_parallelization_by_channel(self):
        sky_data = np.array(
            [
                [20.0, -30.0, 100, 0, 0, 0, 1.0e9, -0.7, 0.0, 0, 0, 0],
                [20.0, -30.5, 100, 2, 2, 0, 1.0e9, -0.7, 0.0, 0, 50, 45],
                [20.5, -30.5, 100, 0, 0, 2, 1.0e9, -0.7, 0.0, 0, 10, -10],
            ]
        )
        sky = SkyModel(sky_data)
        telescope = Telescope.get_SKA1_MID_Telescope()

        simulation = InterferometerSimulation(
            channel_bandwidth_hz=1e6, time_average_sec=1
        )
        observation = Observation(
            phase_centre_ra_deg=20.0,
            start_date_and_time=datetime(2022, 9, 1, 23, 00, 00, 521489),
            length=timedelta(hours=0, minutes=0, seconds=1, milliseconds=0),
            phase_centre_dec_deg=-30.5,
            number_of_time_steps=10,
            start_frequency_hz=200 * 1e6,
            frequency_increment_hz=1e6,
            number_of_channels=1,
        )

        visibility = simulation.run_simulation(telescope, sky, observation)
        visibility.write_to_file("./result/system_noise/noise_vis.ms")

        imager = Imager(
            visibility, imaging_npixel=4096 * 1, imaging_cellsize=50
        )  # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        dirty.write_to_file("result/system_noise/noise_dirty.fits", overwrite=True)
        dirty.plot(title="Flux Density (Jy)")
