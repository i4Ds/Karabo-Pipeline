import os
import unittest
from datetime import timedelta, datetime
import numpy as np
from karabo.imaging.imager import Imager
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.interferometer import InterferometerSimulation


class TestSystemNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists("result/system_noise"):
            os.makedirs("result/system_noise")

    def disabled_test_basic(self):
        sky = SkyModel()
        sky_data = np.array(
            [
                [20.0, -30.0, 100, 0, 0, 0, 1.0e9, -0.7, 0.0, 0, 0, 0],
                [20.0, -30.5, 100, 2, 2, 0, 1.0e9, -0.7, 0.0, 0, 50, 45],
                [20.5, -30.5, 100, 0, 0, 2, 1.0e9, -0.7, 0.0, 0, 10, -10],
            ]
        )
        sky.add_point_sources(sky_data)
        # sky = SkyModel.get_random_poisson_disk_sky((220, -60), (260, -80), 1, 1, 1)
        # sky.plot_sky((240, -70))
        telescope = Telescope.get_SKA1_MID_Telescope()
        # telescope.centre_longitude = 3

        simulation = InterferometerSimulation(
            channel_bandwidth_hz=1e6,
            time_average_sec=1,
            noise_enable=True,
            noise_seed="time",
            noise_freq="Range",
            noise_rms="Range",
            noise_start_freq=1.0e9,
            noise_inc_freq=1.0e8,
            noise_number_freq=24,
            noise_rms_start=5000,
            noise_rms_end=10000,
        )
        observation = Observation(
            phase_centre_ra_deg=20.0,
            start_date_and_time=datetime(2022, 9, 1, 23, 00, 00, 521489),
            length=timedelta(hours=0, minutes=0, seconds=1, milliseconds=0),
            phase_centre_dec_deg=-30.5,
            number_of_time_steps=1,
            start_frequency_hz=1.0e9,
            frequency_increment_hz=1e6,
            number_of_channels=1,
        )

        visibility = simulation.run_simulation(telescope, sky, observation)
        visibility.write_to_file("./result/system_noise/noise_vis.ms")

        imager = Imager(
            visibility, imaging_npixel=4096 * 1, imaging_cellsize=50
        )  # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        dirty.write_to_file("result/system_noise/noise_dirty.fits")
        dirty.plot(title="Flux Density (Jy)")
