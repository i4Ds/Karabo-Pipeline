import os
import unittest
from datetime import datetime, timedelta

import numpy as np
from ARatmospy.ArScreens import ArScreens
from astropy.io import fits
from astropy.wcs import WCS

from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


class TestSystemNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists("result/system_noise"):
            os.makedirs("result/system_noise")

    def sim_ion(
        self,
        screen_width_metres,
        r0,
        bmax,
        sampling,
        speed,
        rate,
        alpha_mag,
        num_times,
        frequency,
        fits_filename,
    ):
        """
        Example from OSKAR
        """
        m = int(bmax / sampling)  # Pixels per sub-aperture (200).
        n = int(screen_width_metres / bmax)  # Sub-apertures across the screen (10).
        num_pix = n * m
        pscale = screen_width_metres / (n * m)  # Pixel scale (100 m/pixel).
        print("Number of pixels %d, pixel size %.3f m" % (num_pix, pscale))
        print("Field of view %.1f (m)" % (num_pix * pscale))
        # Parameters for each layer.
        # (scale size [m], speed [m/s], direction [deg], layer height [m]).
        layer_params = np.array(
            [(r0, speed, 60.0, 300e3), (r0, speed / 2.0, -30.0, 310e3)]
        )
        my_screens = ArScreens(n, m, pscale, rate, layer_params, alpha_mag)
        my_screens.run(num_times)
        phase2tec = -frequency / 8.44797245e9
        w = WCS(naxis=4)
        w.naxis = 4
        w.wcs.cdelt = [pscale, pscale, 1.0 / rate, 1.0]
        w.wcs.crpix = [num_pix // 2 + 1, num_pix // 2 + 1, num_times // 2 + 1, 1.0]
        w.wcs.ctype = ["XX", "YY", "TIME", "FREQ"]
        w.wcs.crval = [0.0, 0.0, 0.0, frequency]
        data = np.zeros([1, num_times, num_pix, num_pix])
        for layer in range(len(my_screens.screens)):
            for i, screen in enumerate(my_screens.screens[layer]):
                data[:, i, ...] += phase2tec * screen[np.newaxis, ...]
        fits.writeto(
            filename=fits_filename, data=data, header=w.to_header(), overwrite=True
        )
        return my_screens

    def test_ionosphere(self):
        # ----------- Ionopsheric Simulations
        screen_width_metres = 200e3
        r0 = 5e3  # Scale size (5 km).
        bmax = 20e3  # 20 km sub-aperture size.
        sampling = 100.0  # 100 m/pixel.
        speed = 150e3 / 3600.0  # 150 km/h in m/s.
        rate = 1.0 / 60.0  # The inverse frame rate (1 per minute).
        alpha_mag = 0.999  # Evolve screen slowly.
        num_times = 1  # one minute (64MB RAM per minute)
        frequency = 1.0e8
        fits_filename = "result/test_screen_60s.fits"
        self.sim_ion(
            screen_width_metres,
            r0,
            bmax,
            sampling,
            speed,
            rate,
            alpha_mag,
            num_times,
            frequency,
            fits_filename,
        )
        # ---------- Simulation with Screen
        sky = SkyModel()
        sky_data = np.array(
            [
                [20.0, -30.0, 100, 0, 0, 0, 1.0e8, 0, 0.0, 0, 0, 0],
                [20.0, -30.5, 100, 2, 2, 0, 1.0e8, 0, 0.0, 0, 50, 45],
                [20.5, -30.5, 100, 0, 0, 2, 1.0e8, 0, 0.0, 0, 10, -10],
            ]
        )
        sky.add_point_sources(sky_data)
        # sky = SkyModel.get_random_poisson_disk_sky((220, -60), (260, -80), 1, 1, 1)
        # sky.explore_sky([240, -70])
        telescope = Telescope.get_SKA1_LOW_Telescope()
        # telescope.centre_longitude = 3
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=1e6,
            time_average_sec=1,
            noise_enable=False,
            station_type="Isotropic",
            ionosphere_fits_path=fits_filename,
            ionosphere_screen_type="External",
            ionosphere_screen_height_km=r0,
            ionosphere_screen_pixel_size_m=sampling,
            ionosphere_isoplanatic_screen=True,
        )
        observation = Observation(
            phase_centre_ra_deg=20.0,
            start_date_and_time=datetime(2022, 9, 1, 3, 00, 00, 521489),
            length=timedelta(hours=0, minutes=10, seconds=1, milliseconds=0),
            phase_centre_dec_deg=-30.5,
            number_of_time_steps=5,
            start_frequency_hz=frequency,
            frequency_increment_hz=1e6,
            number_of_channels=1,
        )
        visibility = simulation.run_simulation(telescope, sky, observation)
        visibility.write_to_file("./result/test_ion.ms")
        imager = Imager(
            visibility, imaging_npixel=2048 * 1, imaging_cellsize=5.0e-5
        )  # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        dirty.write_to_file("result/test_ion.fits", overwrite=True)
        dirty.plot(title="Flux Density (Jy)")
