import os
import tempfile
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits

from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.util import auto_choose_dirty_imager_from_vis
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


@pytest.mark.skip(reason="`run_simulation` is taking way too long")
def test_beam():
    sky = SkyModel()
    sky_data = np.zeros((81, 12))
    a = np.arange(-32, -27.5, 0.5)
    b = np.arange(18, 22.5, 0.5)
    dec_arr, ra_arr = np.meshgrid(a, b)
    sky_data[:, 0] = ra_arr.flatten()
    sky_data[:, 1] = dec_arr.flatten()
    sky_data[:, 2] = 10
    sky.add_point_sources(sky_data)
    telescope = Telescope.constructor("SKA1LOW")
    # telescope.centre_longitude = 3
    enable_array_beam = True
    # Remove beam if already present
    # ------------- Simulation Begins
    with tempfile.TemporaryDirectory() as tmpdir:
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=2e7,
            time_average_sec=1,
            noise_enable=False,
            enable_numerical_beam=True,
            station_type="Gaussian beam",
            enable_array_beam=enable_array_beam,
        )
        observation = Observation(
            mode="Tracking",
            phase_centre_ra_deg=20.0,
            start_date_and_time=datetime(2000, 1, 1, 13, 00, 00, 521489),
            length=timedelta(hours=0, minutes=0, seconds=1, milliseconds=0),
            phase_centre_dec_deg=-30.0,
            number_of_time_steps=1,
            start_frequency_hz=1.0e8,
            frequency_increment_hz=1e6,
            number_of_channels=1,
        )
        visibility = simulation.run_simulation(
            telescope,
            sky,
            observation,
            visibility_format="OSKAR_VIS",
            visibility_path=os.path.join(tmpdir, "beam_vis.vis"),
        )

        dirty_imager = auto_choose_dirty_imager_from_vis(
            visibility,
            DirtyImagerConfig(
                imaging_npixel=4096,
                # TODO Change cellsize to a more reasonable number
                # when test is re-enabled.
                # Suggestion:
                # With the 4096 number of pixels, this would correspond to a cellsize
                # of about (4.5*pi/180) / 4096 ~ 1.9e-5 radians/pixel,
                # if we want the image to just barely fit all sources,
                # or a slightly bigger cellsize to have some room at the edges.
                imaging_cellsize=50,
            ),
        )
        dirty = dirty_imager.create_dirty_image(visibility)

        dirty.write_to_file(
            os.path.join(tmpdir, "ska_low_array_vis.fits"), overwrite=True
        )
        aa = fits.open(os.path.join(tmpdir, "ska_low_array_vis.fits"))
        bb = fits.open(os.path.join(tmpdir, "ska_low_no_array_vis.fits"))
        print(
            np.nanmax(aa[0].data - bb[0].data),
            np.nanmax(aa[0].data),
            np.nanmax(bb[0].data),
        )
        diff = aa[0].data[0][0] - bb[0].data[0][0]
        _, ax = plt.subplots(1, 1)
        ax.imshow(diff, aspect="auto", origin="lower", vmin=-1.0e-1, vmax=1.0e-1)
        plt.show()
