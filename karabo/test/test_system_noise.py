import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from karabo.imaging.imager_rascil import RascilDirtyImager, RascilDirtyImagerConfig
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


def test_basic(sky_data: NDArray[np.float64]):
    sky = SkyModel()
    sky.add_point_sources(sky_data)
    telescope = Telescope.constructor("SKA1MID")

    with tempfile.TemporaryDirectory() as tmpdir:
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

        visibility = simulation.run_simulation(
            telescope,
            sky,
            observation,
            visibility_format="MS",
            visibility_path=os.path.join(tmpdir, "noise_vis.ms"),
        )

        dirty_imager = RascilDirtyImager(
            RascilDirtyImagerConfig(
                imaging_npixel=4096,
                imaging_cellsize=50,
                combine_across_frequencies=False,
            )
        )
        dirty = dirty_imager.create_dirty_image(visibility)
        dirty.write_to_file(os.path.join(tmpdir, "noise_dirty.fits"), overwrite=True)
        dirty.plot(title="Flux Density (Jy)")
