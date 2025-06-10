import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import ObservationLong
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.test.conftest import TFiles
from karabo.test.util import get_compatible_dirty_imager


def test_long_observations(tobject: TFiles, sky_data: NDArray[np.float64]):
    # skips `input` during unit tests if using `karabo.util.data_util.input_wrapper`
    os.environ["SKIP_INPUT"] = str(True)
    number_of_days = 3
    hours_per_day = 4
    enable_array_beam = False
    with tempfile.TemporaryDirectory() as tmpdir:
        combined_ms_filepath = os.path.join(tmpdir, "combined_vis.ms")
        sky = SkyModel()
        sky.add_point_sources(sky_data)
        telescope = Telescope.constructor("MeerKAT")
        observation_long = ObservationLong(
            mode="Tracking",
            phase_centre_ra_deg=20.0,
            start_date_and_time=datetime(2000, 1, 1, 11, 00, 00, 521489),
            length=timedelta(hours=hours_per_day, minutes=0, seconds=0, milliseconds=0),
            phase_centre_dec_deg=-30.0,
            number_of_time_steps=7,
            start_frequency_hz=1.0e9,
            frequency_increment_hz=1e6,
            number_of_channels=3,
            number_of_days=number_of_days,
        )
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=2e7,
            time_average_sec=7,
            noise_enable=False,
            noise_seed="time",
            noise_freq="Range",
            noise_rms="Range",
            noise_start_freq=1.0e9,
            noise_inc_freq=1.0e6,
            noise_number_freq=1,
            noise_rms_start=0.1,
            noise_rms_end=1,
            enable_numerical_beam=enable_array_beam,
            enable_array_beam=enable_array_beam,
        )
        # -------- Iterate over days
        visibility = simulation.run_simulation(
            telescope=telescope,
            sky=sky,
            observation=observation_long,
            visibility_format="MS",
            visibility_path=combined_ms_filepath,
        )

        dirty_imager = get_compatible_dirty_imager(
            visibility,
            DirtyImagerConfig(
                imaging_npixel=4096,
                imaging_cellsize=1.0e-5,
            ),
        )
        dirty_imager.create_dirty_image(visibility)
