from datetime import datetime

import numpy as np
import pytest
from numpy.typing import NDArray

from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import VisibilityFormat
from karabo.simulator_backend import SimulatorBackend
from karabo.test.util import get_compatible_dirty_imager


@pytest.mark.parametrize(
    "backend,telescope_name,visibility_format,combine_across_frequencies",
    [
        (SimulatorBackend.OSKAR, "SKA1MID", "OSKAR_VIS", True),
        (SimulatorBackend.OSKAR, "SKA1MID", "MS", True),
        (SimulatorBackend.RASCIL, "MID", "MS", False),
    ],
)
def test_get_compatible_dirty_imager(
    sky_data: NDArray[np.float64],
    backend: SimulatorBackend,
    telescope_name: str,
    visibility_format: VisibilityFormat,
    combine_across_frequencies: bool,
) -> None:
    sky = SkyModel()
    sky.add_point_sources(sky_data)
    sky = SkyModel.get_random_poisson_disk_sky((220, -60), (260, -80), 1, 1, 1)
    sky.explore_sky([240, -70], s=10)
    telescope = Telescope.constructor(telescope_name, backend=backend)
    telescope.centre_longitude = 3

    simulation = InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        time_average_sec=10,
    )
    observation = Observation(
        start_frequency_hz=100e6,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        phase_centre_ra_deg=240,
        phase_centre_dec_deg=-70,
        number_of_time_steps=4,
        frequency_increment_hz=20e6,
        number_of_channels=4,
    )

    visibility = simulation.run_simulation(
        telescope,
        sky,
        observation,
        backend=backend,
        visibility_format=visibility_format,
    )

    dirty_imager_config = DirtyImagerConfig(
        imaging_npixel=1024,
        imaging_cellsize=3 / 180 * np.pi / 1024,
        combine_across_frequencies=combine_across_frequencies,
    )

    dirty_imager = get_compatible_dirty_imager(visibility, dirty_imager_config)
    dirty_image = dirty_imager.create_dirty_image(visibility)
    assert dirty_image.data.ndim == 4
