from datetime import datetime

import numpy as np
import pytest
from numpy.typing import NDArray
from ska_sdp_datamodels.visibility import Visibility as RASCILVisibility
from typing_extensions import assert_never

from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_oskar import OskarDirtyImager
from karabo.imaging.imager_rascil import RascilDirtyImager
from karabo.imaging.util import (
    auto_choose_dirty_imager_from_sim,
    auto_choose_dirty_imager_from_vis,
)
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.simulator_backend import SimulatorBackend


@pytest.mark.parametrize(
    "backend,telescope_name",
    [
        (SimulatorBackend.OSKAR, "SKA1MID"),
        (SimulatorBackend.RASCIL, "MID"),
    ],
)
def test_auto_choose_dirty_imager(
    sky_data: NDArray[np.float64], backend: SimulatorBackend, telescope_name: str
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

    visibility = simulation.run_simulation(telescope, sky, observation, backend=backend)

    dirty_imager_config = DirtyImagerConfig(
        imaging_npixel=1024,
        imaging_cellsize=3 / 180 * np.pi / 1024,
    )

    dirty_imager = auto_choose_dirty_imager_from_vis(visibility, dirty_imager_config)
    if backend is SimulatorBackend.OSKAR:
        assert isinstance(visibility, Visibility)
        assert isinstance(dirty_imager, OskarDirtyImager)
    elif backend is SimulatorBackend.RASCIL:
        assert isinstance(visibility, RASCILVisibility)
        assert isinstance(dirty_imager, RascilDirtyImager)
    else:
        assert_never(backend)

    dirty_imager = auto_choose_dirty_imager_from_sim(backend, dirty_imager_config)
    if backend is SimulatorBackend.OSKAR:
        assert isinstance(dirty_imager, OskarDirtyImager)
    elif backend is SimulatorBackend.RASCIL:
        assert isinstance(dirty_imager, RascilDirtyImager)
    else:
        assert_never(backend)
