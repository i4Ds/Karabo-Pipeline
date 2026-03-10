from datetime import datetime

import numpy as np
import pytest

from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_oskar import OskarDirtyImager
from karabo.imaging.imager_rascil import RascilDirtyImager
from karabo.imaging.imager_wsclean import WscleanDirtyImager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend
from karabo.test.util import get_compatible_dirty_imager


@pytest.fixture
def sky_model() -> SkyModel:
    sky = SkyModel()
    sky = SkyModel.get_random_poisson_disk_sky((220, -60), (260, -80), 1, 1, 1)
    return sky


@pytest.fixture
def observation() -> Observation:
    return Observation(
        start_frequency_hz=100e6,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        phase_centre_ra_deg=240,
        phase_centre_dec_deg=-70,
        number_of_time_steps=4,
        frequency_increment_hz=20e6,
        number_of_channels=4,
    )


@pytest.fixture
def simulation() -> InterferometerSimulation:
    return InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        time_average_sec=10,
    )


def test_get_compatible_dirty_imager_should_return_OskarDirtyImager(
    sky_model: SkyModel,
    observation: Observation,
    simulation: InterferometerSimulation,
) -> None:
    sim_backend = SimulatorBackend.OSKAR
    telescope = Telescope.constructor("SKA1MID", backend=sim_backend)
    telescope.centre_longitude = 3

    visibility = simulation.run_simulation(
        telescope,
        sky_model,
        observation,
        backend=sim_backend,
        visibility_format="OSKAR_VIS",
    )

    dirty_imager_config = DirtyImagerConfig(
        imaging_npixel=1024,
        imaging_cellsize=3 / 180 * np.pi / 1024,
        combine_across_frequencies=True,
    )

    dirty_imager = get_compatible_dirty_imager(visibility, dirty_imager_config)

    assert isinstance(dirty_imager, OskarDirtyImager)


def test_get_compatible_dirty_imager_should_return_WscleanDirtyImager(
    sky_model: SkyModel,
    observation: Observation,
    simulation: InterferometerSimulation,
) -> None:
    sim_backend = SimulatorBackend.OSKAR
    telescope = Telescope.constructor("SKA1MID", backend=sim_backend)
    telescope.centre_longitude = 3

    visibility = simulation.run_simulation(
        telescope,
        sky_model,
        observation,
        backend=sim_backend,
        visibility_format="MS",
    )

    dirty_imager_config = DirtyImagerConfig(
        imaging_npixel=1024,
        imaging_cellsize=3 / 180 * np.pi / 1024,
        combine_across_frequencies=True,
    )

    dirty_imager = get_compatible_dirty_imager(visibility, dirty_imager_config)

    assert isinstance(dirty_imager, WscleanDirtyImager)


def test_get_compatible_dirty_imager_should_return_RascilDirtyImager(
    sky_model: SkyModel,
    observation: Observation,
    simulation: InterferometerSimulation,
) -> None:
    sim_backend = SimulatorBackend.RASCIL
    telescope = Telescope.constructor("MID", backend=sim_backend)
    telescope.centre_longitude = 3

    visibility = simulation.run_simulation(
        telescope,
        sky_model,
        observation,
        backend=sim_backend,
        primary_beam=None,
        visibility_format="MS",
    )

    dirty_imager_config = DirtyImagerConfig(
        imaging_npixel=1024,
        imaging_cellsize=3 / 180 * np.pi / 1024,
        combine_across_frequencies=False,
    )

    dirty_imager = get_compatible_dirty_imager(visibility, dirty_imager_config)

    assert isinstance(dirty_imager, RascilDirtyImager)
