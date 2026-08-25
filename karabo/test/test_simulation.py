import os
import sys
import tempfile
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pytest
from numpy.typing import NDArray

from karabo.imaging.image import Image
from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.simulation.interferometer import InterferometerSimulation, format_timedelta
from karabo.simulation.observation import Observation, ObservationParallelized
from karabo.simulation.sample_simulation import run_sample_simulation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend
from karabo.test.util import create_compatible_dirty_image


def _assert_valid_multichannel_dirty_image(
    image: Image,
    *,
    nchan: int,
    npixel: int,
    phase_centre_deg: tuple[float, float],
    first_channel_frequency_hz: float,
    frequency_increment_hz: float,
) -> None:
    assert image.data.shape == (nchan, 1, npixel, npixel)
    assert np.isfinite(image.data).all()
    assert np.all(np.std(image.data, axis=(-2, -1)) > 0.0)
    assert np.nanmax(np.abs(image.data)) > 0.0
    assert image.header["CTYPE1"].startswith("RA")
    assert image.header["CTYPE2"].startswith("DEC")
    assert image.header["CTYPE3"] == "STOKES"
    assert image.header["CTYPE4"] == "FREQ"
    assert np.isclose(image.header["CRVAL1"], phase_centre_deg[0])
    assert np.isclose(image.header["CRVAL2"], phase_centre_deg[1])
    assert np.isclose(image.header["CRVAL4"], first_channel_frequency_hz)
    assert np.isclose(abs(image.header["CDELT4"]), frequency_increment_hz)


@pytest.mark.parametrize(
    "backend,telescope_name",
    [
        (SimulatorBackend.OSKAR, "SKA1MID"),
        (SimulatorBackend.SDP, "MID"),
    ],
)
def test_backend_simulations(
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

    # Run the test with low performance settings, i.e. few time steps
    # and number of channels. We only assert that an image was created and
    # that it has 4 dimensions. There is no assessment of the image quality.
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

    dirty = create_compatible_dirty_image(
        visibility,
        DirtyImagerConfig(
            imaging_npixel=1024,
            imaging_cellsize=3 / 180 * np.pi / 1024,
        ),
    )
    assert isinstance(dirty, Image)
    assert len(dirty.data.shape) == 4


@pytest.mark.parametrize(
    "backend,telescope_name",
    [
        (SimulatorBackend.OSKAR, "MeerKAT"),
        (SimulatorBackend.SDP, "MEERKAT+"),
    ],
)
def test_simulation_meerkat(
    backend: SimulatorBackend,
    telescope_name: str,
) -> None:
    """
    Simulate continuous emission and validate scientifically relevant image
    properties for the OSKAR and SDP backends.
    """
    # Parameter definition
    ra_deg = 20
    dec_deg = -30
    start_time = datetime(2000, 3, 20, 12, 6, 39)
    obs_length = timedelta(hours=3, minutes=5, seconds=0, milliseconds=0)
    start_freq = 1.5e9
    freq_bin = 1.0e7

    # Load test sky and MeerKAT telescope
    sky = SkyModel.sky_test()
    telescope = Telescope.constructor(telescope_name, backend=backend)

    # Simulating visibilities
    simulation = InterferometerSimulation(
        channel_bandwidth_hz=1.0e7,
        time_average_sec=8,
        ignore_w_components=True,
        uv_filter_max=3000,
        use_gpus=False,
        enable_power_pattern=True,
        use_dask=False,
    )
    observation = Observation(
        phase_centre_ra_deg=ra_deg,
        phase_centre_dec_deg=dec_deg,
        start_date_and_time=start_time,
        length=obs_length,
        number_of_time_steps=10,
        start_frequency_hz=start_freq,
        frequency_increment_hz=freq_bin,
        number_of_channels=3,
    )
    visibility = simulation.run_simulation(telescope, sky, observation, backend=backend)

    dirty = create_compatible_dirty_image(
        visibility,
        DirtyImagerConfig(
            imaging_npixel=1024,
            imaging_cellsize=3 / 180 * np.pi / 1024,
            combine_across_frequencies=False,
        ),
    )
    _assert_valid_multichannel_dirty_image(
        dirty,
        nchan=3,
        npixel=1024,
        phase_centre_deg=(ra_deg, dec_deg),
        first_channel_frequency_hz=(
            start_freq + freq_bin / 2 if backend is SimulatorBackend.SDP else start_freq
        ),
        frequency_increment_hz=freq_bin,
    )


def test_simulation_noise_meerkat() -> None:
    """
    Executes a simulation of continuous emission with noise and validates
    the output files.

    """
    # Parameter definition
    ra_deg = 20
    dec_deg = -30
    start_time = datetime(2000, 3, 20, 12, 6, 39)
    obs_length = timedelta(hours=3, minutes=5, seconds=0, milliseconds=0)
    start_freq = 1.5e9
    freq_bin = 1.0e7

    # Load test sky and MeerKAT telescope
    sky = SkyModel.sky_test()
    telescope = Telescope.constructor("MeerKAT")

    # Simulating visibilities
    simulation = InterferometerSimulation(
        channel_bandwidth_hz=1.0e7,
        time_average_sec=8,
        ignore_w_components=True,
        uv_filter_max=3000,
        use_gpus=False,
        enable_power_pattern=True,
        use_dask=False,
        noise_enable=True,
        noise_seed=1,
        noise_freq="Observation settings",
        noise_rms_start=10,
        noise_rms_end=10,
    )
    observation = Observation(
        phase_centre_ra_deg=ra_deg,
        phase_centre_dec_deg=dec_deg,
        start_date_and_time=start_time,
        length=obs_length,
        number_of_time_steps=10,
        start_frequency_hz=start_freq,
        frequency_increment_hz=freq_bin,
        number_of_channels=3,
    )
    visibility = simulation.run_simulation(telescope, sky, observation)

    dirty = create_compatible_dirty_image(
        visibility,
        DirtyImagerConfig(
            imaging_npixel=1024,
            imaging_cellsize=3 / 180 * np.pi / 1024,
            combine_across_frequencies=False,
        ),
    )
    _assert_valid_multichannel_dirty_image(
        dirty,
        nchan=3,
        npixel=1024,
        phase_centre_deg=(ra_deg, dec_deg),
        first_channel_frequency_hz=start_freq,
        frequency_increment_hz=freq_bin,
    )
    assert np.all(np.std(dirty.data, axis=(-2, -1)) > 0.01)


@pytest.mark.skip(
    reason="Current issue with Dask makes this test flaky. Test works locally."
)
def test_parallelization_by_observation() -> None:
    sky = SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=80e6)
    phase_center = [250, -80]
    CENTER_FREQUENCIES_HZ = [100e6, 101e6]
    CHANNEL_BANDWIDTHS_HZ = [1.0, 2.0]
    N_CHANNELS = [2, 4]

    sky = sky.filter_by_radius(0, 0.55, phase_center[0], phase_center[1])
    telescope = Telescope.constructor("ASKAP")

    simulation = InterferometerSimulation(channel_bandwidth_hz=1e6, time_average_sec=1)

    obs_parallelized = ObservationParallelized(
        center_frequencies_hz=CENTER_FREQUENCIES_HZ,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        channel_bandwidths_hz=CHANNEL_BANDWIDTHS_HZ,
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_time_steps=24,
        n_channels=N_CHANNELS,
    )

    visibilities = simulation.run_simulation(telescope, sky, obs_parallelized)

    for i, vis in enumerate(visibilities):
        dirty = create_compatible_dirty_image(
            vis,
            DirtyImagerConfig(
                imaging_npixel=512,
                imaging_cellsize=3.878509448876288e-05,
                combine_across_frequencies=False,
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            dirty.write_to_file(os.path.join(tmpdir, f"dirty_{i}.fits"), overwrite=True)
        assert dirty.header["CRVAL4"] == CENTER_FREQUENCIES_HZ[i]
        assert dirty.header["NAXIS4"] == N_CHANNELS[i]
        assert dirty.header["CDELT4"] == CHANNEL_BANDWIDTHS_HZ[i]


def test_run_sample_simulation() -> None:
    """
    Executes the ASKAP sample simulation, captures verbose output,
    validates the output files, and checks the sky model filtering.
    """

    # run simulation and capture output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    visibility, *_ = run_sample_simulation(verbose=True)

    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # verbose output content
    expected_messages = [
        "Getting Sky Survey",
        "Filtering Sky Model",
        "Setting Up Telescope",
        "Setting Up Observation",
        "Generating Visibilities",
    ]
    for message in expected_messages:
        assert message in output

    # Ensure the visibilities file path is valid
    assert os.path.exists(visibility.path)


@pytest.mark.parametrize(
    "td, expected",
    [
        (timedelta(hours=1, minutes=2, seconds=3.456), "01 h 02 m 03.456 s"),
        (timedelta(hours=0, minutes=0, seconds=0), "00 h 00 m 00.000 s"),
        (
            timedelta(hours=-2, minutes=-5, seconds=-30.5),
            "−02 h 05 m 30.500 s".replace("−", "-"),
        ),
        (timedelta(seconds=59.999), "00 h 00 m 59.999 s"),
        (timedelta(days=0, seconds=-0.001), "-00 h 00 m 00.001 s"),
    ],
)
def test_format_timedelta(td, expected):
    """
    Tests the function that formats a datetime.timedelta object as a
    readable time difference string.
    """
    assert format_timedelta(td) == expected
