import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from numpy.typing import NDArray

from karabo.data.external_data import (
    SingleFileDownloadObject,
    cscs_karabo_public_testing_base_url,
)
from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation, ObservationParallized
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


# DownloadObject instances used to download different golden files:
# - FITS file of the test continuous emission simulation of MeerKAT.
# - FITS files of the test continuous emission simulation with noise of MeerKAT.
@pytest.fixture
def continuous_fits_filename() -> str:
    return "test_continuous_emission.fits"


@pytest.fixture
def continuous_noise_fits_filename() -> str:
    return "test_continuous_emission_noise.fits"


@pytest.fixture
def continuous_fits_downloader(
    continuous_fits_filename: str,
) -> SingleFileDownloadObject:
    return SingleFileDownloadObject(
        remote_file_path=continuous_fits_filename,
        remote_base_url=cscs_karabo_public_testing_base_url,
    )


@pytest.fixture
def continuous_noise_fits_downloader(
    continuous_noise_fits_filename: str,
) -> SingleFileDownloadObject:
    return SingleFileDownloadObject(
        remote_file_path=continuous_noise_fits_filename,
        remote_base_url=cscs_karabo_public_testing_base_url,
    )


def test_oskar_simulation_basic(sky_data: NDArray[np.float64]) -> None:
    # Tests oskar simulation. Should use GPU if available and if not, CPU.
    sky = SkyModel()
    sky.add_point_sources(sky_data)
    sky = SkyModel.get_random_poisson_disk_sky((220, -60), (260, -80), 1, 1, 1)
    sky.explore_sky([240, -70], s=10)
    telescope = Telescope.constructor("EXAMPLE")
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


def test_simulation_meerkat(
    continuous_fits_filename: str, continuous_fits_downloader: SingleFileDownloadObject
) -> None:
    """
    Executes a simulation of continuous emission and validates the output files.

    Args:
        continuous_fits_filename:
            Name of FITS file containing the simulated dirty image.
    """
    # Download golden files for comparison
    golden_continuous_fits_path = continuous_fits_downloader.get()

    # Parameter defintion
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

    # We use the Imager to check the simulation
    imager = Imager(
        visibility,
        imaging_npixel=1024,
        imaging_cellsize=3 / 180 * np.pi / 1024,
    )
    dirty = imager.get_dirty_image()
    # Temporary directory containging output files for validation
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)
        continuous_fits_path = outpath / "test_continuous_emission.fits"
        dirty.write_to_file(str(continuous_fits_path), overwrite=True)

        # Verify mosaic fits
        continuous_fits_data, continuous_fits_header = fits.getdata(
            continuous_fits_path, ext=0, header=True
        )
        golden_continuous_fits_data, golden_continuous_fits_header = fits.getdata(
            golden_continuous_fits_path, ext=0, header=True
        )

        # Check FITS data is close to goldenfile
        assert np.allclose(
            golden_continuous_fits_data, continuous_fits_data, equal_nan=True
        )

        # Check that headers contain the same keys
        assert set(golden_continuous_fits_header.keys()) == set(
            continuous_fits_header.keys()
        )


def test_simulation_noise_meerkat(
    continuous_noise_fits_filename: str,
    continuous_noise_fits_downloader: SingleFileDownloadObject,
) -> None:
    """
    Executes a simulation of continuous emission with noise and validates
    the output files.

    Args:
        continuous_noise_fits_filename:
            Name of FITS file containing the simulated dirty image.
    """
    # Download golden files for comparison
    golden_continuous_noise_fits_path = continuous_noise_fits_downloader.get()

    # Parameter defintion
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

    # We use the Imager to check the simulation
    imager = Imager(
        visibility,
        imaging_npixel=1024,
        imaging_cellsize=3 / 180 * np.pi / 1024,
    )
    dirty = imager.get_dirty_image()
    # Temporary directory containging output files for validation
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)
        continuous_noise_fits_path = outpath / "test_continuous_emission_noise.fits"
        dirty.write_to_file(str(continuous_noise_fits_path), overwrite=True)

        # Verify mosaic fits
        continuous_noise_fits_data, continuous_noise_fits_header = fits.getdata(
            continuous_noise_fits_path, ext=0, header=True
        )
        (
            golden_continuous_noise_fits_data,
            golden_continuous_noise_fits_header,
        ) = fits.getdata(golden_continuous_noise_fits_path, ext=0, header=True)

        # Check FITS data is close to goldenfile
        assert np.allclose(
            golden_continuous_noise_fits_data,
            continuous_noise_fits_data,
            equal_nan=True,
        )

        # Check that headers contain the same keys
        assert set(golden_continuous_noise_fits_header.keys()) == set(
            continuous_noise_fits_header.keys()
        )


def test_parallelization_by_observation() -> None:
    sky = SkyModel.get_GLEAM_Sky([76])
    phase_center = [250, -80]
    CENTER_FREQUENCIES_HZ = [100e6, 101e6]
    CHANNEL_BANDWIDTHS_HZ = [1.0, 2.0]
    N_CHANNELS = [2, 4]

    sky = sky.filter_by_radius(0, 0.55, phase_center[0], phase_center[1])
    telescope = Telescope.constructor("ASKAP")

    simulation = InterferometerSimulation(channel_bandwidth_hz=1e6, time_average_sec=1)

    obs_parallized = ObservationParallized(
        center_frequencies_hz=CENTER_FREQUENCIES_HZ,
        channel_bandwidths_hz=CHANNEL_BANDWIDTHS_HZ,
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_time_steps=24,
        n_channels=N_CHANNELS,
    )

    visibilities = simulation.run_simulation(telescope, sky, obs_parallized)

    for i, vis in enumerate(visibilities):
        imager = Imager(
            vis, imaging_npixel=512, imaging_cellsize=3.878509448876288e-05
        )  # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        with tempfile.TemporaryDirectory() as tmpdir:
            dirty.write_to_file(os.path.join(tmpdir, f"dirty_{i}.fits"), overwrite=True)
        assert dirty.header["CRVAL4"] == CENTER_FREQUENCIES_HZ[i]
        assert dirty.header["NAXIS4"] == N_CHANNELS[i]
        assert dirty.header["CDELT4"] == CHANNEL_BANDWIDTHS_HZ[i]
