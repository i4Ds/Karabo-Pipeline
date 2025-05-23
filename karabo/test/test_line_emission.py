import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from karabo.data.external_data import HISourcesSmallCatalogDownloadObject
from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.simulation.beam import (
    REFERENCE_FREQUENCY_HZ,
    REFERENCE_FWHM_DEGREES,
    gaussian_beam_fwhm_for_frequency,
    generate_gaussian_beam_data,
)
from karabo.simulation.interferometer import FilterUnits, InterferometerSimulation
from karabo.simulation.line_emission import CircleSkyRegion, line_emission_pipeline
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend
from karabo.util.file_handler import FileHandler


@pytest.mark.parametrize(
    "simulator_backend,telescope_name",
    [(SimulatorBackend.OSKAR, "SKA1MID"), (SimulatorBackend.RASCIL, "MID")],
)
def test_line_emission_pipeline(simulator_backend, telescope_name):
    print(f"Running test for {simulator_backend = }")
    telescope = Telescope.constructor(telescope_name, backend=simulator_backend)

    output_base_directory = Path(
        FileHandler().get_tmp_dir(
            prefix="line-emission-",
            purpose="Example line emission simulation",
        )
    )

    pointings = [
        CircleSkyRegion(
            radius=1 * u.deg, center=SkyCoord(ra=20, dec=-30, unit="deg", frame="icrs")
        ),
        CircleSkyRegion(
            radius=1 * u.deg,
            center=SkyCoord(ra=20, dec=-31.4, unit="deg", frame="icrs"),
        ),
        CircleSkyRegion(
            radius=1 * u.deg,
            center=SkyCoord(ra=21.4, dec=-30, unit="deg", frame="icrs"),
        ),
        CircleSkyRegion(
            radius=1 * u.deg,
            center=SkyCoord(ra=21.4, dec=-31.4, unit="deg", frame="icrs"),
        ),
    ]

    # The number of time steps is then determined as total_length / integration_time.
    observation_length = timedelta(seconds=10000)  # 14400 = 4hours
    integration_time = timedelta(seconds=10000)

    # Load catalog of sources
    catalog_path = HISourcesSmallCatalogDownloadObject().get()
    sky = SkyModel.get_sky_model_from_h5_to_xarray(
        path=catalog_path,
    )

    # Define observation channels and duration
    observation = Observation(
        start_date_and_time=datetime(2000, 3, 20, 12, 6, 39),
        length=observation_length,
        number_of_time_steps=int(
            observation_length.total_seconds() / integration_time.total_seconds()
        ),
        start_frequency_hz=7e8,
        frequency_increment_hz=8e7,
        number_of_channels=2,
    )

    # Instantiate interferometer
    # Leave time_average_sec as 10, since OSKAR examples use 10.
    # Not sure of the meaning of this parameter.
    interferometer = InterferometerSimulation(
        time_average_sec=10,
        ignore_w_components=True,
        uv_filter_max=10000,
        uv_filter_units=FilterUnits.Metres,
        use_gpus=True,
        station_type="Isotropic beam",
        gauss_beam_fwhm_deg=0,
        gauss_ref_freq_hz=0,
        use_dask=False,
    )

    # Imaging details
    npixels = 4096
    image_width_degrees = 2
    cellsize_radians = np.radians(image_width_degrees) / npixels
    dirty_imager_config = DirtyImagerConfig(
        imaging_npixel=npixels,
        imaging_cellsize=cellsize_radians,
    )

    visibilities, dirty_images = line_emission_pipeline(
        output_base_directory=output_base_directory,
        pointings=pointings,
        sky_model=sky,
        observation_details=observation,
        telescope=telescope,
        interferometer=interferometer,
        simulator_backend=simulator_backend,
        dirty_imager_config=dirty_imager_config,
    )

    assert len(visibilities) == observation.number_of_channels
    assert len(visibilities[0]) == len(pointings)

    assert len(dirty_images) == observation.number_of_channels
    assert len(dirty_images[0]) == len(pointings)


def test_compare_oskar_rascil_dirty_images():
    pointing = CircleSkyRegion(
        radius=1 * u.deg,
        center=SkyCoord(ra=20, dec=-31.4, unit="deg", frame="icrs"),
    )

    # The number of time steps is then determined as total_length / integration_time.
    observation_length = timedelta(seconds=10000)  # 14400 = 4hours
    integration_time = timedelta(seconds=10000)

    # Load catalog of sources
    catalog_path = HISourcesSmallCatalogDownloadObject().get()
    sky = SkyModel.get_sky_model_from_h5_to_xarray(
        path=catalog_path,
    )

    # Define observation channels and duration
    observation = Observation(
        start_date_and_time=datetime(2000, 3, 20, 12, 6, 39),
        length=observation_length,
        number_of_time_steps=int(
            observation_length.total_seconds() / integration_time.total_seconds()
        ),
        start_frequency_hz=7e8,
        frequency_increment_hz=8e7,
        number_of_channels=1,
    )

    # Instantiate interferometer
    # Leave time_average_sec as 10, since OSKAR examples use 10.
    # Not sure of the meaning of this parameter.
    interferometer = InterferometerSimulation(
        time_average_sec=10,
        ignore_w_components=True,
        uv_filter_max=10000,
        uv_filter_units=FilterUnits.Metres,
        use_gpus=True,
        station_type="Isotropic beam",
        gauss_beam_fwhm_deg=0,
        gauss_ref_freq_hz=0,
        use_dask=False,
    )

    # Imaging details
    npixels = 4096
    image_width_degrees = 2
    cellsize_radians = np.radians(image_width_degrees) / npixels
    dirty_imager_config = DirtyImagerConfig(
        imaging_npixel=npixels,
        imaging_cellsize=cellsize_radians,
    )

    backend_to_dirty_images = {}
    for simulator_backend, telescope_name in (
        (SimulatorBackend.OSKAR, "SKA1MID"),
        (SimulatorBackend.RASCIL, "MID"),
    ):
        telescope = Telescope.constructor(telescope_name, backend=simulator_backend)

        output_base_directory = Path(
            FileHandler().get_tmp_dir(
                prefix="line-emission-",
                purpose="Example line emission simulation",
            )
        )

        _, dirty_images = line_emission_pipeline(
            output_base_directory=output_base_directory,
            pointings=[pointing],
            sky_model=sky,
            observation_details=observation,
            telescope=telescope,
            interferometer=interferometer,
            simulator_backend=simulator_backend,
            dirty_imager_config=dirty_imager_config,
        )

        backend_to_dirty_images[simulator_backend] = dirty_images

    # Check that the dirty images are close to each other
    oskar_images = backend_to_dirty_images[SimulatorBackend.OSKAR]
    rascil_images = backend_to_dirty_images[SimulatorBackend.RASCIL]

    channel_index = 0  # This test uses only one frequency channel
    pointing_index = 0  # This tests uses only one pointing
    oskar_data = oskar_images[channel_index][pointing_index].data
    rascil_data = rascil_images[channel_index][pointing_index].data

    assert oskar_data.shape == rascil_data.shape
    assert oskar_data.shape == (1, 1, npixels, npixels)

    # Check pixel value differences are mostly within the tolerance
    # set as 10% of maximum pixel value
    tolerance = 0.1 * max(oskar_data.max(), rascil_data.max())
    count_very_different_pixels = np.sum(
        ~np.isclose(oskar_data, rascil_data, atol=tolerance)
    )
    count_pixels = np.prod(oskar_data.shape)  # Number of entries in data array
    fraction_very_different_pixels = count_very_different_pixels / count_pixels

    # Less than 1% of pixels are more different than this threshold
    assert fraction_very_different_pixels < 0.01


@pytest.mark.parametrize(
    "simulator_backend,telescope_name",
    [
        (SimulatorBackend.OSKAR, "SKA1MID"),
        (SimulatorBackend.RASCIL, "MID"),
    ],
)
def test_primary_beam_effects(simulator_backend, telescope_name):
    print(f"Running test for {simulator_backend = }")
    telescope = Telescope.constructor(telescope_name, backend=simulator_backend)

    pointings = [
        CircleSkyRegion(
            radius=1 * u.deg, center=SkyCoord(ra=20, dec=-30, unit="deg", frame="icrs")
        ),
        CircleSkyRegion(
            radius=1 * u.deg,
            center=SkyCoord(ra=20, dec=-31.4, unit="deg", frame="icrs"),
        ),
        CircleSkyRegion(
            radius=1 * u.deg,
            center=SkyCoord(ra=21.4, dec=-30, unit="deg", frame="icrs"),
        ),
        CircleSkyRegion(
            radius=1 * u.deg,
            center=SkyCoord(ra=21.4, dec=-31.4, unit="deg", frame="icrs"),
        ),
    ]

    # The number of time steps is then determined as total_length / integration_time.
    observation_length = timedelta(seconds=10000)  # 14400 = 4hours
    integration_time = timedelta(seconds=1000)

    # Load catalog of sources
    catalog_path = HISourcesSmallCatalogDownloadObject().get()
    sky = SkyModel.get_sky_model_from_h5_to_xarray(
        path=catalog_path,
    )

    # Define observation channels and duration
    observation = Observation(
        start_date_and_time=datetime(2000, 3, 20, 12, 6, 39),
        length=observation_length,
        number_of_time_steps=int(
            observation_length.total_seconds() / integration_time.total_seconds()
        ),
        start_frequency_hz=7e8,
        frequency_increment_hz=8e7,
        number_of_channels=2,
    )

    # Instantiate interferometer
    interferometer_without_primary_beam = InterferometerSimulation(
        time_average_sec=10,  # Not sure of the meaning of this parameter
        ignore_w_components=True,
        uv_filter_max=10000,
        uv_filter_units=FilterUnits.Metres,
        use_gpus=True,
        station_type="Isotropic beam",
        gauss_beam_fwhm_deg=0,
        gauss_ref_freq_hz=0,
        use_dask=False,
    )

    interferometer_with_primary_beam = InterferometerSimulation(
        time_average_sec=10,  # Not sure of the meaning of this parameter
        ignore_w_components=True,
        uv_filter_max=10000,
        uv_filter_units=FilterUnits.Metres,
        use_gpus=True,
        station_type="Gaussian beam",
        gauss_beam_fwhm_deg=REFERENCE_FWHM_DEGREES,
        gauss_ref_freq_hz=REFERENCE_FREQUENCY_HZ,
        use_dask=False,
    )

    # Imaging details
    npixels = 4096
    image_width_degrees = 2
    cellsize_radians = np.radians(image_width_degrees) / npixels
    dirty_imager_config = DirtyImagerConfig(
        imaging_npixel=npixels,
        imaging_cellsize=cellsize_radians,
    )

    # Compute frequency channels
    frequency_channel_starts = np.linspace(
        observation.start_frequency_hz,
        observation.start_frequency_hz
        + observation.frequency_increment_hz * observation.number_of_channels,
        num=observation.number_of_channels,
        endpoint=False,
    )

    # Generate primary beam data
    primary_beams = []
    for frequency in frequency_channel_starts:
        fwhm_degrees = gaussian_beam_fwhm_for_frequency(frequency)
        fwhm_pixels = (fwhm_degrees / np.degrees(dirty_imager_config.imaging_cellsize),)

        primary_beam = generate_gaussian_beam_data(
            fwhm_pixels=fwhm_pixels,
            x_size=dirty_imager_config.imaging_npixel,
            y_size=dirty_imager_config.imaging_npixel,
        )
        primary_beams.append(primary_beam)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_base_directory = Path(tmpdir)

        # Verify that we apply the correct primary beam corrections
        # for each frequency channels
        _, dirty_images_with_primary_beam = line_emission_pipeline(
            output_base_directory=output_base_directory,
            pointings=pointings,
            sky_model=sky,
            observation_details=observation,
            telescope=telescope,
            interferometer=interferometer_with_primary_beam,
            simulator_backend=simulator_backend,
            dirty_imager_config=dirty_imager_config,
            primary_beams=primary_beams,
        )

        _, dirty_images_without_primary_beam = line_emission_pipeline(
            output_base_directory=output_base_directory,
            pointings=pointings,
            sky_model=sky,
            observation_details=observation,
            telescope=telescope,
            interferometer=interferometer_without_primary_beam,
            simulator_backend=simulator_backend,
            dirty_imager_config=dirty_imager_config,
            primary_beams=None,
            should_perform_primary_beam_correction=False,
        )

        assert len(dirty_images_without_primary_beam) == observation.number_of_channels
        assert len(dirty_images_with_primary_beam) == observation.number_of_channels
