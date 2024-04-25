from datetime import datetime, timedelta
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from karabo.data.external_data import HISourcesSmallCatalogDownloadObject
from karabo.imaging.image import Image, ImageMosaicker
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

    # Image details
    npixels = 4096
    image_width_degrees = 2
    cellsize_radians = np.radians(image_width_degrees) / npixels

    beam_type = "Isotropic beam"
    gaussian_fwhm = 0
    gaussian_ref_freq = 0

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
        station_type=beam_type,
        gauss_beam_fwhm_deg=gaussian_fwhm,
        gauss_ref_freq_hz=gaussian_ref_freq,
        use_dask=False,
    )

    visibilities, dirty_images = line_emission_pipeline(
        output_base_directory=output_base_directory,
        simulator_backend=simulator_backend,
        imaging_backend=None,  # Cause pipeline to use same backend as simulator_backend
        pointings=pointings,
        sky_model=sky,
        observation_details=observation,
        telescope=telescope,
        interferometer=interferometer,
        image_npixels=npixels,
        image_cellsize_radians=cellsize_radians,
    )

    # Create mosaics of pointings for each frequency channel
    mosaicker = ImageMosaicker()

    mosaics = []
    for index_freq in range(observation.number_of_channels):
        mosaic, _ = mosaicker.mosaic(dirty_images[index_freq])
        mosaics.append(mosaic)

    # Add all mosaics across frequency channels to create one final mosaic image
    Image(
        data=sum(m.data for m in mosaics),
        header=mosaics[0].header,
    )
