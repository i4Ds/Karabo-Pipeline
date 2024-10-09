from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from karabo.data.external_data import HISourcesSmallCatalogDownloadObject
from karabo.imaging.image import ImageMosaicker
from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.simulation.interferometer import FilterUnits, InterferometerSimulation
from karabo.simulation.line_emission import CircleSkyRegion, line_emission_pipeline
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend
from karabo.util.file_handler import FileHandler

if __name__ == "__main__":
    simulator_backend = SimulatorBackend.RASCIL

    if simulator_backend == SimulatorBackend.OSKAR:
        telescope = Telescope.constructor("SKA1MID", backend=simulator_backend)
    elif simulator_backend == SimulatorBackend.RASCIL:
        telescope = Telescope.constructor("MID", backend=simulator_backend)

    # Configuration parameters
    should_apply_primary_beam = False

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
    ]

    # The number of time steps is then determined as total_length / integration_time.
    observation_length = timedelta(seconds=10000)  # 14400 = 4hours
    integration_time = timedelta(seconds=10000)

    # Create interferometer simulation
    beam_type: Literal["Gaussian beam", "Isotropic beam"]
    if should_apply_primary_beam:
        beam_type = "Gaussian beam"
        # Options: "Aperture array", "Isotropic beam", "Gaussian beam", "VLA (PBCOR)"
        gaussian_fwhm = 50  # Degrees
        gaussian_ref_freq = 8e8  # Hz
    else:
        beam_type = "Isotropic beam"
        gaussian_fwhm = 0
        gaussian_ref_freq = 0

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
        station_type=beam_type,
        gauss_beam_fwhm_deg=gaussian_fwhm,
        gauss_ref_freq_hz=gaussian_ref_freq,
        use_dask=False,
    )

    # Imaging details
    npixels = 2048
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

    # Create mosaics of pointings for each frequency channel
    print("Creating mosaic of images for each frequency channel")
    mosaicker = ImageMosaicker()

    mosaics = []
    for index_freq in range(observation.number_of_channels):
        mosaic, _ = mosaicker.mosaic(dirty_images[index_freq])
        mosaics.append(mosaic)
