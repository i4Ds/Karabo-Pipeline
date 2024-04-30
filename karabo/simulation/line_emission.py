from collections import namedtuple
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import astropy.units as u
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.visibility import Visibility as RASCILVisibility

from karabo.imaging.image import Image, ImageMosaicker
from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import FilterUnits, InterferometerSimulation
from karabo.simulation.line_emission_helpers import convert_frequency_to_z
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.simulator_backend import SimulatorBackend

CircleSkyRegion = namedtuple("CircleSkyRegion", ["center", "radius"])


def line_emission_pipeline(
    output_base_directory: Union[Path, str],
    simulator_backend: SimulatorBackend,
    imaging_backend: Optional[SimulatorBackend],
    pointings: List[CircleSkyRegion],
    sky_model: SkyModel,
    observation_details: Observation,
    telescope: Telescope,
    interferometer: InterferometerSimulation,
    image_npixels: int,
    image_cellsize_radians: float,
) -> Tuple[List[List[Union[Visibility, RASCILVisibility]]], List[List[Image]]]:
    """Perform a line emission simulation, to compute visibilities and dirty images.
    A line emission simulation involves assuming every source in the input SkyModel
    only emits within one frequency channel.

    If requested, combine the produced dirty images into a mosaic.
    """
    print(f"Selected backend: {simulator_backend}")

    output_base_directory = Path(output_base_directory)
    # If output filepath does not exist, mkdir
    print(f"Creating {output_base_directory} directory if it does not exist yet.")
    output_base_directory.mkdir(exist_ok=True, parents=True)

    # Compute frequency channels
    frequency_channel_starts = np.linspace(
        observation_details.start_frequency_hz,
        observation_details.start_frequency_hz
        + observation_details.frequency_increment_hz
        * observation_details.number_of_channels,
        num=observation_details.number_of_channels,
        endpoint=False,
    )

    # Create observation instance to be used for each pointing and each channel
    observation = deepcopy(observation_details)

    print("Computing visibilities...")

    # Loop through pointings
    visibilities: List[List[Union[Visibility, RASCILVisibility]]] = []
    for index_freq, frequency_start in enumerate(frequency_channel_starts):
        print(f"Processing frequency channel {index_freq}...")
        visibilities.append([])
        for index_p, p in enumerate(pointings):
            print(f"Processing pointing {index_p}...")
            center = p.center
            radius = p.radius.to(u.deg).value
            # Create observation details
            observation.phase_centre_ra_deg = center.ra.deg
            observation.phase_centre_dec_deg = center.dec.deg
            observation.number_of_channels = 1  # For line emission
            observation.start_frequency_hz = frequency_start

            # Filter sky based on pointing and on frequency channel
            filtered_sky = sky_model.filter_by_radius_euclidean_flat_approximation(
                inner_radius_deg=0,
                outer_radius_deg=radius,
                ra0_deg=center.ra.deg,
                dec0_deg=center.dec.deg,
            )
            z_min = convert_frequency_to_z(
                frequency_start + observation.frequency_increment_hz
            )
            z_max = convert_frequency_to_z(frequency_start)

            filtered_sky = filtered_sky.filter_by_column(
                col_idx=13,
                min_val=z_min,
                max_val=z_max,
            )

            assert (
                filtered_sky.num_sources > 0
            ), f"""For frequency channel {index_freq}
                    and pointing {index_p}, there are 0 sources in the sky model.
                    Setting visibility to None, and skipping analysis."""

            interferometer.vis_path = (
                f"{output_base_directory}/visibilities_f{index_freq}_p{index_p}"
            )

            vis = interferometer.run_simulation(
                telescope=telescope,
                sky=filtered_sky,
                observation=observation,
                backend=simulator_backend,
            )

            visibilities[-1].append(vis)

    assert len(visibilities) == observation_details.number_of_channels
    assert len(visibilities[0]) == len(pointings)

    print("Creating dirty images from visibilities...")

    dirty_images: List[List[Image]] = []
    for index_freq, _ in enumerate(frequency_channel_starts):
        print(f"Processing frequency channel {index_freq}...")
        dirty_images.append([])
        for index_p, _ in enumerate(pointings):
            print(f"Processing pointing {index_p}...")
            vis = visibilities[index_freq][index_p]

            imager = Imager(
                vis,
                imaging_npixel=image_npixels,
                imaging_cellsize=image_cellsize_radians,
            )

            dirty = imager.get_dirty_image(
                fits_path=str(
                    output_base_directory
                    / f"dirty_{'OSKAR' if simulator_backend is SimulatorBackend.OSKAR else 'RASCIL'}_{index_p}.fits"  # noqa: E501
                ),
                imaging_backend=imaging_backend,
                combine_across_frequencies=True,
            )

            dirty_images[-1].append(dirty)

            # dirty.data.shape meaning:
            # (frequency channels, polarisations, pixels_x, pixels_y)
            assert dirty.data.ndim == 4
            # I.e. only one frequency channel,
            # since we are performing a line emission analysis
            assert dirty.data.shape[0] == 1

    assert len(dirty_images) == observation_details.number_of_channels
    assert len(dirty_images[0]) == len(
        pointings
    ), f"{len(dirty_images[0])}, {len(pointings)}"

    return visibilities, dirty_images


if __name__ == "__main__":
    # This executes an example line emission pipeline
    # with a sample sky and example simulation parameters
    from karabo.data.external_data import HISourcesSmallCatalogDownloadObject
    from karabo.util.file_handler import FileHandler

    try:
        matplotlib.use("tkagg")
    except Exception as e:
        print(e)

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

    for index_freq in range(observation.number_of_channels):
        for index_p, p in enumerate(pointings):
            dirty = dirty_images[index_freq][index_p]

            dirty.plot(
                block=True,
                vmin=0,
                vmax=2e-7,
                title=f"Dirty image for pointing {index_p} and channel {index_freq}",
            )

    # Overlay SkyModel onto dirty image
    dirty_images[0][0].plot_side_by_side_with_skymodel(
        sky=sky,
        block=True,
        vmin_sky=0,
        vmax_sky=2e-6,
        vmin_image=0,
        vmax_image=2e-7,
    )

    dirty_images[0][0].overplot_with_skymodel(
        sky=sky,
        block=True,
        vmin_image=0,
        vmax_image=2e-7,
    )

    # Create mosaics of pointings for each frequency channel
    print("Creating mosaic of images for each frequency channel")
    mosaicker = ImageMosaicker()

    mosaics = []
    for i in range(observation.number_of_channels):
        mosaic, footprint = mosaicker.mosaic(dirty_images[i])
        mosaics.append(mosaic)

        mosaic.plot(
            filename=str(output_base_directory / f"mosaic_{i}.png"),
            block=True,
            vmin=0,
            vmax=2e-7,
            title=f"Mosaic for channel {i}",
        )

    # Add all mosaics across frequency channels to create one final mosaic image
    summed_mosaic = Image(
        data=np.sum([m.data for m in mosaics]),
        header=mosaics[0].header,
    )
    summed_mosaic.plot(
        filename=str(output_base_directory / "summed_mosaic.png"),
        block=True,
        vmin=0,
        vmax=2e-7,
        title="Summed mosaic across channels",
    )

    # Overlay SkyModel onto dirty image
    summed_mosaic.overplot_with_skymodel(
        sky=sky,
        filename=str(output_base_directory / "summed_mosaic_sources_overlay.png"),
        block=True,
        vmin_image=0,
        vmax_image=2e-7,
    )
