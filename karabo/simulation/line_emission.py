import warnings
from collections import namedtuple
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import NDArray

from karabo.imaging.image import Image, ImageMosaicker
from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import (
    FilterUnits,
    InterferometerSimulation,
    StationTypeType,
)
from karabo.simulation.line_emission_helpers import convert_frequency_to_z
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.simulator_backend import SimulatorBackend
from karabo.util._types import DirPathType, FilePathType, IntFloat, NPFloatLikeStrict
from karabo.util.plotting_util import get_slices

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
):
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
    visibilities = []
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

            # If the filtered sky has no remaining sources,
            # set the visibility as None and skip this evaluation
            # TODO should be able to create an empty visibility
            # instead of setting to None
            if filtered_sky.num_sources == 0:
                warnings.warn(
                    f"""For frequency channel {index_freq},
                    pointing {index_p}, there are 0 sources in the sky model.
                    Setting visibility to None, and skipping analysis."""
                )
                visibilities[-1].append(None)
                continue

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

    dirty_images = []
    for index_freq, _ in enumerate(frequency_channel_starts):
        print(f"Processing frequency channel {index_freq}...")
        dirty_images.append([])
        for index_p, p in enumerate(pointings):
            print(f"Processing pointing {index_p}...")
            vis = visibilities[index_freq][index_p]
            if vis is None:
                warnings.warn(
                    f"""For frequency channel {index_freq},
                    pointing {index_p}, the visibility is None.
                    Setting dirty image to None, and skipping analysis."""
                )
                dirty_images[-1].append(None)
                continue

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
            assert len(dirty.data.shape) == 4
            # I.e. only one frequency channel,
            # since we are performing a line emission analysis
            assert dirty.data.shape[0] == 1

    assert len(dirty_images) == observation_details.number_of_channels
    assert len(dirty_images[0]) == len(
        pointings
    ), f"{len(dirty_images[0])}, {len(pointings)}"

    return visibilities, dirty_images


def polar_corrdinates_grid(
    im_shape: Tuple[int, int], center: Tuple[float, float]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    raise DeprecationWarning("This function is no longer available.")


def circle_image(image: NDArray[np.float_]) -> NDArray[np.float_]:
    raise DeprecationWarning("Use the circle method within the Image class instead.")


def header_for_mosaic(
    img_size: int, ra_deg: IntFloat, dec_deg: IntFloat, cut: IntFloat
) -> fits.header.Header:
    raise DeprecationWarning("To generate image mosaics, use the ImageMosaicker class.")


def rascil_imager(
    outfile: str, visibility: Visibility, cut: IntFloat = 1.0, img_size: int = 4096
) -> NDArray[np.float_]:
    raise DeprecationWarning("Use the Imager class, with the RASCIL imaging backend.")


def oskar_imager(
    outfile: str,
    ra_deg: IntFloat = 20,
    dec_deg: IntFloat = -30,
    cut: IntFloat = 1.0,
    img_size: int = 4096,
) -> NDArray[np.float_]:
    raise DeprecationWarning("Use the Imager class, with the OSKAR imaging backend.")


def plot_scatter_recon(
    sky: SkyModel,
    recon_image: NDArray[np.float_],
    outfile: FilePathType,
    header: fits.header.Header,
    vmin: IntFloat = 0,
    vmax: Optional[IntFloat] = None,
    f_min: Optional[IntFloat] = None,
) -> None:
    # TODO
    """
    Plotting the sky as a scatter plot and its reconstruction and saving it as a pdf.

    :param sky: Oskar or Karabo sky.
    :param recon_image: Reconstructed sky from Oskar or Karabo.
    :param outfile: The path where we save the plot.
    :param header: The header of the recon_image.
    :param vmin: Minimum value of the colorbar.
    :param vmax: Maximum value of the colorbar.
    :param f_min: Minimal flux of the sources to be plotted in the scatter plot
    """

    wcs = WCS(header)
    slices = get_slices(wcs)

    # Do only plot sources with a flux above f_min if f_min is not None
    if f_min is not None:
        f_max = np.max(sky[:, 2])
        sky = sky.filter_by_flux(min_flux_jy=f_min, max_flux_jy=f_max)

    # Plot the scatter plot and the sky reconstruction next to each other
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121)
    scatter = ax1.scatter(sky[:, 0], sky[:, 1], c=sky[:, 2], vmin=0, s=10, cmap="jet")
    ax1.set_aspect("equal")
    plt.colorbar(scatter, ax=ax1, label="Flux [Jy]")
    ra_deg = header["CRVAL1"]
    dec_deg = header["CRVAL2"]
    img_size_ra = header["NAXIS1"]
    img_size_dec = header["NAXIS2"]
    cut_ra = -header["CDELT1"] * float(img_size_ra)
    cut_dec = header["CDELT2"] * float(img_size_dec)
    ax1.set_xlim((ra_deg - cut_ra / 2, ra_deg + cut_ra / 2))
    ax1.set_ylim((dec_deg - cut_dec / 2, dec_deg + cut_dec / 2))
    ax1.set_xlabel("RA [deg]")
    ax1.set_ylabel("DEC [deg]")
    ax1.invert_xaxis()

    ax2 = fig.add_subplot(122, projection=wcs, slices=slices)
    recon_img = ax2.imshow(
        recon_image, cmap="YlGnBu", origin="lower", vmin=vmin, vmax=vmax
    )
    plt.colorbar(recon_img, ax=ax2, label="Flux Density [Jy]")

    plt.tight_layout()
    plt.savefig(outfile)


def sky_slice(sky: SkyModel, z_min: IntFloat, z_max: IntFloat) -> SkyModel:
    raise DeprecationWarning("Use sky.filter_by_column(13, z_min, z_max) instead.")


def karabo_reconstruction(
    outfile: FilePathType,
    mosaic_pntg_file: Optional[str] = None,
    sky: Optional[SkyModel] = None,
    telescope: Optional[Telescope] = None,
    ra_deg: IntFloat = 20,
    dec_deg: IntFloat = -30,
    start_time: Union[datetime, str] = datetime(2000, 3, 20, 12, 6, 39),
    obs_length: timedelta = timedelta(hours=3, minutes=5, seconds=0, milliseconds=0),
    start_freq: IntFloat = 1.4639e9,
    freq_bin: IntFloat = 1.0e7,
    beam_type: StationTypeType = "Isotropic beam",
    gaussian_fwhm: IntFloat = 1.0,
    gaussian_ref_freq: IntFloat = 1.4639e9,
    cut: IntFloat = 1.0,
    img_size: int = 4096,
    channel_num: int = 10,
    pdf_plot: bool = False,
    circle: bool = False,
    rascil: bool = True,
    verbose: bool = False,
) -> Tuple[NDArray[np.float_], fits.header.Header]:
    raise DeprecationWarning("Use line_emission_pipeline() instead.")


def run_one_channel_simulation(
    path: FilePathType,
    sky: SkyModel,
    telescope: Telescope,
    freq_bin_start: float,
    freq_bin_width: float,
    ra_deg: IntFloat,
    dec_deg: IntFloat,
    beam_type: StationTypeType,
    gaussian_fwhm: IntFloat,
    gaussian_ref_freq: IntFloat,
    start_time: Union[datetime, str],
    obs_length: timedelta,
    cut: IntFloat,
    img_size: int,
    circle: bool,
    rascil: bool,
    verbose: bool = False,
) -> Tuple[NDArray[np.float_], fits.header.Header]:
    raise DeprecationWarning("Use line_emission_pipeline() instead.")


def create_line_emission_h5_file(
    filename: FilePathType,
    dirty_images: List[NDArray[np.float_]],
    redshift_channel: NDArray[np.float_],
    header: fits.header.Header,
) -> None:
    raise DeprecationWarning(
        """
        As a replacement, use the Image class to access FITS data for an image,
        which contains per-channel and per-polarisation data,
        as well as the corresponding header information.
    """
    )


def line_emission_pointing(
    outpath: DirPathType,
    sky: SkyModel,
    telescope: Optional[Telescope] = None,
    ra_deg: IntFloat = 20,
    dec_deg: IntFloat = -30,
    num_bins: int = 10,
    equally_spaced_freq: bool = True,
    beam_type: StationTypeType = "Gaussian beam",
    gaussian_fwhm: IntFloat = 1.0,
    gaussian_ref_freq: IntFloat = 1.4639e9,
    start_time: Union[datetime, str] = datetime(2000, 3, 20, 12, 6, 39),
    obs_length: timedelta = timedelta(hours=3, minutes=5, seconds=0, milliseconds=0),
    cut: IntFloat = 3.0,
    img_size: int = 4096,
    circle: bool = True,
    rascil: bool = True,
    verbose: bool = False,
) -> Tuple[NDArray[np.float_], List[NDArray[np.float_]], fits.header.Header, np.float_]:
    raise DeprecationWarning("Use line_emission_pipeline() instead.")


def gaussian_fwhm_meerkat(freq: NPFloatLikeStrict) -> np.float64:
    raise DeprecationWarning("This function has been removed.")


def gaussian_beam(
    ra_deg: IntFloat,
    dec_deg: IntFloat,
    img_size: int = 2048,
    cut: IntFloat = 1.2,
    fwhm: NPFloatLikeStrict = 1.0,
    outfile: FilePathType = "beam",
) -> Tuple[NDArray[np.float_], fits.header.Header]:
    raise DeprecationWarning("This function has been removed.")


def simple_gaussian_beam_correction(
    outpath: DirPathType,
    dirty_image: NDArray[np.float_],
    gaussian_fwhm: NPFloatLikeStrict,
    ra_deg: IntFloat = 20,
    dec_deg: IntFloat = -30,
    cut: IntFloat = 3.0,
    img_size: int = 4096,
) -> Tuple[NDArray[np.float_], fits.header.Header]:
    raise DeprecationWarning("This function has been removed.")


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
        telescope_name = "SKA1MID"
    elif simulator_backend == SimulatorBackend.RASCIL:
        telescope_name = "MID"

    telescope = Telescope.constructor(telescope_name, backend=simulator_backend)

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
        for index_p, _ in enumerate(pointings):
            dirty = dirty_images[index_freq][index_p]
            if dirty is None:
                continue

            dirty.plot(
                block=True,
                vmin=0,
                vmax=2e-7,
                title=f"Dirty image for pointing {index_p} and channel {index_freq}",
            )

    # Create mosaics of pointings for each frequency channel
    print("Creating mosaic of images for each frequency channel")
    mosaicker = ImageMosaicker()

    mosaics = []
    for index_freq in range(observation.number_of_channels):
        mosaic, _ = mosaicker.mosaic(dirty_images[index_freq])
        mosaics.append(mosaic)

        mosaic.plot(
            filename=str(output_base_directory / f"mosaic_{index_freq}.png"),
            block=True,
            vmin=0,
            vmax=2e-7,
            title=f"Mosaic for channel {index_freq}",
        )

    # Add all mosaics across frequency channels to create one final mosaic image
    summed_mosaic = Image(
        data=sum(m.data for m in mosaics),
        header=mosaics[0].header,
    )
    summed_mosaic.plot(
        filename=str(output_base_directory / "summed_mosaic.png"),
        block=True,
        vmin=0,
        vmax=2e-7,
        title="Summed mosaic across channels",
    )
