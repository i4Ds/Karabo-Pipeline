import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
import oskar
import xarray as xr
from astropy.constants import c
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.wcs import WCS

# from dask.delayed import Delayed
from dask import compute, delayed  # type: ignore[attr-defined]
from dask.distributed import Client
from numpy.typing import NDArray

from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation, StationTypeType
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.util._types import DirPathType, FilePathType, IntFloat, NPFloatLikeStrict
from karabo.util.dask import DaskHandler


def polar_corrdinates_grid(
    im_shape: Tuple[int, int], center: Tuple[float, float]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Creates a corresponding r-phi grid for the x-y coordinate system

    :param im_shape: (x_len, y_len) is the shape of the image in x-y (pixel) coordinates
    :param center: The pixel values of the center (x_center, y_center)
    :return: The corresponding r-phi grid.
    """
    x, y = np.ogrid[: im_shape[0], : im_shape[1]]
    cx, cy = center[0], center[1]

    # convert cartesian --> polar coordinates
    r_array = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    phi_array = np.arctan2(-(x - cx), (y - cy))

    # Needed so that phi = [0, 2*pi] otherwise phi = [-pi, pi]
    phi_array %= 2 * np.pi

    return r_array, phi_array


def circle_image(image: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Cuts the image to a circle, where it takes the x_len/2 as a radius, where x_len is
    the length of the image. Assuming a square image.

    :param image: Input image.
    :return: Image cut, so that only a circle of radius x_len/2 is taken into account.
    """
    x_len, y_len = image.shape
    x_center = x_len / 2 - 1
    y_center = y_len / 2 - 1

    r_array, phi_array = polar_corrdinates_grid((x_len, y_len), (x_center, y_center))

    mask = r_array <= x_center
    image[~mask] = np.NAN

    return image


def header_for_mosaic(
    img_size: int, ra_deg: IntFloat, dec_deg: IntFloat, cut: IntFloat
) -> fits.header.Header:
    """
    Create a header for the fits file of the reconstructed image, which is compatible
    with the mosaicking done by MontagePy
    :param img_size: Pixel size of the image.
    :param ra_deg: Right ascension of the center.
    :param dec_deg: Declination of the center.
    :param cut: Size of the reconstructed image in degree.
    :return: Fits header.
    """

    # Create the header
    header = fits.Header()
    header["SIMPLE"] = "T"
    header["BITPIX"] = -64
    header["NAXIS"] = 2
    header["NAXIS1"] = img_size
    header["NAXIS2"] = img_size
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRVAL1"] = ra_deg
    header["CRVAL2"] = dec_deg
    header["CDELT1"] = -cut / float(img_size)  # 1 arcsecond per pixel
    header["CDELT2"] = cut / float(img_size)
    header["CRPIX1"] = img_size / 2.0
    header["CRPIX2"] = img_size / 2.0
    header["EQUINOX"] = 2000.0

    return header


def rascil_imager(
    outfile: str, visibility: Visibility, cut: IntFloat = 1.0, img_size: int = 4096
) -> NDArray[np.float_]:
    """
    Reconstruct the image from the visibilities with rascil.

    :param outfile: Path/Name of the output files.
    :param visibility: Calculated visibilities from sky reconstruction.
    :param cut: Size of the reconstructed image.
    :param img_size: The pixel size of the reconstructed image.
    :return: Dirty image reconstruction of sky.
    """
    cut = cut / 180 * np.pi
    imager = Imager(
        visibility,
        imaging_npixel=img_size,
        imaging_cellsize=cut / img_size,
        imaging_dopsf=True,
    )
    dirty = imager.get_dirty_image()
    dirty.write_to_file(f"{outfile}.fits", overwrite=True)
    dirty_image = cast(NDArray[np.float_], dirty.data[0][0])

    return dirty_image


def oskar_imager(
    outfile: str,
    ra_deg: IntFloat = 20,
    dec_deg: IntFloat = -30,
    cut: IntFloat = 1.0,
    img_size: int = 4096,
) -> NDArray[np.float_]:
    """
    Reconstructs the image from the visibilities with oskar.

    :param outfile: Path/Name of the output files.
    :param ra_deg: Phase center right ascension.
    :param dec_deg: Phase center declination.
    :param cut: Size of the reconstructed image.
    :param img_size: The pixel size of the reconstructed image.
    :return: Dirty image reconstruction of sky.
    """
    imager = oskar.Imager()
    # Here plenty of options are available that could be found in the documentation.
    # uv_filter_max can be used to change the baseline length threshold
    imager.set(
        input_file=f"{outfile}.vis",
        output_root=outfile,
        fov_deg=cut,
        image_size=img_size,
        weighting="Uniform",
        uv_filter_max=3000,
    )
    imager.set_vis_phase_centre(ra_deg, dec_deg)
    imager.output_root = outfile

    output = imager.run(return_images=1)
    image = cast(NDArray[np.float_], output["images"][0])
    return image


def plot_scatter_recon(
    sky: SkyModel,
    recon_image: NDArray[np.float_],
    outfile: FilePathType,
    header: fits.header.Header,
    vmin: IntFloat = 0,
    vmax: Optional[IntFloat] = None,
    f_min: Optional[IntFloat] = None,
    cut: Optional[IntFloat] = None,
) -> None:
    """
    Plotting the sky as a scatter plot and its reconstruction and saving it as a pdf.

    :param sky: Oskar or Karabo sky.
    :param recon_image: Reconstructed sky from Oskar or Karabo.
    :param outfile: The path where we save the plot.
    :param header: The header of the recon_image.
    :param vmin: Minimum value of the colorbar.
    :param vmax: Maximum value of the colorbar.
    :param f_min: Minimal flux of the sources to be plotted in the scatter plot
    :param cut: Smaller FOV
    """

    wcs = WCS(header)
    slices = []
    for i in range(wcs.pixel_n_dim):
        if i == 0:
            slices.append("x")
        elif i == 1:
            slices.append("y")
        else:
            slices.append(0)  # type: ignore [arg-type]

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
    if cut is not None:
        ra_deg = header["CRVAL1"]
        dec_deg = header["CRVAL2"]
        ax1.set_xlim((ra_deg - cut / 2, ra_deg + cut / 2))
        ax1.set_ylim((dec_deg - cut / 2, dec_deg + cut / 2))
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


def sky_slice(sky: SkyModel, z_min: np.float_, z_max: np.float_) -> SkyModel:
    """
    Extracting a slice from the sky which includes only sources between redshift z_min
    and z_max.

    :param sky: Sky model which is used for simulating line emission. This sky model
                needs to include a 13th axis (extra_column) with the observed redshift
                of each source.
    :param z_min: Smallest redshift of this sky bin.
    :param z_max: Largest redshift of this sky bin.

    :return: Sky model only including the sources with redshifts between z_min and
             z_max.
    """
    sky_bin = SkyModel.copy_sky(sky)
    if sky_bin.sources is None:
        raise TypeError("`sky.sources` is None which is not allowed.")

    z_obs = sky_bin.sources[:, 13]
    sky_bin_idx = np.where((z_obs > z_min) & (z_obs < z_max))
    sky_bin.sources = sky_bin.sources[sky_bin_idx]

    return sky_bin


def convert_z_to_frequency(
    z: Union[NDArray[np.float_], xr.DataArray]
) -> Union[NDArray[np.float_], xr.DataArray]:
    """Turn given redshift into corresponding frequency (Hz) for 21cm emission.

    :param z: Redshift values to be converted into frequencies.

    :return: Frequencies corresponding to input redshifts.
    """

    return c.value / (0.21 * (1 + z))


def convert_frequency_to_z(
    freq: Union[NDArray[np.float_], xr.DataArray]
) -> Union[NDArray[np.float_], xr.DataArray]:
    """Turn given frequency (Hz) into corresponding redshift for 21cm emission.

    :param freq: Frequency values to be converted into redshifts.

    :return: Redshifts corresponding to input frequencies.
    """

    return (c.value / (0.21 * freq)) - 1


def freq_channels(
    z_obs: Union[NDArray[np.float_], xr.DataArray],
    channel_num: int = 10,
    equally_spaced_freq: bool = True,
) -> Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], np.float_]:
    """
    Calculates the frequency channels from the redshifts.
    :param z_obs: Observed redshifts from the HI sources.
    :param channel_num: Number of channels.
    :param equally_spaced_freq: If True (default), create channels
        equally spaced in frequency.
        If False, create channels equally spaced in redshift.

    :return: Redshift channels array,
        frequency channels array (in Hz),
        array of bin widths of frequency channel (in Hz), for convenience,
        and middle frequency (in Hz)
    """
    z_start = np.min(z_obs)
    z_end = np.max(z_obs)

    freq_start, freq_end = convert_z_to_frequency(np.array([z_start, z_end]))

    freq_mid = freq_start + (freq_end - freq_start) / 2

    if equally_spaced_freq is True:
        freq_channels_array = np.linspace(
            freq_start,
            freq_end,
            channel_num + 1,
        )

        redshift_channels_array = convert_frequency_to_z(freq_channels_array)
    else:
        redshift_channels_array = np.linspace(
            np.amin(z_obs),
            np.amax(z_obs),
            channel_num + 1,
        )

        freq_channels_array = convert_z_to_frequency(redshift_channels_array)

    freq_bins = np.abs(np.diff(freq_channels_array))

    print("The frequency channel starts at:", freq_start, "Hz")
    print("The bin sizes of the freq channel are:", freq_bins, "Hz")

    return redshift_channels_array, freq_channels_array, freq_bins, freq_mid


def karabo_reconstruction(
    outfile: FilePathType,
    mosaic_pntg_file: Optional[str] = None,
    sky: Optional[SkyModel] = None,
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
    """
    Performs a sky reconstruction for our test sky.

    :param outfile: Path/Name of the output files.
    :param mosaic_pntg_file: If provided an additional output fits file which has the
                             correct format for creating a
                             mosaic with Montage is created and saved at this path.
    :param sky: Sky model. If None, a test sky (out of equally spaced sources) is used.
    :param ra_deg: Phase center right ascension.
    :param dec_deg: Phase center declination.
    :param start_time: Observation start time.
    :param obs_length: Observation length (time).
    :param start_freq: The frequency at the midpoint of the first channel in Hz.
    :param freq_bin: The frequency width of the channel.
    :param beam_type: Primary beam assumed, e.g. "Isotropic beam", "Gaussian beam",
                      "Aperture Array".
    :param gaussian_fwhm: If the primary beam is gaussian, this is its FWHM. In power
                          pattern. Units = degrees.
    :param gaussian_ref_freq: If you choose "Gaussian beam" as station type you need
                              specify the reference frequency of
                              the reference frequency of the full-width half maximum
                              here.
    :param cut: Size of the reconstructed image.
    :param img_size: The pixel size of the reconstructed image.
    :param channel_num: The number of frequency channels to be used for the
                        simulation of the continuous emission and therefore for the
                        reconstruction.
    :param pdf_plot: Shall we plot the scatter plot and the reconstruction as a pdf?
    :param circle: If set to True, the pointing has a round shape of size cut.
    :param rascil: If True we use the Imager Rascil otherwise the Imager from Oskar is
                   used.
    :param verbose: If True you get more print statements.
    :return: Reconstructed sky of one pointing of size cut.
    """
    if verbose:
        print("Create Sky...")
    if sky is None:
        sky = SkyModel.sky_test()

    telescope = Telescope.get_MEERKAT_Telescope()

    if verbose:
        print("Sky Simulation...")
    simulation = InterferometerSimulation(
        vis_path=f"{outfile}.vis",
        channel_bandwidth_hz=1.0e7,
        time_average_sec=8,
        ignore_w_components=True,
        uv_filter_max=3000,
        use_gpus=False,
        station_type=beam_type,
        enable_power_pattern=True,
        gauss_beam_fwhm_deg=gaussian_fwhm,
        gauss_ref_freq_hz=gaussian_ref_freq,
        use_dask=False,
    )
    if verbose:
        print("Setup observation parameters...")
    observation = Observation(
        phase_centre_ra_deg=ra_deg,
        phase_centre_dec_deg=dec_deg,
        start_date_and_time=start_time,
        length=obs_length,
        number_of_time_steps=10,
        start_frequency_hz=start_freq,
        frequency_increment_hz=freq_bin,
        number_of_channels=channel_num,
    )
    if verbose:
        print("Calculate visibilites...")
    visibility = simulation.run_simulation(telescope, sky, observation)

    if rascil:
        if verbose:
            print("Sky reconstruction with Rascil...")
        dirty_image = rascil_imager(str(outfile), visibility, cut, img_size)
    else:
        if verbose:
            print("Sky reconstruction with the Oskar Imager")
        dirty_image = oskar_imager(str(outfile), ra_deg, dec_deg, cut, img_size)

    if circle:
        if verbose:
            print("Cutout a circle from image...")
        dirty_image = circle_image(dirty_image)

    header = header_for_mosaic(img_size, ra_deg, dec_deg, cut)
    if pdf_plot:
        if verbose:
            print(
                "Creation of a pdf with scatter plot and reconstructed image to ",
                str(outfile),
            )
        plot_scatter_recon(sky, dirty_image, f"{outfile}.pdf", header)

    if mosaic_pntg_file is not None:
        if verbose:
            print(
                "Write the reconstructed image to a fits file which can be used for "
                "coaddition.",
                mosaic_pntg_file,
            )
        fits.writeto(f"{mosaic_pntg_file}.fits", dirty_image, header, overwrite=True)

    return dirty_image, header


def run_one_channel_simulation(
    path: FilePathType,
    sky: SkyModel,
    z_min: np.float_,
    z_max: np.float_,
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
    """
    Run simulation for one pointing and one channel

    :param path: Pathname of the output file and folder.
    :param sky: Sky model which is used for simulating line emission. This sky model
                needs to include a 13th axis (extra_column) with the observed redshift
                of each source.
    :param z_min: Smallest redshift in this bin.
    :param z_max: Largest redshift in this bin.
    :param freq_bin_start: Starting frequency in this bin
        (i.e., largest frequency in the bin).
    :param freq_bin_width: Size of the sky frequency bin which is simulated.
    :param ra_deg: Phase center right ascension.
    :param dec_deg: Phase center declination.
    :param beam_type: Primary beam assumed, e.g. "Isotropic beam", "Gaussian beam",
                      "Aperture Array".
    :param gaussian_fwhm: If the primary beam is gaussian, this is its FWHM. In power
                          pattern. Units = degrees.
    :param gaussian_ref_freq: If you choose "Gaussian beam" as station type you need
                              specify the reference frequency of the reference
                              frequency of the full-width half maximum here.
    :param start_time: Observation start time.
    :param obs_length: Observation length (time).
    :param cut: Size of the reconstructed image.
    :param img_size: The pixel size of the reconstructed image.
    :param circle: If set to True, the pointing has a round shape of size cut.
    :param rascil: If True we use the Imager Rascil otherwise the Imager from Oskar is
                   used.
    :param verbose: If True you get more print statements.
    :return: Reconstruction of one bin slice of the sky and its header.
    """

    sky_bin = sky_slice(sky, z_min, z_max)

    if verbose:
        print("Starting simulation...")

    freq_bin_middle = freq_bin_start - freq_bin_width / 2
    dirty_image, header = karabo_reconstruction(
        path,
        sky=sky_bin,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        start_freq=freq_bin_middle,
        freq_bin=freq_bin_width,
        beam_type=beam_type,
        gaussian_fwhm=gaussian_fwhm,
        gaussian_ref_freq=gaussian_ref_freq,
        start_time=start_time,
        obs_length=obs_length,
        cut=cut,
        img_size=img_size,
        channel_num=1,
        circle=circle,
        rascil=rascil,
        verbose=verbose,
    )

    return dirty_image, header


def line_emission_pointing(
    outpath: DirPathType,
    sky: SkyModel,
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
    client: Optional[Client] = None,
    verbose: bool = False,
) -> Tuple[NDArray[np.float_], List[NDArray[np.float_]], fits.header.Header, np.float_]:
    """
    Simulating line emission for one pointing.

    :param outpath: Path where output files will be saved.
    :param sky: Sky model which is used for simulating line emission. This sky model
                needs to include a 13th axis (extra_column) with the observed redshift
                of each source.
    :param ra_deg: Phase center right ascension.
    :param dec_deg: Phase center declination.
    :param num_bins: Number of redshift/frequency slices used to simulate line emission.
                     The more the better the line emission is simulated.
                     This value also restricts the parallelization. The number of bins
                     restricts the number of nodes which are effectively used.
    :param equally_spaced_freq: If True (default), create channels
        equally spaced in frequency.
        If False, create channels equally spaced in redshift.
    :param beam_type: Primary beam assumed, e.g. "Isotropic beam", "Gaussian beam",
                      "Aperture Array".
    :param gaussian_fwhm: If the primary beam is gaussian, this is its FWHM. In power
                          pattern. Units = degrees.
    :param gaussian_ref_freq: If you choose "Gaussian beam" as station type you need
                              specify the reference frequency of the reference
                              frequency of the full-width half maximum here.
    :param start_time: Observation start time.
    :param obs_length: Observation length (time).
    :param cut: Size of the reconstructed image.
    :param img_size: The pixel size of the reconstructed image.
    :param circle: If set to True, the pointing has a round shape of size cut.
    :param rascil: If True we use the Imager Rascil otherwise the Imager from Oskar is
                   used.
    :param client: Setting a dask client is optional.
    :param verbose: If True you get more print statements.
    :return: Total line emission reconstruction, 3D line emission reconstruction,
             Header of reconstruction and mean frequency.


    E.g. for how to do the simulation of line emission for one pointing and then
    applying gaussian primary beam correction to it at karabo/test/test_line_emission.py
    and karabo/examples/HIIM_Img_Recovery.ipynb
    """
    # Create folders to save outputs/ delete old one if it already exists
    outpath = Path(outpath)

    if os.path.exists(outpath):
        shutil.rmtree(outpath)

    os.makedirs(outpath)

    # Load sky into memory and close connection to h5
    sky.compute()

    if sky.sources is None:
        raise TypeError(
            "`sources` None is not allowed! Please set them in"
            " the `SkyModel` before calling this function."
        )

    if not client:
        client = DaskHandler.get_dask_client()

    redshift_channel, freq_channel, freq_bin, freq_mid = freq_channels(
        z_obs=sky.sources[:, 13],
        channel_num=num_bins,
        equally_spaced_freq=equally_spaced_freq,
    )

    dirty_images = []
    header: Optional[fits.header.Header] = None

    # Calculate the number of jobs
    n_jobs = num_bins
    print(f"Submitting {n_jobs} jobs to the cluster.")

    delayed_results = []

    for bin_idx in range(num_bins):
        if verbose:
            print(
                f"Channel {bin_idx} is being processed...\n"
                "Extracting the corresponding frequency slice from the sky model..."
            )
        delayed_ = delayed(run_one_channel_simulation)(
            path=outpath / (f"slice_{bin_idx}"),
            sky=sky,
            z_min=redshift_channel[bin_idx],
            z_max=redshift_channel[bin_idx + 1],
            freq_bin_start=freq_channel[bin_idx],
            freq_bin_width=freq_bin[bin_idx],
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            beam_type=beam_type,
            gaussian_fwhm=gaussian_fwhm,
            gaussian_ref_freq=gaussian_ref_freq,
            start_time=start_time,
            obs_length=obs_length,
            cut=cut,
            img_size=img_size,
            circle=circle,
            rascil=rascil,
            verbose=verbose,
        )
        delayed_results.append(delayed_)

    result = compute(*delayed_results, scheduler="distributed")
    dirty_images = [x[0] for x in result]
    headers = [x[1] for x in result]
    header = headers[0]

    if header is None:
        raise ValueError("No Header found.")
    dirty_image = cast(NDArray[np.float_], np.einsum("ijk->jk", dirty_images))

    print("Save summed dirty images as fits file")
    dirty_img = fits.PrimaryHDU(dirty_image, header=header)
    dirty_img.writeto(
        outpath / ("line_emission_total_dirty_image.fits"),
        overwrite=True,
    )

    print("Save 3-dim reconstructed dirty images as h5")
    z_bin = redshift_channel[1] - redshift_channel[0]
    z_channel_mid = redshift_channel + z_bin / 2

    f = h5py.File(
        outpath / ("line_emission_dirty_images.h5"),
        "w",
    )
    dataset_dirty = f.create_dataset("Dirty Images", data=dirty_images)
    dataset_dirty.attrs["Units"] = "Jy"
    f.create_dataset("Observed Redshift Channel Center", data=z_channel_mid)
    f.create_dataset("Observed Redshift Bin Size", data=z_bin)

    return dirty_image, dirty_images, header, freq_mid


def gaussian_fwhm_meerkat(freq: NPFloatLikeStrict) -> np.float64:
    """
    Computes the FWHM of MeerKAT for a certain observation frequency.

    :param freq: Frequency of interest in Hz.
    :return: The power pattern FWHM of the MeerKAT telescope at this frequency in
             degrees.
    """
    root = cast(np.float64, np.sqrt(89.5 * 86.2))
    gaussian_fwhm = root / 60.0 * (1e3 / (freq / 10**6))

    return gaussian_fwhm


def gaussian_beam(
    ra_deg: IntFloat,
    dec_deg: IntFloat,
    img_size: int = 2048,
    cut: IntFloat = 1.2,
    fwhm: NPFloatLikeStrict = 1.0,
    outfile: FilePathType = "beam",
) -> Tuple[NDArray[np.float_], fits.header.Header]:
    """
    Creates a Gaussian beam at RA, DEC.
    :param ra_deg: Right ascension coordinate of center of Gaussian.
    :param dec_deg: Declination coordinate of center of Gaussian.
    :param img_size: Pixel image size.
    :param cut: Image size in degrees.
    :param fwhm: FWHM of the Gaussian in degrees.
    :param outfile: Name of the image file with the Gaussian.
    :return:
    """
    # We create the image header and the wcs frame of the Gaussian
    header = header_for_mosaic(
        img_size=img_size, ra_deg=ra_deg, dec_deg=dec_deg, cut=cut
    )
    wcs = WCS(header)

    # Calculate Gaussian
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gauss_kernel = Gaussian2DKernel(
        sigma / wcs.wcs.cdelt[0], x_size=img_size, y_size=img_size
    )

    beam_image = gauss_kernel.array
    # normalize the kernel, such that max=1.0
    beam_image = beam_image / np.max(beam_image)

    # make the beam image circular and save it as a fits file
    beam_image = circle_image(beam_image)
    fits.writeto(outfile, beam_image, header, overwrite=True)

    return beam_image, header


def simple_gaussian_beam_correction(
    outpath: DirPathType,
    dirty_image: NDArray[np.float_],
    gaussian_fwhm: NPFloatLikeStrict,
    ra_deg: IntFloat = 20,
    dec_deg: IntFloat = -30,
    cut: IntFloat = 3.0,
    img_size: int = 4096,
) -> Tuple[NDArray[np.float_], fits.header.Header]:
    """
    Apply Gaussian Beam correction to the Dirty Image.

    :param outpath: Path where beam data and output image will be saved.
    :param dirty_image: Dirty Image, e.g. output from line_emission_pointing.
    :param gaussian_fwhm: FWHM of the Gaussian in degrees.
    :param ra_deg: Right ascension coordinate of the phase center, in degrees.
        This is also used as the RA for the center of the Gaussian beam.
    :param dec_deg: Declination coordinate of the phase center, in degrees.
        This is also used as the Dec for the center of the Gaussian beam.
    :param cut: Image size in degrees.
    :param img_size: Pixel image size.
    :return: Corrected image and header data.
    """
    outpath = Path(outpath)
    print("Calculate gaussian beam for primary beam correction...")
    beam, header = gaussian_beam(
        img_size=img_size,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        cut=cut,
        fwhm=gaussian_fwhm,
        outfile=outpath / "gaussian_beam.fits",
    )

    print("Apply primary beam correction...")
    dirty_image_corrected = dirty_image / beam

    fits.writeto(
        outpath / "line_emission_total_image_beamcorrected.fits",
        dirty_image_corrected,
        header,
        overwrite=True,
    )

    return dirty_image_corrected, header
