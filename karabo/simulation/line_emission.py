import os
import shutil
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
import oskar
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
from karabo.util._types import IntFloat, NPFloatLikeStrict
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
    dirty.write_to_file(outfile + ".fits", overwrite=True)
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
        input_file=outfile + ".vis",
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
    outfile: str,
    header: fits.header.Header,
    vmin: IntFloat = 0,
    vmax: Optional[IntFloat] = None,
    cut: Optional[IntFloat] = None,
) -> None:
    """
    Plotting the sky as a scatter plot and its reconstruction and saving it as a pdf.

    :param sky: Oskar or Karabo sky.
    :param recon_image: Reconstructed sky from Oskar or Karabo.
    :param outfile: The path of the plot.
    :param header: The header of the recon_image.
    :param vmin: Minimum value of the colorbar.
    :param vmax: Maximum value of the colorbar.
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
    plt.savefig(outfile + ".pdf")


def sky_slice(
    sky: SkyModel, z_obs: NDArray[np.float_], z_min: np.float_, z_max: np.float_
) -> SkyModel:
    """
    Extracting a slice from the sky which includes only sources between redshift z_min
    and z_max.

    :param sky: Sky model.
    :param z_obs: Redshift information of the sky sources. # TODO change as soon as
    branch 400 is merged
    :param z_min: Smallest redshift of this sky bin.
    :param z_max: Largest redshift of this sky bin.

    :return: Sky model only including the sources with redshifts between z_min and
             z_max.
    """
    sky_bin = SkyModel.copy_sky(sky)
    sky_bin_idx = np.where((z_obs > z_min) & (z_obs < z_max))
    if sky_bin.sources is None:
        raise TypeError("`sky.sources` is None which is not allowed.")

    sky_bin.sources = sky_bin.sources[sky_bin_idx]

    return sky_bin


def redshift_slices(
    redshift_obs: NDArray[np.float_], channel_num: int = 10
) -> NDArray[np.float_]:
    """
    Creation of the redshift bins used for the line emission simulation based on the
    observed redshift range from the sky.

    :param redshift_obs: Observed redshifts of the sources.
    :param channel_num: Number of redshift bins/channels.

    :return: The channels of the observed redshift range.
    """
    print("Smallest redshift:", np.amin(redshift_obs))
    print("Largest redshift:", np.amax(redshift_obs))

    redshift_channel = np.linspace(
        np.amin(redshift_obs), np.amax(redshift_obs), channel_num + 1
    )

    return redshift_channel


def freq_channels(
    z_obs: NDArray[np.float_], channel_num: int = 10
) -> Tuple[NDArray[np.float_], NDArray[np.float_], np.float_, np.float_]:
    """
    Calculates the frequency channels from the redshifs.
    :param z_obs: Observed redshifts from the HI sources.
    :param channel_num: Number uf channels.

    :return: Redshift channel, frequency channel in Hz, bin width of frequency channel
             in Hz, middle frequency in Hz
    """

    redshift_channel = redshift_slices(z_obs, channel_num)

    freq_channel = c.value / (0.21 * (1 + redshift_channel))
    freq_start = freq_channel[0]
    freq_end = freq_channel[-1]
    freq_mid = freq_start + (freq_end - freq_start) / 2
    freq_bin = freq_channel[0] - freq_channel[1]
    print("The frequency channel starts at:", freq_start, "Hz")
    print("The bin size of the freq channel is:", freq_bin, "Hz")

    return redshift_channel, freq_channel, freq_bin, freq_mid


def karabo_reconstruction(
    outfile: str,
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
    :param channel_num:
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
        vis_path=outfile + ".vis",
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
        dirty_image = rascil_imager(outfile, visibility, cut, img_size)
    else:
        if verbose:
            print("Sky reconstruction with the Oskar Imager")
        dirty_image = oskar_imager(outfile, ra_deg, dec_deg, cut, img_size)

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
        plot_scatter_recon(sky, dirty_image, outfile, header)

    if mosaic_pntg_file is not None:
        if verbose:
            print(
                "Write the reconstructed image to a fits file which can be used for "
                "coaddition.",
                mosaic_pntg_file,
            )
        fits.writeto(mosaic_pntg_file + ".fits", dirty_image, header, overwrite=True)

    return dirty_image, header


def run_one_channel_simulation(
    path_outfile: str,
    sky: SkyModel,
    bin_idx: int,
    z_obs: NDArray[np.float_],
    z_min: np.float_,
    z_max: np.float_,
    freq_min: float,
    freq_bin: float,
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

    :param path_outfile: Pathname of the output file and folder.
    :param sky: Sky model which is used for simulating line emission. If None, a test
                sky (out of equally spaced sources) is used.
    :param bin_idx: Index of the channel which is currently being simulated.
    :param z_obs: Redshift information of the sky sources.
    :param z_min: Smallest redshift in this bin.
    :param z_max: Largest redshift in this bin.
    :param freq_min: Smallest frequency in this bin.
    :param freq_bin: Size of the sky frequency bin which is simulated.
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
    if verbose:
        print(
            "Channel " + str(bin_idx) + " is being processed...\n"
            "Extracting the corresponding frequency slice from the sky model..."
        )

    sky_bin = sky_slice(sky, z_obs, z_min, z_max)

    if verbose:
        print("Starting simulation...")
    start_freq = freq_min + freq_bin / 2
    dirty_image, header = karabo_reconstruction(
        path_outfile + os.path.sep + "slice_" + str(bin_idx),
        sky=sky_bin,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        start_freq=start_freq,
        freq_bin=freq_bin,
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
    path_outfile: str,
    sky: SkyModel,
    z_obs: NDArray[np.float_],  # TODO: After branch 400-read_in_sky-exists the sky
    # includes this information -> rewrite
    ra_deg: IntFloat = 20,
    dec_deg: IntFloat = -30,
    num_bins: int = 10,
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

    :param path_outfile: Pathname of the output file and folder.
    :param sky: Sky model which is used for simulating line emission. If None, a test
                sky (out of equally spaced sources) is used.
    :param z_obs: Redshift information of the sky sources.
    :param ra_deg: Phase center right ascension.
    :param dec_deg: Phase center declination.
    :param num_bins: Number of redshift/frequency slices used to simulate line emission.
                     The more the better the line emission is simulated.
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
    applying gaussian primary beam correction to it.

    outpath = (
        "/home/user/Documents/SKAHIIM_Pipeline/result/Reconstructions/"
        "Line_emission_pointing_2"
    )
    catalog_path = (
        "/home/user/Documents/SKAHIIM_Pipeline/Flux_calculation/"
        "Catalog/point_sources_OSKAR1_FluxBattye_diluted5000.h5"
    )
    ra = 20
    dec = -30
    sky_pointing = SkyModel.sky_from_h5_with_redshift_filtered(
        catalog_path, ra, dec
    )
    dirty_im, _, header_dirty, freq_mid_dirty = line_emission_pointing(
        outpath, sky_pointing, z_obs_pointing
    )
    plot_scatter_recon(
        sky_pointing, dirty_im, outpath, header_dirty, vmax=0.15, cut=3.0
    )
    gauss_fwhm = gaussian_fwhm_meerkat(freq_mid_dirty)
    beam_corrected, _ = simple_gaussian_beam_correction(outpath, dirty_im, gauss_fwhm)
    plot_scatter_recon(
        sky_pointing,
        beam_corrected,
        outpath + "_GaussianBeam_Corrected",
        header_dirty,
        vmax=0.15,
        cut=3.0,
    )
    """
    # Create folders to save outputs/ delete old one if it already exists
    if os.path.exists(path_outfile):
        shutil.rmtree(path_outfile)

    os.makedirs(path_outfile)

    if not client:
        "Print: Get dask client"
        client = DaskHandler.get_dask_client()

    redshift_channel, freq_channel, freq_bin, freq_mid = freq_channels(z_obs, num_bins)

    dirty_images = []
    header: Optional[fits.header.Header] = None

    # Run the simulation on the das cluster
    if client is not None:
        # Calculate the number of jobs
        n_jobs = num_bins
        print(f"Submitting {n_jobs} jobs to the cluster.")

        delayed_results = []

        sources = sky.sources

        if sources is None:
            raise TypeError(
                "`sources` None is not allowed! Please set them in"
                " the `SkyModel` before calling this function."
            )

        for bin_idx in range(num_bins):
            delayed_ = delayed(run_one_channel_simulation)(
                path_outfile=path_outfile,
                sky=sky,
                bin_idx=bin_idx,
                z_obs=z_obs,
                z_min=redshift_channel[bin_idx],
                z_max=redshift_channel[bin_idx + 1],
                freq_min=freq_channel[bin_idx],
                freq_bin=freq_bin,
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
    dirty_image = cast(NDArray[np.float_], sum(dirty_images))

    print("Save summed dirty images as fits file")
    dirty_img = fits.PrimaryHDU(dirty_image)
    dirty_img.writeto(path_outfile + ".fits", overwrite=True)

    print("Save 3-dim reconstructed dirty images as h5")
    z_bin = redshift_channel[1] - redshift_channel[0]
    z_channel_mid = redshift_channel + z_bin / 2

    f = h5py.File(path_outfile + ".h5", "w")
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
    outfile: str = "beam",
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
    fits.writeto(outfile + ".fits", beam_image, header, overwrite=True)

    return beam_image, header


def simple_gaussian_beam_correction(
    path_outfile: str,
    dirty_image: NDArray[np.float_],
    gaussian_fwhm: NPFloatLikeStrict,
    ra_deg: IntFloat = 20,
    dec_deg: IntFloat = -30,
    cut: IntFloat = 3.0,
    img_size: int = 4096,
) -> Tuple[NDArray[np.float_], fits.header.Header]:
    print("Calculate gaussian beam for primary beam correction...")
    beam, header = gaussian_beam(
        img_size=img_size,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        cut=cut,
        fwhm=gaussian_fwhm,
        outfile=path_outfile + os.path.sep + "gaussian_beam",
    )

    print("Apply primary beam correction...")
    dirty_image_corrected = dirty_image / beam

    fits.writeto(
        path_outfile + "_GaussianBeam_Corrected.fits",
        dirty_image_corrected,
        header,
        overwrite=True,
    )

    return dirty_image_corrected, header
