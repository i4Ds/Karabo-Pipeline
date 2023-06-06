# import copy
# import os
# import shutil
from datetime import datetime, timedelta
from typing import Tuple

# import h5py
import matplotlib.pyplot as plt
import numpy as np
import oskar

# from astropy.constants import c
# from astropy.convolution import Gaussian2DKernel
from astropy.io import fits

# from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from numpy.typing import NDArray

from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation, StationTypeType
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


def polar_corrdinates_grid(im_shape: Tuple[int, int], center: Tuple[int, int]):
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


def circle_image(image: NDArray[np.float_]):
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


def header_for_mosaic(img_size: int, ra: float, dec: float, cut: float):
    """
    Create a header for the fits file of the reconstructed image, which is compatible
    with the mosaicking done by MontagePy
    :param img_size: Pixel size of the image.
    :param ra: Right ascension of the center.
    :param dec: Declination of the center.
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
    header["CRVAL1"] = ra
    header["CRVAL2"] = dec
    header["CDELT1"] = -cut / float(img_size)  # 1 arcsecond per pixel
    header["CDELT2"] = cut / float(img_size)
    header["CRPIX1"] = img_size / 2.0
    header["CRPIX2"] = img_size / 2.0
    header["EQUINOX"] = 2000.0

    return header


def rascil_imager(outfile: str, visibility, cut: float = 1.0, img_size: int = 4096):
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
    dirty_image = dirty.data[0][0]
    return dirty_image


def oskar_imager(
    outfile: str,
    ra: float = 20,
    dec: float = -30,
    cut: float = 1.0,
    img_size: int = 4096,
):
    """
    Reconstructs the image from the visibilities with oskar.

    :param outfile: Path/Name of the output files.
    :param ra: Phase center right ascension.
    :param dec: Phase center declination.
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
    imager.set_vis_phase_centre(ra, dec)
    imager.output_root = outfile

    output = imager.run(return_images=1)
    image = output["images"][0]
    return image


def plot_scatter_recon(
    sky: SkyModel,
    recon_image: NDArray[np.float_],
    outfile: str,
    header: fits.header.Header = None,
    vmin: float = 0,
    vmax: float = 0.4,
    cut: float = None,
):
    """
    Plotting the sky as a scatter plot and its reconstruction and saving it as a pdf.

    :param sky: Oskar or Karabo sky.
    :param recon_image: Reconstructed sky from Oskar or Karabo.
    :param outfile: The path of the plot.
    :param header: The header of the recon_image.
    :param vmin: Minimum value of the colorbar.
    :param vmax: Maximum value of the colorbar.
    :param cut: Smaller FOV
    :return:
    """

    wcs = WCS(header)
    slices = []
    for i in range(wcs.pixel_n_dim):
        if i == 0:
            slices.append("x")
        elif i == 1:
            slices.append("y")
        else:
            slices.append(0)

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


def karabo_reconstruction(
    outfile: str,
    mosaic_pntg_file: str = None,
    sky: SkyModel = None,
    ra_deg: float = 20,
    dec_deg: float = -30,
    start_time=datetime(2000, 3, 20, 12, 6, 39),
    obs_length=timedelta(hours=3, minutes=5, seconds=0, milliseconds=0),
    start_freq: float = 1.4639e9,
    freq_bin: float = 1.0e7,
    beam_type: StationTypeType = "Isotropic beam",
    gaussian_fwhm: float = 1.0,
    gaussian_ref_freq: float = 1.4639e9,
    cut: float = 1.0,
    img_size: int = 4096,
    channel_num: int = 10,
    pdf_plot: bool = False,
    circle: bool = False,
    rascil: bool = True,
):
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
    :return: Reconstructed sky of one pointing of size cut.
    """
    print("Create Sky...")
    if sky is None:
        sky = SkyModel.sky_test()

    telescope = Telescope.get_MEERKAT_Telescope()

    print("Sky Simulation...")
    simulation = InterferometerSimulation(
        vis_path=outfile + ".vis",
        channel_bandwidth_hz=1.0e7,
        time_average_sec=8,
        ignore_w_components=True,
        uv_filter_max=3000,
        use_gpus=True,
        station_type=beam_type,
        enable_power_pattern=True,
        gauss_beam_fwhm_deg=gaussian_fwhm,
        gauss_ref_freq_hz=gaussian_ref_freq,
    )
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
    print("Calculate visibilites...")
    visibility = simulation.run_simulation(telescope, sky, observation)

    if rascil:
        print("Sky reconstruction with Rascil...")
        dirty_image = rascil_imager(outfile, visibility, cut, img_size)
    else:
        print("Sky reconstruction with the Oskar Imager")
        dirty_image = oskar_imager(outfile, ra_deg, dec_deg, cut, img_size)

    if circle:
        print("Cutout a circle from image...")
        dirty_image = circle_image(dirty_image)

    header = header_for_mosaic(img_size, ra_deg, dec_deg, cut)
    if pdf_plot:
        print(
            "Creation of a pdf with scatter plot and reconstructed image to ",
            str(outfile),
        )
        plot_scatter_recon(sky, dirty_image, outfile, header)

    if mosaic_pntg_file is not None:
        print(
            "Write the reconstructed image to a fits file which can be used for "
            "coaddition.",
            mosaic_pntg_file,
        )
        fits.writeto(mosaic_pntg_file + ".fits", dirty_image, header, overwrite=True)

    return dirty_image, header


if __name__ == "__main__":
    karabo_reconstruction(
        "/home/jennifer/Documents/SKAHIIM_Pipeline/result/Beam/karabo_test1",
        mosaic_pntg_file="/home/jennifer/Documents/SKAHIIM_Pipeline/result/Beam/"
        "test4_karabo",
        beam_type="Gaussian beam",
        circle=True,
        pdf_plot=True,
        cut=2,
    )
