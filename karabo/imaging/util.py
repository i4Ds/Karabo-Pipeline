from __future__ import annotations

import warnings
from typing import List, Tuple, Union

import numpy as np
from astropy.modeling import fitting, models
from astropy.wcs import WCS
from numpy.typing import NDArray
from rascil import processing_components as rpc
from scipy.optimize import minpack
from ska_sdp_datamodels.image.image_model import Image as SkaSdpImage

from karabo.data.external_data import MGCLSContainerDownloadObject
from karabo.imaging.image import Image
from karabo.simulation.sky_model import SkyModel
from karabo.util._types import BeamType
from karabo.warning import KaraboWarning


def get_MGCLS_images(regex_pattern: str, verbose: bool = False) -> List[SkaSdpImage]:
    """
    MeerKAT Galaxy Cluster Legacy Survey Data Release 1 (MGCLS DR1)
    https://doi.org/10.48479/7epd-w356
    The first data release of the MeerKAT Galaxy Cluster Legacy Survey (MGCLS)
    consists of the uncalibrated visibilities, a set of continuum imaging products,
    and several source catalogues. All clusters have Stokes-I products,
    and approximately 40% have Stokes-Q and U products as well. For full details,
    including caveats for usage,
    see the survey overview and DR1 paper (Knowles et al., 2021).

    When using any of the below products, please cite Knowles et al. (2021)
    and include the following Observatory acknowledgement:
    "MGCLS data products were provided by the South African Radio
    Astronomy Observatory and the MGCLS team and were derived from observations
    with the MeerKAT radio telescope. The MeerKAT telescope is operated by the
    South African Radio Astronomy Observatory, which is a facility of the National
    Research Foundation, an agency of the Department of Science and Innovation."

    The final enhanced image data products are five-plane cubes
    (referred to as the 5pln cubes in the following) in which the first
    plane is the brightness at the reference frequency, and the second
    is the spectral index, a**1656/908 , both determined by a least-squares fit
    to log(I) vs. log(v) at each pixel. The third plane is the brightness
    uncertainty estimate, fourth is the spectral index uncertainty, and
    fifth is the χ2 of the least-squares fit. Uncertainty estimates are
    only the statistical noise component and do not include calibration
    or other systematic effects. The five planes are accessible in the
    Xarray.Image in the frequency dimension (first dimension).

    Data will be accessed from the karabo_public folder. The data was downloaded
    from https://archive-gw-1.kat.ac.za/public/repository/10.48479/7epd-w356/
    data/enhanced_products/bucket_contents.html

    Parameters:
    ----------
    regex_pattern : str
        Regex pattern to match the files to download. Best is to check in the bucket
        and paper which data is available and then use the regex pattern to match
        the files you want to download.
    verbose : bool, optional
        If True, prints out the files being downloaded. Defaults to False.

    Returns:
    -------
    List[SkaSdpImage]
        List of images from the MGCLS Enhanced Products bucket.
    """
    mgcls_cdo = MGCLSContainerDownloadObject(regexr_pattern=regex_pattern)
    local_file_paths = mgcls_cdo.get_all(verbose=verbose)
    if len(local_file_paths) == 0:
        raise FileNotFoundError(
            f"No files in {mgcls_cdo._remote_container_url} for {regex_pattern=}"
        )
    mgcls_images: List[SkaSdpImage] = list()
    for local_file_path in local_file_paths:
        mgcls_images.append(
            rpc.image.operations.import_image_from_fits(local_file_path)
        )
    return mgcls_images


def _convert_clean_beam_to_degrees(
    im: Image,
    beam_pixels: tuple[float, float, float],
) -> BeamType:
    """Convert clean beam in pixels to arcsec, arcsec, degree.

    Source: https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/blob/main/src/ska_sdp_func_python/image/operations.py  # noqa: E501

    Args:
        im: Image
        beam_pixels: Beam size in pixels

    Returns:
        "bmaj" (arcsec), "bmin" (arcsec), "bpa" (degree)
    """
    cellsize = im.get_cellsize()
    to_mm: np.float64 = np.sqrt(8.0 * np.log(2.0))
    clean_beam: BeamType
    if beam_pixels[1] > beam_pixels[0]:
        clean_beam = {
            "bmaj": np.rad2deg(beam_pixels[1] * cellsize * to_mm),
            "bmin": np.rad2deg(beam_pixels[0] * cellsize * to_mm),
            "bpa": np.rad2deg(beam_pixels[2]),
        }
    else:
        clean_beam = {
            "bmaj": np.rad2deg(beam_pixels[0] * cellsize * to_mm),
            "bmin": np.rad2deg(beam_pixels[1] * cellsize * to_mm),
            "bpa": np.rad2deg(beam_pixels[2]) + 90.0,
        }
    return clean_beam


def guess_beam_parameters(img: Image) -> BeamType:
    """Fit a two-dimensional Gaussian to img using astropy.modeling.

    This function is usually applied on a PSF-image. Therefore, just
    images who don't have beam-params in the header (e.g. dirty image) may need a
    beam-guess.

    Source: https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python/-/blob/main/src/ska_sdp_func_python/image/deconvolution.py  # noqa: E501

    Args:
        img: Image to guess the beam

    Returns:
        major-axis (arcsec), minor-axis (arcsec), position-angle (degree)
    """
    if img.has_beam_parameters():
        warnings.warn(
            f"Image {img.path} already has beam-info in the header.",
            KaraboWarning,
        )
    npixel = img.data.shape[3]
    sl = slice(npixel // 2 - 7, npixel // 2 + 8)
    y, x = np.mgrid[sl, sl]
    z = img.data[0, 0, sl, sl]

    # isotropic at the moment!
    try:
        p_init = models.Gaussian2D(
            amplitude=np.max(z), x_mean=np.mean(x), y_mean=np.mean(y)
        )
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter("ignore")
            fit = fit_p(p_init, x, y, z)
        if fit.x_stddev <= 0.0 or fit.y_stddev <= 0.0:
            warnings.warn(
                "guess_beam_parameters: error in fitting to psf, "
                + "using 1 pixel stddev"
            )
            beam_pixels = (1.0, 1.0, 0.0)
        else:
            beam_pixels = (
                fit.x_stddev.value,
                fit.y_stddev.value,
                fit.theta.value,
            )
    except minpack.error:
        warnings.warn("guess_beam_parameters: minpack error, using 1 pixel stddev")
        beam_pixels = (1.0, 1.0, 0.0)
    except ValueError:
        warnings.warn(
            "guess_beam_parameters: warning in fit to psf, using 1 pixel stddev"
        )
        beam_pixels = (1.0, 1.0, 0.0)

    return _convert_clean_beam_to_degrees(img, beam_pixels)


def project_sky_to_image(
    sky: SkyModel,
    phase_center: Union[List[int], List[float]],
    imaging_cellsize: float,
    imaging_npixel: int,
    filter_outlier: bool = True,
    invert_ra: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Calculates the pixel coordinates `sky` sources as floats.
    If you want to have integer indices, just round them.

    :param sky: `SkyModel` with the sources
    :param phase_center: [RA,DEC]
    :param imaging_cellsize: Image cellsize in radian (pixel coverage)
    :param imaging_npixel: Number of pixels of the image
    :param filter_outlier: Exclude source outside of image?
    :param invert_ra: Invert RA axis?

    :return: image-coordinates as np.ndarray[px,py] and
    `SkyModel` sources indices as np.ndarray[idxs]
    """

    # calc WCS args
    def radian_degree(rad: float) -> float:
        return rad * (180 / np.pi)

    cdelt = radian_degree(imaging_cellsize)
    crpix = np.floor((imaging_npixel / 2)) + 1

    # setup WCS
    w = WCS(naxis=2)
    w.wcs.crpix = np.array([crpix, crpix])  # coordinate reference pixel per axis
    ra_sign = -1 if invert_ra else 1
    w.wcs.cdelt = np.array(
        [ra_sign * cdelt, cdelt]
    )  # coordinate increments on sphere per axis
    w.wcs.crval = phase_center
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]  # coordinate axis type

    # convert coordinates
    px, py = w.wcs_world2pix(sky[:, 0], sky[:, 1], 1)

    # check length to cover single source pre-filtering
    if len(px.shape) == 0 and len(py.shape) == 0:
        px, py = [px], [py]
        idxs = np.arange(sky.num_sources)
    # post processing, pre filtering before calling wcs.wcs_world2pix would be
    # more efficient, however this has to be done in the ra-dec space.
    # Maybe for future work!?
    elif filter_outlier:
        px_idxs = np.where(np.logical_and(px <= imaging_npixel, px >= 0))[0]
        py_idxs = np.where(np.logical_and(py <= imaging_npixel, py >= 0))[0]
        idxs = np.intersect1d(px_idxs, py_idxs)
        px, py = px[idxs], py[idxs]
    else:
        idxs = np.arange(sky.num_sources)
    img_coords = np.array([px, py])

    return img_coords, idxs
