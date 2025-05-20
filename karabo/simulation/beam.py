from typing import cast

import numpy as np
from astropy.convolution import Gaussian2DKernel
from numpy.typing import NDArray

# Reference values below were obtained for a MeerKAT-like beam
REFERENCE_FWHM_DEGREES = 1.8
REFERENCE_FREQUENCY_HZ = 8e8


def gaussian_beam_fwhm_for_frequency(
    desired_frequency: float,
    reference_fwhm_degrees: float = REFERENCE_FWHM_DEGREES,
    reference_frequency_Hz: float = REFERENCE_FREQUENCY_HZ,
) -> float:
    return reference_fwhm_degrees * reference_frequency_Hz / desired_frequency


def generate_gaussian_beam_data(
    fwhm_pixels: float,
    x_size: int,
    y_size: int,
) -> NDArray[np.float_]:
    """Given a FWHM in pixel units, and a size in x and y coordinates,
    return a 2D array of shape (x_size, y_size) containing normalized Gaussian values
    (such that the central value of the 2D array is 1.0).
    """
    sigma = fwhm_pixels / (2 * np.sqrt(2 * np.log(2)))
    gauss_kernel = Gaussian2DKernel(
        sigma,
        x_size=x_size,
        y_size=y_size,
    )
    beam = cast(NDArray[np.float_], gauss_kernel.array)
    beam = beam / np.max(beam)

    return beam
