import subprocess
from typing import cast

import numpy as np
from astropy.convolution import Gaussian2DKernel
from numpy.typing import NDArray
from scipy.special import j1

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


def generate_eidos_beam(
    npixels: int,
    image_width_degrees: float,
    frequencies: NDArray[np.float_],
    stokes: str = "I",
) -> None:
    """This function generates a beam using EIDOS. Takes Image details as input and
    saves the primary beams in same location as the code in which the function is
    called in

    Parameters:
    npixels: size of the image in pixels
    image_width_degrees: width of the image in degrees
    frequencies: frequency bins for which you want a primary beam
    stoke: default 'I'
    """

    for freq in frequencies:
        cmd = [
            "eidos",
            "-p",
            str(npixels),
            "-d",
            str(image_width_degrees),
            "-f",
            str(freq),
            "-S",
            str(stokes),
        ]
        print(f"Running EIDOS for frequency {freq} MHz")
        subprocess.run(cmd, check=True)


def generate_airy_beam_data(
    fwhm_pixels: float, x_size: int, y_size: int
) -> NDArray[np.float_]:
    """Given a FWHM in pixel units, and a size in x and y coordinates,
    return a 2D array of shape (x_size, y_size) containing normalized Airy values
    (such that the central value of the 2D array is 1.0).
    """
    # Create coordinate grid
    y, x = np.indices((y_size, x_size))
    center_x = (x_size - 1) / 2
    center_y = (y_size - 1) / 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Convert FWHM to first null of the Airy disk
    # The first zero of the Airy function occurs at 1.22 * lambda / D
    # For simulation, we scale r accordingly
    first_zero = fwhm_pixels / (
        2 * np.sqrt(2 * np.log(2))
    )  # approximate matching to Gaussian FWHM
    normalized_r = (r / first_zero) * (1.22 * np.pi)

    # Avoid division by zero at the center
    normalized_r[normalized_r == 0] = 1e-10

    # Airy pattern
    airy_pattern = (2 * j1(normalized_r) / normalized_r) ** 2

    # Normalize peak to 1
    airy_pattern /= np.max(airy_pattern)

    return airy_pattern


def airy_beam_fwhm_for_frequency(frequency_hz: float, dish_diameter_m: float) -> float:
    """
    Compute the Airy beam FWHM in degrees for a given observing frequency and dish
    diameter.

    Parameters:
    -----------
    frequency_hz : float
        Observing frequency in Hz.
    dish_diameter_m : float
        Diameter of the telescope dish in meters.

    Returns:
    --------
    fwhm_degrees : float
        FWHM of the Airy beam in degrees.
    """
    c = 299792458.0  # Speed of light in m/s
    wavelength_m = c / frequency_hz
    fwhm_radians = 0.61 * wavelength_m / dish_diameter_m
    fwhm_degrees = np.degrees(fwhm_radians)
    return fwhm_degrees
