import numpy as np
import pytest
from astropy import constants as consts
from astropy.io import fits
from astropy.wcs import WCS

from karabo.imaging.image import Image
from karabo.imaging.sdp_io import import_sdp_image_from_fits


def _canonical_wcs() -> WCS:
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
    wcs.wcs.crpix = [3.0, 2.0, 1.0, 1.0]
    wcs.wcs.crval = [15.0, -30.0, 1.0, 1.0e8]
    wcs.wcs.cdelt = [-0.01, 0.01, 1.0, 1.0e6]
    return wcs


def test_import_sdp_image_from_canonical_fits(tmp_path) -> None:
    data = np.arange(2 * 4 * 3 * 5, dtype=float).reshape(2, 4, 3, 5)
    header = _canonical_wcs().to_header()
    header["BMAJ"] = 0.1
    header["BMIN"] = 0.05
    header["BPA"] = 23.0
    fits_path = tmp_path / "cube.fits"
    fits.writeto(fits_path, data, header)

    image = import_sdp_image_from_fits(fits_path)

    np.testing.assert_array_equal(image["pixels"].data, data)
    np.testing.assert_allclose(image["frequency"].data, [1.0e8, 1.01e8])
    assert image.image_acc.polarisation_frame.type == "stokesIQUV"
    assert image.attrs["clean_beam"] == {
        "bmaj": 0.1,
        "bmin": 0.05,
        "bpa": 23.0,
    }
    assert list(image.image_acc.wcs.wcs.ctype) == [
        "RA---SIN",
        "DEC--SIN",
        "STOKES",
        "FREQ",
    ]


def test_import_sdp_image_expands_two_dimensional_fits(tmp_path) -> None:
    data = np.arange(12, dtype=float).reshape(3, 4)
    celestial_wcs = WCS(naxis=2)
    celestial_wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    celestial_wcs.wcs.crpix = [2.0, 2.0]
    celestial_wcs.wcs.crval = [15.0, -30.0]
    celestial_wcs.wcs.cdelt = [-0.01, 0.01]
    fits_path = tmp_path / "plane.fits"
    fits.writeto(fits_path, data, celestial_wcs.to_header())

    image = import_sdp_image_from_fits(fits_path)

    assert image["pixels"].shape == (1, 1, 3, 4)
    np.testing.assert_array_equal(image["pixels"].data[0, 0], data)
    assert image.image_acc.polarisation_frame.type == "stokesI"


def test_power_spectrum_of_central_point_source() -> None:
    size = 16
    frequency = 1.0e8
    resolution = 5.0e-4
    data = np.zeros((1, 1, size, size))
    data[0, 0, size // 2, size // 2] = 1.0
    image = Image(data=data, header=_canonical_wcs().to_header())

    profile, theta = image.get_power_spectrum(resolution=resolution, signal_channel=0)

    omega = np.pi * resolution**2 / (4.0 * np.log(2.0))
    wavelength = consts.c / frequency
    kelvin_per_jansky = (1.0e-26 * wavelength**2 / (2.0 * consts.k_B * omega)).value
    np.testing.assert_allclose(profile, kelvin_per_jansky / size**2)
    assert np.all(np.isfinite(theta))
    assert np.all(np.diff(theta) < 0.0)

    with pytest.raises(IndexError, match="outside"):
        image.get_power_spectrum(signal_channel=1)
