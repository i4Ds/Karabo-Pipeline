import numpy as np
import pytest
from astropy import constants as consts
from astropy.io import fits
from astropy.wcs import WCS
from ska_sdp_datamodels.image import import_image_from_fits
from ska_sdp_datamodels.image.image_model import Image as SdpImage
from ska_sdp_datamodels.science_data_model import PolarisationFrame

import karabo.imaging.sdp_io as sdp_io
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


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize(
    "frame,npol,code,step",
    [
        ("stokesI", 1, 1, 1),
        ("stokesIQUV", 4, 1, 1),
        ("stokesIQ", 2, 1, 1),
        ("stokesIV", 2, 1, 3),
        ("linear", 4, -5, -1),
        ("circular", 4, -1, -1),
        ("linearnp", 2, -5, -1),
        ("circularnp", 2, -1, -1),
    ],
)
@pytest.mark.parametrize("fixpol", [False, True])
def test_import_delegates_four_dimensional_fits(
    tmp_path, monkeypatch, swapped, frame, npol, code, step, fixpol
) -> None:
    wcs = _canonical_wcs()
    wcs.wcs.crval[2] = code
    wcs.wcs.cdelt[2] = step
    wcs.wcs.crpix[3] = 2.5
    data = np.arange(2 * npol * 3 * 5, dtype=float).reshape(2, npol, 3, 5)
    header = (wcs.swapaxes(2, 3) if swapped else wcs).to_header()
    header.update(BMAJ=0.1, BMIN=0.05, BPA=23.0)
    path = tmp_path / "cube.fits"
    fits.writeto(path, data.transpose(1, 0, 2, 3) if swapped else data, header)
    calls = []

    def upstream(*args, **kwargs):
        calls.append((args, kwargs))
        return import_image_from_fits(*args, **kwargs)

    monkeypatch.setattr(sdp_io, "import_image_from_fits", upstream)
    image = import_sdp_image_from_fits(path, fix_polarisation_order=fixpol)
    assert calls == [((str(path),), {"fixpol": False})]
    # Explicit legacy ordering; do not derive the expectation from the wrapper.
    order = [0, 3, 1, 2] if fixpol and frame in ("linear", "circular") else range(npol)
    np.testing.assert_array_equal(image.pixels.data, data[:, order])
    np.testing.assert_allclose(image.frequency.data, [98.5e6, 99.5e6])
    assert image.image_acc.polarisation_frame.type == frame
    reference = import_image_from_fits(str(path), fixpol=False)
    assert (
        image.image_acc.wcs.to_header_string()
        == reference.image_acc.wcs.to_header_string()
    )
    assert image.attrs["clean_beam"] == {"bmaj": 0.1, "bmin": 0.05, "bpa": 23.0}


@pytest.mark.parametrize(
    "layout", ["2d", "frequency", "stokes", "singleton", "swapped-singleton"]
)
@pytest.mark.parametrize("beam", ["complete", "partial", "absent"])
def test_import_compatibility_formats(tmp_path, monkeypatch, layout, beam) -> None:
    wcs = _canonical_wcs()
    if layout == "2d":
        data = np.arange(15.0).reshape(3, 5)
        input_wcs = wcs.celestial
        expected = data[None, None]
        frame = "stokesI"
    elif layout == "frequency":
        data = np.arange(30.0).reshape(2, 3, 5)
        input_wcs = wcs.sub([1, 2, 4])
        input_wcs.wcs.crpix[2] = 2.5
        expected = data[:, None]
        frame = "stokesI"
    elif layout == "stokes":
        data = np.arange(60.0).reshape(4, 3, 5)
        input_wcs = wcs.sub([1, 2, 3])
        expected = data[None]
        frame = "stokesIQUV"
    else:
        wcs.wcs.crval[2] = -5  # Historically interpreted as I, not XX.
        data = np.arange(30.0).reshape(2, 1, 3, 5)
        expected = data
        input_wcs = wcs
        frame = "stokesI"
        if layout == "swapped-singleton":
            input_wcs = wcs.swapaxes(2, 3)
            data = data.transpose(1, 0, 2, 3)
    header = input_wcs.to_header()
    if beam != "absent":
        header.update(BMAJ=0.1, BMIN=0.05)
    if beam == "complete":
        header["BPA"] = 23.0
    path = tmp_path / "fallback.fits"
    fits.writeto(path, data, header)

    def unsupported(*args, **kwargs):
        pytest.fail("SDP 1.4.1 cannot import this format directly")

    monkeypatch.setattr(sdp_io, "import_image_from_fits", unsupported)
    image = import_sdp_image_from_fits(path)
    if layout in ("2d", "stokes"):
        wcs.wcs.cdelt[3] = 1.0  # Legacy default when no frequency axis exists.
    reference = SdpImage.constructor(expected, PolarisationFrame(frame), wcs)
    np.testing.assert_array_equal(image.pixels.data, expected)
    np.testing.assert_array_equal(image.frequency.data, reference.frequency.data)
    assert (
        image.image_acc.wcs.to_header_string()
        == reference.image_acc.wcs.to_header_string()
    )
    assert image.image_acc.polarisation_frame.type == frame
    assert image.attrs["clean_beam"] == (
        {"bmaj": 0.1, "bmin": 0.05, "bpa": 23.0} if beam == "complete" else None
    )


@pytest.mark.parametrize(
    "problem", ["empty", "1d", "5d", "3d-axis", "4d-axes", "polarisation"]
)
def test_import_rejects_unsupported_fits(tmp_path, monkeypatch, problem) -> None:
    wcs = _canonical_wcs()
    data = np.zeros((2, 4, 3, 5))
    error = ValueError
    if problem == "empty":
        data = None
        match = "no data"
    elif problem in ("1d", "5d"):
        data = np.zeros((3,) if problem == "1d" else (1, 1, 1, 3, 5))
        match = "Expected a 2D, 3D, or 4D"
    elif problem == "3d-axis":
        data = data[0]
        wcs = wcs.sub([1, 2, 3])
        wcs.wcs.ctype[2] = "TIME"
        error, match = NotImplementedError, "Unsupported third"
    elif problem == "4d-axes":
        wcs.wcs.ctype[2] = "TIME"
        error, match = NotImplementedError, "Expected FITS axes"
    else:
        wcs.wcs.crval[2] = 7
        match = "Cannot determine polarisation"
    path = tmp_path / "invalid.fits"
    fits.PrimaryHDU(data, wcs.to_header()).writeto(path)

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid input must be rejected before SDP's permissive fallback")

    monkeypatch.setattr(sdp_io, "import_image_from_fits", unexpected)
    with pytest.raises(error, match=match):
        import_sdp_image_from_fits(path)


def test_import_scaled_integer_fits(tmp_path) -> None:
    data = np.arange(30, dtype=np.int16).reshape(2, 1, 3, 5)
    data[0, 0, 0, 0] = -32768
    header = _canonical_wcs().to_header()
    hdu = fits.PrimaryHDU(data, header)
    hdu.header.update(BSCALE=0.5, BZERO=3.0, BLANK=-32768)
    path = tmp_path / "scaled.fits"
    hdu.writeto(path)
    original_memmap = fits.conf.use_memmap
    image = import_sdp_image_from_fits(path)
    expected = data.astype(float) * 0.5 + 3.0
    expected[data == -32768] = np.nan
    np.testing.assert_allclose(image.pixels.data, expected)
    assert fits.conf.use_memmap == original_memmap


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_sdp_backend_and_mgcls_share_fits_compatibility(
    tmp_path, monkeypatch, ndim
) -> None:
    from karabo.data.external_data import MGCLSContainerDownloadObject
    from karabo.imaging.backends.sdp_backend import SdpImager
    from karabo.imaging.util import get_MGCLS_images

    data = np.arange(24.0).reshape(1, 1, 4, 6)
    wcs = _canonical_wcs()
    if ndim == 2:
        input_data, input_wcs = data[0, 0], wcs.celestial
    elif ndim == 3:
        input_data, input_wcs = data[:, 0], wcs.sub([1, 2, 4])
    else:
        input_data, input_wcs = data, wcs
    path = tmp_path / "shared.fits"
    fits.writeto(path, input_data, input_wcs.to_header())
    monkeypatch.setattr(
        MGCLSContainerDownloadObject, "get_all", lambda self, verbose=False: [str(path)]
    )
    calls = []

    def upstream(*args, **kwargs):
        calls.append(args[0])
        return import_image_from_fits(*args, **kwargs)

    monkeypatch.setattr(sdp_io, "import_image_from_fits", upstream)
    mgcls_image = get_MGCLS_images("shared.fits")[0]
    backend_image = SdpImager()._import_native_image(Image(path=path))
    for image in (mgcls_image, backend_image):
        np.testing.assert_array_equal(image.pixels.data, data)
    assert calls == ([str(path)] * 2 if ndim == 4 else [])


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
