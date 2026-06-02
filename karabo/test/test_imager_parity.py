import math

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from karabo.imaging.imager_factory import ImagingBackend, get_imager
from karabo.imaging.imager_interface import ImageSpec
from karabo.imaging.util import guess_beam_parameters
from karabo.simulation.visibility import Visibility


def _peak_index(data: np.ndarray) -> tuple[int, int]:
    return tuple(int(i) for i in np.unravel_index(np.nanargmax(data), data.shape))


def _chebyshev_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _windowed_peak_index(
    data: np.ndarray,
    center: tuple[int, int],
    half_size: int = 32,
) -> tuple[int, int]:
    y0 = max(center[0] - half_size, 0)
    y1 = min(center[0] + half_size + 1, data.shape[0])
    x0 = max(center[1] - half_size, 0)
    x1 = min(center[1] + half_size + 1, data.shape[1])
    local = data[y0:y1, x0:x1]
    ly, lx = _peak_index(local)
    return (y0 + ly, x0 + lx)


def _pixel_to_skycoord(image, pixel_yx: tuple[int, int]) -> SkyCoord:
    wcs = image.get_2d_wcs()
    x = float(pixel_yx[1])
    y = float(pixel_yx[0])
    ra_deg, dec_deg = wcs.wcs_pix2world(x, y, 0)
    return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")


@pytest.fixture(scope="function")
def minimal_backend_images(
    minimal_casa_ms: Visibility,
):
    spec = ImageSpec(
        npix=256,
        cellsize_arcsec=math.degrees(5e-5) * 3600.0,
        phase_centre_deg=(0.0, 0.0),
    )
    images = {}
    for backend in (ImagingBackend.RASCIL, ImagingBackend.SDP):
        imager = get_imager(backend)
        images[backend] = imager.invert(minimal_casa_ms, spec)
    return images


def test_rascil_sdp_psf_peak_parity(minimal_backend_images) -> None:
    _, psf_rascil = minimal_backend_images[ImagingBackend.RASCIL]
    _, psf_sdp = minimal_backend_images[ImagingBackend.SDP]

    peak_rascil = _peak_index(psf_rascil.get_squeezed_data())
    peak_sdp = _peak_index(psf_sdp.get_squeezed_data())
    assert _chebyshev_distance(peak_rascil, peak_sdp) <= 1


def test_rascil_sdp_psf_width_parity(minimal_backend_images) -> None:
    _, psf_rascil = minimal_backend_images[ImagingBackend.RASCIL]
    _, psf_sdp = minimal_backend_images[ImagingBackend.SDP]

    beam_rascil = guess_beam_parameters(psf_rascil)
    beam_sdp = guess_beam_parameters(psf_sdp)
    assert np.isclose(beam_rascil["bmaj"], beam_sdp["bmaj"], rtol=0.35, atol=5.0)
    assert np.isclose(beam_rascil["bmin"], beam_sdp["bmin"], rtol=0.35, atol=5.0)


@pytest.mark.xfail(
    reason=(
        "Known parity gap: dirty-image source peak location differs between the "
        "RASCIL and SDP adapters for the minimal CASA MS fixture."
    ),
    strict=False,
)
def test_rascil_sdp_dirty_peak_location_parity(minimal_backend_images) -> None:
    dirty_rascil, psf_rascil = minimal_backend_images[ImagingBackend.RASCIL]
    dirty_sdp, psf_sdp = minimal_backend_images[ImagingBackend.SDP]

    psf_peak_rascil = _peak_index(psf_rascil.get_squeezed_data())
    psf_peak_sdp = _peak_index(psf_sdp.get_squeezed_data())

    dirty_peak_rascil = _windowed_peak_index(
        dirty_rascil.get_squeezed_data(), psf_peak_rascil
    )
    dirty_peak_sdp = _windowed_peak_index(dirty_sdp.get_squeezed_data(), psf_peak_sdp)

    separation_arcsec = (
        _pixel_to_skycoord(dirty_rascil, dirty_peak_rascil)
        .separation(_pixel_to_skycoord(dirty_sdp, dirty_peak_sdp))
        .arcsec
    )
    assert separation_arcsec <= 30.0


@pytest.mark.xfail(
    reason="RASCIL restore is identity while SDP restore performs CLEAN;"
    " parity not expected yet.",
    strict=False,
)
def test_rascil_sdp_restored_flux_parity(minimal_backend_images) -> None:
    dirty_rascil, psf_rascil = minimal_backend_images[ImagingBackend.RASCIL]
    dirty_sdp, psf_sdp = minimal_backend_images[ImagingBackend.SDP]

    rascil_restored = get_imager(ImagingBackend.RASCIL).restore(
        dirty_rascil, psf_rascil
    )
    sdp_restored = get_imager(ImagingBackend.SDP).restore(dirty_sdp, psf_sdp)

    # Compare integrated flux in a central window; expect mismatch
    # until SDP CLEAN config is tuned.
    def _window_sum(img, half_size=32):
        data = img.get_squeezed_data()
        cy, cx = data.shape[0] // 2, data.shape[1] // 2
        return float(
            np.sum(
                data[cy - half_size : cy + half_size, cx - half_size : cx + half_size]
            )
        )

    r_sum = _window_sum(rascil_restored)
    s_sum = _window_sum(sdp_restored)
    assert np.isclose(r_sum, s_sum, rtol=0.25, atol=1e-3)
