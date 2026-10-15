from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from ska_sdp_datamodels.image import import_image_from_fits
from ska_sdp_datamodels.image.image_model import Image as SdpImage
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

from karabo.util._types import FilePathType


def _polarisation_frame_from_wcs(wcs: WCS, npol: int) -> PolarisationFrame:
    if npol == 1:
        return PolarisationFrame("stokesI")

    codes = np.asarray(wcs.sub(["stokes"]).wcs_pix2world(range(npol), 0)[0], dtype=int)
    for name, expected_codes in PolarisationFrame.fits_codes.items():
        if np.array_equal(codes, np.asarray(expected_codes)):
            return PolarisationFrame(name)
    raise ValueError(f"Cannot determine polarisation frame from FITS codes {codes}")


def _expand_wcs_to_four_axes(wcs: WCS, header: fits.Header, third_axis: str) -> WCS:
    celestial = wcs.celestial
    expanded = WCS(naxis=4)
    expanded.wcs.ctype = [
        celestial.wcs.ctype[0],
        celestial.wcs.ctype[1],
        "STOKES",
        "FREQ",
    ]
    expanded.wcs.crpix = [
        float(celestial.wcs.crpix[0]),
        float(celestial.wcs.crpix[1]),
        1.0,
        1.0,
    ]
    expanded.wcs.crval = [
        float(celestial.wcs.crval[0]),
        float(celestial.wcs.crval[1]),
        float(header.get("CRVAL3", 1.0)) if third_axis == "STOKES" else 1.0,
        float(header.get("CRVAL3", 1.0e8)) if third_axis == "FREQ" else 1.0e8,
    ]
    expanded.wcs.cdelt = [
        float(celestial.wcs.cdelt[0]),
        float(celestial.wcs.cdelt[1]),
        float(header.get("CDELT3", 1.0)) if third_axis == "STOKES" else 1.0,
        float(header.get("CDELT3", 1.0)) if third_axis == "FREQ" else 1.0,
    ]
    return expanded


def import_sdp_image_from_fits(
    fits_path: Union[str, FilePathType], *, fix_polarisation_order: bool = True
) -> SdpImage:
    """Read FITS with SDP, retaining Karabo's supported formats and ordering.

    SDP 1.4.1 handles canonical and axis-swapped 4D cubes. Its importer cannot
    expand 2D/3D images or interpret non-I singleton polarization planes, so
    those cases use the same data model's constructor in memory. Validation and
    the historical polarization permutation remain here to avoid silently
    changing previously supported inputs or their scientific interpretation.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        if hdul[0].data is None:
            raise ValueError(f"FITS image has no data: {fits_path}")
        data = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()

    return _sdp_image_from_fits_data(
        data,
        header,
        fits_path=fits_path,
        fix_polarisation_order=fix_polarisation_order,
    )


def _sdp_image_from_fits_data(
    data: np.ndarray,
    header: fits.Header,
    *,
    fits_path: Optional[FilePathType] = None,
    fix_polarisation_order: bool = True,
) -> SdpImage:
    """Shared in-memory normalization for formats SDP cannot import directly.

    A supplied path enables upstream import of supported 4D files. Without one,
    this also supports SkyModel's squeezed celestial plane without a disk write.
    """
    original_ndim = data.ndim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header)

    if data.ndim == 2:
        data = data[np.newaxis, np.newaxis, :, :]
        wcs = _expand_wcs_to_four_axes(wcs, header, "")
    elif data.ndim == 3:
        third_axis = str(wcs.wcs.ctype[2]).upper()
        if third_axis == "FREQ":
            data = data[:, np.newaxis, :, :]
        elif third_axis == "STOKES":
            data = data[np.newaxis, :, :, :]
        else:
            raise NotImplementedError(
                f"Unsupported third FITS image axis {wcs.wcs.ctype[2]!r}"
            )
        wcs = _expand_wcs_to_four_axes(wcs, header, third_axis)
    elif data.ndim == 4:
        axis_types = [str(axis).upper() for axis in wcs.wcs.ctype]
        if axis_types[2] == "FREQ" and axis_types[3] == "STOKES":
            wcs = wcs.swapaxes(2, 3)
            data = np.transpose(data, (1, 0, 2, 3))
        elif axis_types[2] != "STOKES" or axis_types[3] != "FREQ":
            raise NotImplementedError(
                "Expected FITS axes 3/4 to be STOKES/FREQ or FREQ/STOKES, "
                f"received {axis_types[2:]}"
            )
    else:
        raise ValueError(f"Expected a 2D, 3D, or 4D FITS image, got {data.shape}")

    polarisation_frame = _polarisation_frame_from_wcs(wcs, data.shape[1])
    # Karabo has always interpreted singleton planes as Stokes I, even when the
    # header says Q or XX. SDP cannot construct these as singleton images.
    native_polarisation = (
        data.shape[1] > 1 or int(wcs.sub(["stokes"]).wcs_pix2world([0], 0)[0][0]) == 1
    )
    if fits_path is not None and original_ndim == 4 and native_polarisation:
        # Disabling memmap also preserves support for BSCALE/BZERO/BLANK images.
        with fits.conf.set_temp("use_memmap", False):
            image = import_image_from_fits(str(fits_path), fixpol=False)
    else:
        clean_beam = None
        if all(key in header for key in ("BMAJ", "BMIN", "BPA")):
            clean_beam = {
                "bmaj": float(header["BMAJ"]),
                "bmin": float(header["BMIN"]),
                "bpa": float(header["BPA"]),
            }
        image = SdpImage.constructor(
            data=data,
            polarisation_frame=polarisation_frame,
            wcs=wcs,
            clean_beam=clean_beam,
        )

    if fix_polarisation_order:
        # Deliberately keep Karabo's gather permutation. SDP's fixpol=True uses
        # the inverse (scatter) permutation for linear/circular correlations.
        permutation = PolarisationFrame.fits_to_datamodels[polarisation_frame.type]
        image["pixels"].data = image["pixels"].data[:, permutation, ...]
    return image
