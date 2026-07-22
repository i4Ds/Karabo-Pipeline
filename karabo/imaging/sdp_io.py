from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
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
    """Read a FITS image into the pinned SKA-SDP image data model.

    This is the small FITS-I/O boundary missing from ska-sdp-datamodels 0.1.3.
    It accepts 2D images and canonical 3D/4D FITS cubes and preserves clean-beam
    metadata when present.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        if hdul[0].data is None:
            raise ValueError(f"FITS image has no data: {fits_path}")
        data = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()

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
    if fix_polarisation_order:
        permutation = PolarisationFrame.fits_to_datamodels[polarisation_frame.type]
        data = data[:, permutation, ...]

    clean_beam = None
    if all(key in header for key in ("BMAJ", "BMIN", "BPA")):
        clean_beam = {
            "bmaj": float(header["BMAJ"]),
            "bmin": float(header["BMIN"]),
            "bpa": float(header["BPA"]),
        }

    return SdpImage.constructor(
        data=data,
        polarisation_frame=polarisation_frame,
        wcs=wcs,
        clean_beam=clean_beam,
    )
