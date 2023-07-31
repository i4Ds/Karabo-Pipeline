import os
import shutil
from pathlib import Path
from typing import Optional

from MontagePy.archive import *  # noqa
from MontagePy.main import mHdr

from karabo.util._types import FilePathType


def mosaic_directories(output_directory_path: FilePathType) -> None:
    """
    Creating a directory structure which can be used for coadding several fits files
    with MontagePy.

    :param output_directory_path: The new folder (path) which is created and in which
                                  all subfolders used for the mosaic are created.
    """
    output_directory_path = Path(output_directory_path)

    if os.path.exists(output_directory_path):
        shutil.rmtree(output_directory_path)

    os.makedirs(output_directory_path)

    os.makedirs(output_directory_path / "raw")
    os.makedirs(output_directory_path / "projected")
    os.makedirs(output_directory_path / "unused_output")


def mosaic_header(
    output_directory_path: FilePathType,
    location: str,
    width: float,
    height: Optional[float] = None,
    resolution: Optional[float] = 1.0,
    sin_projection: Optional[bool] = False,
) -> None:
    """
    Creates the header for the mosaic you want to create.

    :param output_directory_path: The folder path in which all subfolders used for the
                                  mosaic are.
    :param location: The center of the mosaic in "RA, Dec" in degrees (Equatorial).
    :param width: Image width in degrees.
    :param height: Image height in degrees.
    :param resolution: Image pixel resolution (in arcsec).
    :param sin_projection: Set to true if you want the output mosaic to be in SIN
                           projection. The output fits files from a karabo simulation of
                           a dirty image are in SIN projection.
    """
    output_directory_path = Path(output_directory_path)

    if height is None:
        height = width

    rtn = mHdr(
        locstr=location,
        width=width,
        height=height,
        outfile=str(output_directory_path / "region.hdr"),
        resolution=resolution,
        csys="Equatorial",
    )
    print("mHdr:             " + str(rtn), flush=True)

    if sin_projection:
        with open(output_directory_path / "region.hdr", "r") as file:
            data = file.readlines()
        data[5] = "CTYPE1  = 'RA---SIN'\n"
        data[6] = "CTYPE2  = 'DEC--SIN'\n"
        with open(output_directory_path / "region.hdr", "w") as file:
            file.writelines(data)
