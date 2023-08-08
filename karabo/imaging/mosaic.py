import os
import shutil
from pathlib import Path
from typing import Optional

from MontagePy.main import mAdd, mHdr, mImgtbl, mProjExec

from karabo.util._types import DirPathType


def mosaic_directories(
    output_directory_path: DirPathType, overwrite: bool = False
) -> None:
    """
    Creating a directory structure which can be used for coadding several fits files
    with MontagePy.

    :param output_directory_path: The new folder (path) which is created and in which
                                  all subfolders used for the mosaic are created.
                                  A graphical representation of the directory structure
                                  created:
                                  output_directory_path
                                  |-projected
                                  |-raw
                                  |-unused_output
    :param overwrite: If the directory already exists and overwrite is True, then the
                      directory is overwritten.
    """
    output_directory_path = Path(output_directory_path)

    if os.path.exists(output_directory_path):
        if not overwrite:
            raise FileExistsError(
                f"Could not delete {output_directory_path}, because"
                f"overwrite is False. Set overwrite to True, if you "
                f"want {output_directory_path} to be overwritten."
            )
        shutil.rmtree(output_directory_path)

    os.makedirs(output_directory_path)

    os.makedirs(output_directory_path / "raw")
    os.makedirs(output_directory_path / "projected")
    os.makedirs(output_directory_path / "unused_output")


def mosaic_header(
    output_directory_path: DirPathType,
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
                           a dirty image are in SIN projection. So if you want your that
                           your mosaic is in the same format as the outputs from karabo
                           set sin_projection to True.
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


def mosaic(
    output_directory_path: DirPathType,
    image_directory: DirPathType = "raw",
    projected_directory: DirPathType = "projected",
) -> None:
    """
    Coadds the images saved in the "raw" folder by using the mean value. This function
    does not do any corrections.

    :param output_directory_path: The folder path in which all subfolders used for the
                                  mosaic are.
    :param image_directory: The name of the subfolder in which the images to be coadded
                            are saved.
    :param projected_directory: The name of the subfolder in which the reprojected
                                images will be stored.
    """
    output_directory_path = Path(output_directory_path)
    image_directory = Path(image_directory)

    # Scan the images for their coverage metadata.
    rtn = mImgtbl(
        str(output_directory_path / image_directory),
        str(output_directory_path / "rimages.tbl"),
    )
    print("mImgtbl (raw):    " + str(rtn), flush=True)

    # Reproject the original images to the  frame of the output FITS header we created
    rtn = mProjExec(
        str(output_directory_path / image_directory),
        str(output_directory_path / "rimages.tbl"),
        str(output_directory_path / "region.hdr"),
        projdir=str(output_directory_path / projected_directory),
    )
    print("mProjExec:           " + str(rtn), flush=True)

    mImgtbl(
        str(output_directory_path / projected_directory),
        str(output_directory_path / "pimages.tbl"),
    )
    print("mImgtbl (projected): " + str(rtn), flush=True)

    # Coaddition
    rtn = mAdd(
        str(output_directory_path / projected_directory),
        str(output_directory_path / "pimages.tbl"),
        str(output_directory_path / "region.hdr"),
        str(output_directory_path / "mosaic_uncorrected.fits"),
    )
    print("mAdd:                " + str(rtn), flush=True)
