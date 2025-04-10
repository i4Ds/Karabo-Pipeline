"""This module is to create according survey-files for Karabo."""
from __future__ import annotations

import os
import tempfile

import astropy.units as u
import pandas as pd
from astropy.io import fits
from astropy.table.table import Table
from astropy.units import UnitBase

from karabo.data.external_data import DownloadObject
from karabo.util._types import DirPathType


def create_MALS_survey_as_fits(
    directory: DirPathType,
    version: int = 3,
    check_for_updates: bool = False,
    verbose: bool = True,
) -> str:
    """Creates MALS (https://mals.iucaa.in/) survey as a .fits.gz file.

    It takes care of downloading the file from
    'https://mals.iucaa.in/catalogue/catalogue_malsdr1v{0}_all.csv', and convert it
    into a .fits.gz file. Downloading the .csv catalogue may take a while, because
    it is about 3.6GB large and downloading-speed depends on some uncontrollable
    factors. However, if you already have the .csv catalogue, just put it into
    `directory` to take it from the disk-cache. All file-products (.csv & .fits.gz)
    are saved in and loaded from `directory`.

    This is just a utility function and is not meant to be embedded in any
    library-code.

    In case the .fits.gz  file already exists, it just returns the the file-path
    without doing anything like downloading or creating any file.

    Args:
        directory: Directory to save and load the according catalogue files.
        version: Survey version.
        check_for_updates: Also check for new updates?
        verbose: Verbose?

    Returns:
        .fits.gz file-path.
    """
    directory = str(directory)
    try:
        version = int(version)  # run-time check
    except ValueError as e:
        raise TypeError(f"{version=} must be an integer!") from e
    base_url = "https://mals.iucaa.in"
    remote_path = "/catalogue/"
    fname_csv_template = "catalogue_malsdr1v{0}_all.csv"
    fname_csv = fname_csv_template.format(version)
    url = base_url + remote_path + fname_csv
    if check_for_updates:
        update_address = base_url + remote_path + fname_csv_template.format(version + 1)
        if DownloadObject.is_url_available(update_address):
            print(
                f"An updated version of {url} is available! Please "
                + f"have a look at {base_url}"
            )
    local_csv_file = os.path.join(directory, fname_csv)
    if not local_csv_file.endswith(
        ".csv"
    ):  # should be impossible, but just for double-checking
        raise ValueError(f"{local_csv_file=} is not a valid csv file!")
    local_fits_file = local_csv_file[:-4] + ".fits.gz"
    if os.path.exists(local_fits_file):
        if verbose:
            print(f"{local_fits_file} already exists.")
        return local_fits_file
    if not os.path.exists(local_csv_file):
        _ = DownloadObject.download(
            url=url, local_file_path=local_csv_file, verify=False, verbose=verbose
        )  # about 3.6GB, may take VERY long

    if verbose:
        print(f"Read {local_csv_file} from disk ...")
    df = pd.read_csv(local_csv_file)  # takes about 40-45 s
    # source-uniqueness is only ensured for a combination of pointing-id and spw-id
    cols_of_interest = [
        "ra_max",
        "dec_max",
        "peak_flux",
        "ref_freq",
        "maj",
        "min",
        "pa",
        "source_name",
        "pointing_name",
        "spw_id",
    ]
    units: dict[str, UnitBase] = {
        "ra_max": u.deg,
        "dec_max": u.deg,
        "peak_flux": u.mJy / u.beam,
        "ref_freq": u.MHz,
        "maj": u.arcsec,
        "min": u.arcsec,
        "pa": u.deg,
    }
    table = Table.from_pandas(df[cols_of_interest], units=units)
    if verbose:
        print(f"Creating {local_fits_file} ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fits = os.path.join(tmpdir, "mals.fits")
        table.write(tmp_fits, format="fits")  # about 368MB
        with fits.open(tmp_fits) as f:
            f.writeto(local_fits_file)

    return local_fits_file
