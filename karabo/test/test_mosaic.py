import tempfile

import astropy.units as u
import numpy as np
import pytest
from astropy import coordinates as coords
from astropy.io import fits

from karabo.data.external_data import (
    DilutedBATTYESurveyDownloadObject,
    SingleFileDownloadObject,
    cscs_karabo_public_base_url,
)
from karabo.imaging.mosaic import mosaic, mosaic_directories, mosaic_header
from karabo.simulation.line_emission import freq_channels, karabo_reconstruction
from karabo.simulation.sky_model import SkyModel


# DownloadObject instances used to download different golden files:
# - FITS file of the uncorrected (for primary beam) mosaic
# - FITS file of areas covered by the patches which form the uncorrected mosaic.
@pytest.fixture
def uncorrected_mosaic_fits_filename() -> str:
    return "test_mosaic_uncorrected.fits"


@pytest.fixture
def uncorrected_area_fits_filename() -> str:
    return "test_mosaic_uncorrected_area.fits"


@pytest.fixture
def uncorrected_mosaic_fits_downloader(
    uncorrected_mosaic_fits_filename,
) -> SingleFileDownloadObject:
    return SingleFileDownloadObject(
        remote_file_path=uncorrected_mosaic_fits_filename,
        remote_base_url=cscs_karabo_public_base_url,
    )


@pytest.fixture
def uncorrected_area_fits_downloader(
    uncorrected_area_fits_filename,
) -> SingleFileDownloadObject:
    return SingleFileDownloadObject(
        remote_file_path=uncorrected_area_fits_filename,
        remote_base_url=cscs_karabo_public_base_url,
    )


def test_mosaic_run(
    uncorrected_mosaic_fits_filename: str,
    uncorrected_area_fits_filename: str,
    uncorrected_mosaic_fits_downloader: SingleFileDownloadObject,
    uncorrected_area_fits_downloader: SingleFileDownloadObject,
) -> None:
    """
    Executes the mosaic pipeline and validate the output files.

    Args:
        uncorrected_mosaic_fits_filename:
            Name of FITS file containing the mosaic of the dirty image.
        uncorrected_area_fits_filename:
            Name of FITS file containing the coverage of the pointings in the mosaic.
    """
    # Download golden files for comparison
    golden_uncorrected_mosaic_fits_path = uncorrected_mosaic_fits_downloader.get()
    golden_uncorrected_area_fits_path = uncorrected_area_fits_downloader.get()

    # Load sky model data
    survey = DilutedBATTYESurveyDownloadObject()
    catalog_path = survey.get()

    # Set sky position for sky outcut
    ra = 20
    dec = -30
    outer_rad = 3

    sky_pointing = SkyModel.sky_from_h5_with_redshift_filtered(
        path=catalog_path, ra_deg=ra, dec_deg=dec, outer_rad=outer_rad
    )
    sky_pointing.compute()

    redshift_channel, freq_channel, freq_bin, freq_mid = freq_channels(
        sky_pointing.sources[:, 13], 10
    )

    # Setting for pointings
    img_size = 1024
    fac = 0.87
    FWHM_real = np.sqrt(89.5 * 86.2) / 60.0 * (1e3 / (freq_mid / 10**6)) * u.deg
    size_pntg = 1.2 * FWHM_real
    offset = FWHM_real * fac
    center1 = coords.SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    center2 = coords.SkyCoord(ra=ra * u.deg + offset, dec=dec * u.deg, frame="icrs")
    pointings = [center1, center2]

    # Settings for mosaic
    location = "20.84, -30.0"  # Center of the mosaic
    size_w = 4.0 * u.deg  # Width of the mosaic
    size_h = 2.5 * u.deg  # Height of the mosaic

    # Directory containing output files for validation
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = tmpdir + "/Mosaic_test"
        mosaic_directories(workdir, overwrite=True)
        uncorrected_mosaic_fits_path = workdir + "/mosaic_uncorrected.fits"
        uncorrected_area_fits_path = workdir + "/mosaic_uncorrected_area.fits"

        # Simulate dirty images
        outfile = workdir + "/unused_output/pointing"

        for k in range(len(pointings)):
            print("Reconstruction of pointing " + str(k) + "...")

            karabo_reconstruction(
                outfile=outfile + str(k),
                mosaic_pntg_file=workdir + "/raw/pointing" + str(k),
                sky=sky_pointing,
                ra_deg=pointings[k].ra.deg,
                dec_deg=pointings[k].dec.deg,
                img_size=img_size,
                start_freq=freq_mid,
                freq_bin=freq_channel[0] - freq_channel[-1],
                beam_type="Gaussian beam",
                cut=size_pntg.value,
                channel_num=1,
                pdf_plot=False,
                circle=True,
            )

        # Create mosaic
        mosaic_header(
            output_directory_path=workdir,
            location=location,
            width=size_w.value,
            height=size_h.value,
            resolution=20.0,
            sin_projection=True,
        )
        mosaic(output_directory_path=workdir)

        # Verify mosaic FITS file
        uncorrected_mosaic_fits_data, uncorrected_mosaic_fits_header = fits.getdata(
            uncorrected_mosaic_fits_path, ext=0, header=True
        )
        (
            golden_uncorrected_mosaic_fits_data,
            golden_uncorrected_mosaic_fits_header,
        ) = fits.getdata(golden_uncorrected_mosaic_fits_path, ext=0, header=True)

        # Check FITS data is close to goldenfile
        assert np.allclose(
            golden_uncorrected_mosaic_fits_data,
            uncorrected_mosaic_fits_data,
            equal_nan=True,
        )

        # Check that headers contain the same keys
        assert set(golden_uncorrected_mosaic_fits_header.keys()) == set(
            uncorrected_mosaic_fits_header.keys()
        )

        # Verify mosaic area FITS file
        uncorrected_area_fits_data, uncorrected_area_fits_header = fits.getdata(
            uncorrected_area_fits_path, ext=0, header=True
        )
        (
            golden_uncorrected_area_fits_data,
            golden_uncorrected_area_fits_header,
        ) = fits.getdata(golden_uncorrected_area_fits_path, ext=0, header=True)

        # Check FITS data is close to goldenfile
        assert np.allclose(
            golden_uncorrected_area_fits_data,
            uncorrected_area_fits_data,
            equal_nan=True,
        )

        # Check that headers contain the same keys
        assert set(golden_uncorrected_area_fits_header.keys()) == set(
            uncorrected_area_fits_header.keys()
        )
