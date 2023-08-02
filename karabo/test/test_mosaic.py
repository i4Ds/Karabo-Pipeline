import tempfile

import astropy.units as u
import numpy as np
from astropy import coordinates as coords

from karabo.data.external_data import DilutedBATTYESurveyDownloadObject
from karabo.imaging.mosaic import mosaic, mosaic_directories, mosaic_header
from karabo.simulation.line_emission import freq_channels, karabo_reconstruction
from karabo.simulation.sky_model import SkyModel


def test_mosaic_run() -> None:
    """
    Executes the mosaic pipeline and validate the output files.
    """
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
    pointings = [
        center1,
    ] + [center2]

    # Settings for mosaic
    location = "20.84, -30.0"
    size_w = 4.0 * u.deg
    size_h = 2.5 * u.deg

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = tmpdir + "/Mosaic_test"
        mosaic_directories(workdir)

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
