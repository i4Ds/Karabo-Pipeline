import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from ska_sdp_datamodels.image import create_image

from karabo.data.external_data import (
    SingleFileDownloadObject,
    cscs_karabo_public_testing_base_url,
)
from karabo.imaging.imager_rascil import RascilDirtyImager, RascilDirtyImagerConfig
from karabo.simulation.beam import generate_gaussian_beam_data
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend


# DownloadObject instances used to download different golden files:
# - FITS file of a test continuous emission simulation including a gaussian beam of
# SKA-Mid using OSKAR.
# - FITS file of a test continuous emission simulation including a gaussian beam of
# SKA-Mid using RASCIL.
@pytest.fixture
def beam_gauss_O_fits_filename() -> str:
    return "test_beam_Gauss_OSKAR_v1.fits"


@pytest.fixture
def beam_gauss_R_fits_filename() -> str:
    return "test_beam_Gauss_RASCIL_v1.fits"


@pytest.fixture
def beam_gauss_O_fits_downloader(
    beam_gauss_O_fits_filename: str,
) -> SingleFileDownloadObject:
    return SingleFileDownloadObject(
        remote_file_path=beam_gauss_O_fits_filename,
        remote_base_url=cscs_karabo_public_testing_base_url,
    )


@pytest.fixture
def beam_gauss_R_fits_downloader(
    beam_gauss_R_fits_filename: str,
) -> SingleFileDownloadObject:
    return SingleFileDownloadObject(
        remote_file_path=beam_gauss_R_fits_filename,
        remote_base_url=cscs_karabo_public_testing_base_url,
    )


@pytest.mark.parametrize(
    "backend,telescope_name",
    [
        (SimulatorBackend.OSKAR, "SKA1MID"),
        (SimulatorBackend.RASCIL, "MID"),
    ],
)
def test_gaussian_beam(
    beam_gauss_O_fits_filename: str,
    beam_gauss_O_fits_downloader: SingleFileDownloadObject,
    beam_gauss_R_fits_filename: str,
    beam_gauss_R_fits_downloader: SingleFileDownloadObject,
    backend: SimulatorBackend,
    telescope_name: str,
) -> None:
    """
    We test that image reconstruction works with a Gaussian beam and
    test both visibility simulators: Oskar and Rascil.
    """
    # --------------------------
    # Download golden files for comparison
    if backend == SimulatorBackend.OSKAR:
        golden_beam_Gauss_path = beam_gauss_O_fits_downloader.get()
    else:
        golden_beam_Gauss_path = beam_gauss_R_fits_downloader.get()

    # Simulation parameters
    freq = 1.5e9
    freq_bin = 1e7
    npixels = 512
    cellsize = 3 / 180 * np.pi / npixels
    ra_deg = 20.0
    dec_deg = -30.0
    nchannels = 2

    # Beam for OSKAR
    beam_type = "Gaussian beam"
    fwhm_deg = 1.0

    # Beam for RASCIL
    primary_beam = create_image(
        npixel=npixels,
        cellsize=cellsize,
        phasecentre=SkyCoord(ra_deg, dec_deg, unit=(u.deg, u.deg), frame="icrs"),
        frequency=freq,
        channel_bandwidth=freq_bin,
        nchan=nchannels,
    )
    fwhm_pixels = fwhm_deg / np.degrees(cellsize)
    beam = generate_gaussian_beam_data(
        fwhm_pixels=fwhm_pixels,
        x_size=npixels,
        y_size=npixels,
    )

    for i in range(nchannels):
        primary_beam["pixels"][i][:] = beam

    # Load the test sky and the telescope
    sky = SkyModel.sky_test()
    telescope = Telescope.constructor(telescope_name, backend=backend)

    # Remove beam data if already present
    test = os.listdir(telescope.path)
    for item in test:
        if item.endswith(".bin"):
            os.remove(os.path.join(telescope.path, item))
    # ------------- Simulation Begins
    with tempfile.TemporaryDirectory() as tmpdir:
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=2e7,
            time_average_sec=8,
            noise_enable=False,
            ignore_w_components=True,
            use_gpus=False,
            station_type=beam_type,
            gauss_beam_fwhm_deg=fwhm_deg,
            gauss_ref_freq_hz=1.5e9,
        )
        observation = Observation(
            phase_centre_ra_deg=ra_deg,
            start_date_and_time=datetime(2000, 3, 20, 12, 6, 39, 0),
            length=timedelta(hours=1, minutes=5, seconds=0, milliseconds=0),
            phase_centre_dec_deg=dec_deg,
            number_of_time_steps=10,
            start_frequency_hz=freq,
            frequency_increment_hz=freq_bin,
            number_of_channels=nchannels,
        )
        visibility = simulation.run_simulation(
            telescope,
            sky,
            observation,
            backend=backend,
            primary_beam=primary_beam,
            visibility_format="MS",
            visibility_path=os.path.join(tmpdir, "beam_vis.ms"),
        )

        # RASCIL IMAGING
        dirty_imager = RascilDirtyImager(
            RascilDirtyImagerConfig(
                imaging_npixel=npixels,
                imaging_cellsize=cellsize,
                combine_across_frequencies=False,
            )
        )
        dirty = dirty_imager.create_dirty_image(visibility)

        outpath = Path(tmpdir)
        beam_fits_path = outpath / "test_beam.fits"
        dirty.write_to_file(str(beam_fits_path), overwrite=True)

        # Verify fits
        beam_fits_data, beam_fits_header = fits.getdata(
            beam_fits_path, ext=0, header=True
        )
        golden_beam_fits_data, golden_beam_fits_header = fits.getdata(
            golden_beam_Gauss_path, ext=0, header=True
        )

        # Check FITS data is close to goldenfile
        assert np.allclose(golden_beam_fits_data, beam_fits_data, equal_nan=True)

        # Check FITS header contain the same keys
        assert set(golden_beam_fits_header.keys()) == set(beam_fits_header.keys())
