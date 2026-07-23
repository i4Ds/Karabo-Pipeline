import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.image import create_image

from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.simulation.beam import generate_gaussian_beam_data
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend
from karabo.test.util import create_compatible_dirty_image


@pytest.mark.parametrize(
    "backend,telescope_name",
    [
        (SimulatorBackend.OSKAR, "SKA1MID"),
        (SimulatorBackend.SDP, "MID"),
    ],
)
def test_gaussian_beam(
    backend: SimulatorBackend,
    telescope_name: str,
) -> None:
    """
    We test that image reconstruction works with a Gaussian beam and
    test both visibility simulators: OSKAR and SDP.
    """
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

    # Custom beam image for SDP
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

        dirty = create_compatible_dirty_image(
            visibility,
            DirtyImagerConfig(
                imaging_npixel=npixels,
                imaging_cellsize=cellsize,
                combine_across_frequencies=False,
            ),
        )

        assert dirty.data.shape == (nchannels, 1, npixels, npixels)
        assert np.isfinite(dirty.data).all()
        assert np.all(np.std(dirty.data, axis=(-2, -1)) > 0.0)
        assert np.nanmax(np.abs(dirty.data)) > 0.0
        assert dirty.header["CTYPE1"].startswith("RA")
        assert dirty.header["CTYPE2"].startswith("DEC")
        assert dirty.header["CTYPE3"] == "STOKES"
        assert dirty.header["CTYPE4"] == "FREQ"
        assert np.isclose(dirty.header["CRVAL1"], ra_deg)
        assert np.isclose(dirty.header["CRVAL2"], dec_deg)
        first_channel_frequency = (
            freq + freq_bin / 2 if backend is SimulatorBackend.SDP else freq
        )
        assert np.isclose(dirty.header["CRVAL4"], first_channel_frequency)
        assert np.isclose(abs(dirty.header["CDELT4"]), freq_bin)

        centre = np.array([npixels // 2, npixels // 2])
        for channel in dirty.data[:, 0]:
            peak = np.array(np.unravel_index(np.nanargmax(channel), channel.shape))
            assert np.max(np.abs(peak - centre)) <= 8
