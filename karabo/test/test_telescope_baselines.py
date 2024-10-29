import math
import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from karabo.imaging.imager_rascil import RascilDirtyImager, RascilDirtyImagerConfig
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend


def test_baselines_based_cutoff(sky_data: NDArray[np.float64]):
    lcut = 5000
    hcut = 10000  # Lower cut off and higher cut-off in meters
    tel = Telescope.constructor("MeerKAT")
    with tempfile.TemporaryDirectory() as tmpdir:
        tm_path = os.path.join(tmpdir, "tel-cut.tm")
        telescope_path, _ = Telescope.create_baseline_cut_telescope(
            lcut,
            hcut,
            tel,
            tm_path=tm_path,
        )
        telescope = Telescope.read_OSKAR_tm_file(telescope_path)
        sky = SkyModel()
        sky.add_point_sources(sky_data)
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=1e6,
            time_average_sec=1,
            noise_enable=False,
            noise_seed="time",
            noise_freq="Range",
            noise_rms="Range",
            noise_start_freq=1.0e9,
            noise_inc_freq=1.0e8,
            noise_number_freq=24,
            noise_rms_start=5000,
            noise_rms_end=10000,
        )
        observation = Observation(
            phase_centre_ra_deg=20.0,
            start_date_and_time=datetime(2022, 1, 1, 11, 00, 00, 521489),
            length=timedelta(hours=0, minutes=0, seconds=1, milliseconds=0),
            phase_centre_dec_deg=-30.5,
            number_of_time_steps=1,
            start_frequency_hz=1.0e9,
            frequency_increment_hz=1e6,
            number_of_channels=1,
        )

        visibility = simulation.run_simulation(
            telescope,
            sky,
            observation,
            visibility_format="MS",
            visibility_path=os.path.join(tmpdir, "out.ms"),
        )

        dirty_imager = RascilDirtyImager(
            RascilDirtyImagerConfig(
                imaging_npixel=4096,
                imaging_cellsize=50,
                combine_across_frequencies=False,
            )
        )
        dirty = dirty_imager.create_dirty_image(visibility)
        dirty.write_to_file(os.path.join(tmpdir, "baseline_cut.fits"), overwrite=True)
        dirty.plot(title="Flux Density (Jy)")


def test_oskar_telescope_baseline():
    site_tel = Telescope.constructor("LOFAR", backend=SimulatorBackend.OSKAR)
    baseline_wgs = site_tel.get_baselines_wgs84()
    assert len(baseline_wgs) == 134

    max_baseline_length = site_tel.max_baseline()
    assert math.isclose(max_baseline_length, 998420.050)

    freq_Hz = 100e6
    angular_res = site_tel.ang_res(freq_Hz, max_baseline_length)
    assert math.isclose(angular_res, 0.01081, rel_tol=1e-4)
