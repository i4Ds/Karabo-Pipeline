import math
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from karabo.imaging.imager_rascil import RascilDirtyImager, RascilDirtyImagerConfig
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend
from karabo.util.data_util import get_module_absolute_path
from karabo.util.file_handler import DirPathType


@pytest.fixture
def oskar_telescope() -> Telescope:
    return Telescope.constructor("MeerKAT", backend=SimulatorBackend.OSKAR)


@pytest.fixture
def rascil_telescope() -> Telescope:
    return Telescope.constructor("LOFAR", backend=SimulatorBackend.RASCIL)


@pytest.fixture
def sky_model() -> SkyModel:
    sky = SkyModel()
    sky_data = np.array(
        [
            [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
            [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
            [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10],
        ]
    )
    sky.add_point_sources(sky_data)
    return sky


# This test only tests that the function runs without error. This includes
#   - counting number of stations in telescope after cut
#   - assert that output measurement set was created
#   - assert that FITS image calculated with cut baseline was written
# However, it doesn't test if the result is correct. You would expect a
# different image quality. This is because a cut baseline reduces the
# resolution of the instrument.
def test_baselines_based_cutoff(oskar_telescope: Telescope, sky_data: SkyModel):
    #  Max. baselength of MeerKAT is 7500 m. Thus, we cut somewhere
    # inbetween,
    lcut = 5000
    hcut = 10000  # Lower cut off and higher cut-off in meters
    with tempfile.TemporaryDirectory() as tmpdir:
        tm_path = os.path.join(tmpdir, "tel-cut.tm")
        baseline_cut: Tuple[
            DirPathType, Dict[str, str]
        ] = Telescope.create_baseline_cut_telescope(
            lcut,
            hcut,
            oskar_telescope,
            tm_path=tm_path,
        )

        telescope_path, _ = baseline_cut
        telescope = Telescope.read_OSKAR_tm_file(telescope_path)
        # There are 64 stations fpr MeerKAT. After baseline cut there are
        # 11 left.
        assert len(telescope.get_stations_wgs84()) == 11

        sky = SkyModel()
        sky.add_point_sources(sky_data)
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=1e6,
            time_average_sec=1,
        )

        observation = Observation(
            phase_centre_ra_deg=20.0,
            phase_centre_dec_deg=-30.5,
            start_date_and_time=datetime(2022, 1, 1, 11, 00, 00, 521489),
            length=timedelta(hours=0, minutes=0, seconds=1, milliseconds=0),
            number_of_time_steps=1,
            start_frequency_hz=100e6,
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
        assert os.path.isdir(visibility.path)

        dirty.write_to_file(os.path.join(tmpdir, "baseline_cut.fits"), overwrite=True)

        assert os.path.isfile(os.path.join(tmpdir, "baseline_cut.fits"))

        dirty.plot(
            title="Flux Density (Jy)",
            filename=os.path.join(
                get_module_absolute_path(), "test/data/image_cut.png"
            ),
        )


def test_telescope_max_baseline_length(
    oskar_telescope: Telescope, rascil_telescope: Telescope
):
    max_length_oskar: np.float64 = oskar_telescope.max_baseline()
    # Should be the same +/- 1 m
    assert math.isclose(max_length_oskar - 7500.0, 0, abs_tol=1)

    max_length_rascil: np.float64 = rascil_telescope.max_baseline()
    # Should be the same +/- 1 m
    assert math.isclose(max_length_rascil - 995242.0, 0, abs_tol=1)

    freq_Hz = 100e6
    angular_res: float = Telescope.ang_res(freq_Hz, max_length_oskar)
    assert math.isclose(angular_res, 1.44, rel_tol=1e-2)


def test_telescope_stations(oskar_telescope: Telescope, rascil_telescope: Telescope):
    # station has 30 stations according to *.tm file
    baseline_wgs: NDArray[np.float64] = oskar_telescope.get_stations_wgs84()
    assert len(baseline_wgs) == 64

    baseline_wgs = rascil_telescope.get_stations_wgs84()
    assert len(baseline_wgs) == 134


def test_telescope_baseline_length(rascil_telescope):
    stations_wgs: NDArray[np.float64] = rascil_telescope.get_stations_wgs84()
    num_stations = len(stations_wgs)
    baseline_length: NDArray[np.float64] = Telescope.get_baseline_lengths(stations_wgs)
    assert len(baseline_length) == num_stations * (num_stations - 1) / 2
