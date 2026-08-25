import pytest

from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend

sdp_telescopes_to_test = [
    # (site_name, num_stations)
    # Configuration data is provided by ska_sdp_datamodels.
    ("LOWBD2", 512),
    ("MID", 197),
    ("ASKAP", 36),
    ("LOFAR", 134),
]


def test_set_telescope_name():
    site_name = "ASKAP"

    site = Telescope(0.0, 0.0, 0.0)
    assert site.name is None

    site.name = site_name
    assert site.name == site_name


@pytest.mark.parametrize("site_name, _", sdp_telescopes_to_test)
def test_set_telescope_from_sdp_configuration(site_name, _):
    site = Telescope.constructor(site_name, backend=SimulatorBackend.SDP)

    site.name = site_name
    assert site.name == site_name
    assert site.backend == SimulatorBackend.SDP


@pytest.mark.parametrize("site_name, num_stations", sdp_telescopes_to_test)
def test_num_of_stations(site_name, num_stations):
    site = Telescope.constructor(site_name, backend=SimulatorBackend.SDP)
    assert len(site.stations) == num_stations


@pytest.mark.parametrize("site_name, num_stations", sdp_telescopes_to_test)
def test_num_of_baselines(site_name, num_stations):
    site = Telescope.constructor(site_name, backend=SimulatorBackend.SDP)

    num_baselines = num_stations * (num_stations - 1) // 2
    stations = site.get_stations_wgs84()
    assert len(site.get_baseline_lengths(stations)) == num_baselines
