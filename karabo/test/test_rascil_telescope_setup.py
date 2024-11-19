import pytest

from karabo.simulation.telescope import SimulatorBackend, Telescope

rascil_telesecopes_to_test = [
    # (site_name, num_stations)
    # data files are read from
    # /envs/karabo/lib/python3.9/site-packages/ska_sdp_datamodels/configuration/ \
    # example_antenna_files/
    ("LOWBD2", 512),
    ("MID", 197),
    ("ASKAP", 36),
    ("LOFAR", 134),
]


def test_set_telescope_name():
    site_name = "ASKAP"

    # create dummy telescope, name will be overriden later
    site: Telescope = Telescope(0.0, 0.0, 0.0)
    # __init__() sets name to None
    assert site.name is None

    site.name = site_name
    assert site.name == site_name


@pytest.mark.parametrize("site_name, _", rascil_telesecopes_to_test)
def test_set_telescope_from_oskar_telescope(site_name, _):
    # we must set the backend to RASCIL. Otherwise OSKAR is used by default
    site: Telescope = Telescope.constructor(site_name, backend=SimulatorBackend.RASCIL)

    site.name = site_name
    assert site.name == site_name
    assert site.backend == SimulatorBackend.RASCIL


@pytest.mark.parametrize("site_name, num_stations", rascil_telesecopes_to_test)
def test_num_of_stations(site_name, num_stations):
    site: Telescope = Telescope.constructor(site_name, backend=SimulatorBackend.RASCIL)
    assert len(site.stations) == num_stations


@pytest.mark.parametrize("site_name, num_stations", rascil_telesecopes_to_test)
def test_num_of_baselines(site_name, num_stations):
    site: Telescope = Telescope.constructor(site_name, backend=SimulatorBackend.RASCIL)

    # This is the predicted number of baselines
    num_baselines = num_stations * (num_stations - 1) // 2
    stations = site.get_stations_wgs84()
    assert len(site.get_baseline_lengths(stations)) == num_baselines
