from typing import get_args

import pytest

from karabo.simulation.telescope import RASCILTelescopes, Telescope


@pytest.mark.parametrize("site_name", get_args(RASCILTelescopes))
def test_set_telescope_name_from_oskar_telescope(site_name):
    # create dummy telescope, name will be overriden later
    site: Telescope = Telescope.constructor("EXAMPLE")

    site.name = site_name
    assert site.name == site_name


def test_set_telescope_name():
    site_name = "ASKAP"

    # create dummy telescope, name will be overriden later
    site: Telescope = Telescope(0.0, 0.0, 0.0)
    # __init__() sets name to None
    assert site.name is None

    site.name = site_name
    assert site.name == site_name
