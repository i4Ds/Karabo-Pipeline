from typing import get_args

import pytest

from karabo.simulation.telescope import RASCILTelescopes, Telescope


@pytest.mark.parametrize("site_name", get_args(RASCILTelescopes))
def test_set_telescope_name(site_name):
    # create dummy telescope, name will be overriden later
    site: Telescope = Telescope.constructor("EXAMPLE")
    site.name = site_name
    assert site.name == site_name
