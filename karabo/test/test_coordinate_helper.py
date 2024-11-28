import math

import numpy as np
import pytest
from numpy.typing import NDArray

from karabo.simulation.coordinate_helper import (
    east_north_to_long_lat,
    wgs84_to_cartesian,
)


# wgs for LOFAR
# longitude 6.86763008, latitude 52.91139459, height 50.11317741
# expected geocentric (a.k.a. EarthLocation in astropy) (metres)
# 3826923.9, 460915.1, 5064643.2
def test_wgs84_to_cartesian():
    cart_coord: NDArray[np.float64] = wgs84_to_cartesian(
        lon=6.86763008, lat=52.91139459, alt=50.11317741
    )

    geocentric_coords_expected = [3826923.9, 460915.1, 5064643.2]
    assert np.isclose(cart_coord, geocentric_coords_expected, rtol=0.01).all()


#
# east,north = 1000, 1000   --> lat,lon = 52.920378,6.876678
# east,north = -1000, -1000 --> lat,lon = 52.902411,6.858582

testdata = [
    (1000, 1000, 52.920378, 6.876678),  # go east and north 1000m
    (-1000, -1000, 52.902411, 6.858582),  # go west and south 1000m
]


@pytest.mark.parametrize("east, north, test_lat, test_lon", testdata)
def test_east_north_to_long_lat(east, north, test_lat, test_lon):
    # coords of LOFAR in NL
    lon = 6.86763008
    lat = 52.91139459

    new_lon, new_lat = east_north_to_long_lat(
        east_relative=east, north_relative=north, long=lon, lat=lat
    )

    assert math.isclose(new_lon - test_lon, 0.0, abs_tol=1e-4)
    assert math.isclose(new_lat - test_lat, 0.0, abs_tol=1e-4)
