import math

import numpy as np
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

    assert math.isclose(cart_coord[0], 3826923.9, rel_tol=0.01)
    assert math.isclose(cart_coord[1], 460915.1, rel_tol=0.01)
    assert math.isclose(cart_coord[2], 5064643.2, rel_tol=0.01)


#
# east,north = 1000, 1000   --> lat,lon = 52.920378,6.876678
# east,north = -1000, -1000 --> lat,lon = 52.902411,6.858582


def test_east_north_to_long_lat():
    # coords of LOFAR in NL
    lon = 6.86763008
    lat = 52.91139459
    # go to new position 1000 m east and north.
    east = 1000
    north = 1000

    new_lon, new_lat = east_north_to_long_lat(
        east_relative=east, north_relative=north, long=lon, lat=lat
    )

    print(f"{new_lon=}  {new_lat=}")
    assert math.isclose(new_lon - 6.876678, 0.0, abs_tol=1e-4)
    assert math.isclose(new_lat - 52.920378, 0.0, abs_tol=1e-4)
