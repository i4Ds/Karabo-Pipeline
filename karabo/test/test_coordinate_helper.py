import math

import numpy as np
from numpy.typing import NDArray

from karabo.simulation.coordinate_helper import (  # east_north_to_long_lat,
    wgs84_to_cartesian,
)


# wgs for LOFAR
# Longitude 6.86763008, Latitude 52.91139459, height=50.11317741
# Earth location (metres)
# 3826923.9, 460915.1, 5064643.2
def test_wgs84_to_cartesian():
    cart_coord: NDArray[np.float64] = wgs84_to_cartesian(
        lon=6.86763008, lat=52.91139459, alt=50.11317741
    )

    print(cart_coord[0])
    assert math.isclose(cart_coord[0], 3826923.9, rel_tol=0.01)
    assert math.isclose(cart_coord[1], 460915.1, rel_tol=0.1)
    assert math.isclose(cart_coord[2], 5064643.2, rel_tol=0.1)
