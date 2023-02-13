from typing import Tuple

import numpy as np


def east_north_to_long_lat(
    east_relative: float, north_relative: float, long: float, lat: float
) -> Tuple[float, float]:
    """
    Calculate the longitude and latitude of an east-north coordinate
    based on some reference location.

    :param east_relative: east coordinate in meters
    :param north_relative: north coordinate in meters
    :param long: reference location longitude
    :param lat: reference location latitude
    :return: Tuple of calculated longitude and latitude of passed east-north coordinate
    """

    # https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters
    r_earth = 6371000
    new_latitude = lat + (east_relative / r_earth) * (180 / np.pi)
    new_longitude = long + (north_relative / r_earth) * (180 / np.pi) / np.cos(
        long * np.pi / 180
    )
    return new_longitude, new_latitude
