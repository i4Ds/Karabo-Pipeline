from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray


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


def wgs84_to_cartesian(
    lon: Union[float, NDArray[np.float64]],
    lat: Union[float, NDArray[np.float64]],
    alt: Union[float, NDArray[np.float64]],
    radius: int = 6371000,
) -> NDArray[np.float64]:
    """Transforms WGS84 to cartesian in meters.

    Args:
        lon: Longitude [deg].
        lat: Latitude [deg].
        alt: Altitude [m].
        radius: Radius of earth in m.

    Returns:
        Cartesian x,y,z coordinates (nx3) in meters.
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = (radius + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (radius + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (radius + alt) * np.sin(lat_rad)
    return np.array([x, y, z]).T
