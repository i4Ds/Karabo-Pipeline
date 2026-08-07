from typing import Tuple, Union

import numpy as np
from astropy.coordinates import EarthLocation
from numpy.typing import NDArray

import astropy.units as u


def east_north_to_long_lat(
    east_relative: float,
    north_relative: float,
    long: float,
    lat: float,
    alt: float = 0.0,
    up: float = 0.0,
) -> Tuple[float, float]:
    """
    Calculate the longitude and latitude of an east-north coordinate
    based on some reference location, using a rigorous ENU → ECEF → geodetic
    conversion on the WGS84 ellipsoid.

    :param east_relative: east coordinate in meters
    :param north_relative: north coordinate in meters
    :param long: reference location longitude in degrees
    :param lat: reference location latitude in degrees
    :param alt: reference location altitude in meters (default 0.0)
    :param up: up (vertical) offset in meters (default 0.0)
    :return: Tuple of calculated (longitude, latitude) in degrees
    """
    # 1. Reference point in ECEF
    ref = EarthLocation.from_geodetic(
        lon=long * u.deg, lat=lat * u.deg, height=alt * u.m
    )
    x0 = ref.x.to_value(u.m)
    y0 = ref.y.to_value(u.m)
    z0 = ref.z.to_value(u.m)

    # 2. ENU → ECEF rotation (must be done manually)
    lon_rad = np.deg2rad(long)
    lat_rad = np.deg2rad(lat)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    dx = (
        -sin_lon * east_relative
        - sin_lat * cos_lon * north_relative
        + cos_lat * cos_lon * up
    )
    dy = (
        cos_lon * east_relative
        - sin_lat * sin_lon * north_relative
        + cos_lat * sin_lon * up
    )
    dz = cos_lat * north_relative + sin_lat * up

    # 3. ECEF → geodetic
    target = EarthLocation.from_geocentric(
        x=(x0 + dx) * u.m, y=(y0 + dy) * u.m, z=(z0 + dz) * u.m
    )
    new_lon = target.geodetic.lon.deg
    new_lat = target.geodetic.lat.deg
    return float(new_lon), float(new_lat)


def wgs84_to_cartesian(
    lon: Union[float, NDArray[np.float64]],
    lat: Union[float, NDArray[np.float64]],
    alt: Union[float, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Transforms WGS84 to cartesian in meters.

    Args:
        lon: Longitude [deg].
        lat: Latitude [deg].
        alt: Altitude [m].

    Returns:
        Cartesian x,y,z coordinates (nx3) in meters.
    """
    loc = EarthLocation.from_geodetic(
        lon=np.asarray(lon) * u.deg,
        lat=np.asarray(lat) * u.deg,
        height=np.asarray(alt) * u.m,
    )
    x = loc.x.to_value(u.m)
    y = loc.y.to_value(u.m)
    z = loc.z.to_value(u.m)
    return np.array([x, y, z]).T
