import numpy as np


def east_north_to_long_lat(east_relative: float, north_relative: float, long: float, lat: float) -> (float, float):
    # https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters
    r_earth = 6371000
    new_latitude = lat + (east_relative / r_earth) * (180 / np.pi)
    new_longitude = long + (north_relative / r_earth) * (180 / np.pi) / np.cos(
        long * np.pi / 180)
    return new_longitude, new_latitude
