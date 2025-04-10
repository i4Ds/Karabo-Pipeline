import math
from math import ceil, cos, floor, pi, sin, sqrt
from typing import List, Literal, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from karabo.util._types import FloatLike, NPFloatLike


def poisson_disc_samples(
    width: FloatLike,
    height: FloatLike,
    r: int,
    k: int = 5,
    ord: Union[None, float, Literal["fro", "nuc"]] = None,
) -> List[Tuple[float, float]]:
    tau = 2 * pi
    cellsize = r / sqrt(2)
    random = np.random.rand
    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [(np.inf, np.inf)] * (grid_width * grid_height)

    def grid_coords(p: Tuple[float, float]) -> Tuple[int, int]:
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(
        p: Tuple[float, float],
        gx: int,
        gy: int,
    ) -> bool:
        yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in yrange:
                g = grid[x + y * grid_width]
                if g == (np.inf, np.inf):
                    continue
                if np.linalg.norm(np.array(p) - np.array(g), ord=ord) <= r:
                    return False
        return True

    p = width * random(), height * random()
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = int(random() * len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            alpha = tau * random()
            d = r * sqrt(3 * random() + 1)
            px = qx + d * cos(alpha)
            py = qy + d * sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = (px, py)
            grid_x, grid_y = grid_coords(p)
            if not fits(p, grid_x, grid_y):
                continue
            queue.append(p)
            grid[grid_x + grid_y * grid_width] = p
    grid = [p for p in grid if p != (np.inf, np.inf)]
    return grid


def get_poisson_disk_sky(
    min_size: Tuple[FloatLike, FloatLike],
    max_size: Tuple[FloatLike, FloatLike],
    flux_min: FloatLike,
    flux_max: FloatLike,
    r: int = 10,
) -> NDArray[np.float_]:
    assert flux_max >= flux_min
    x = min_size[0]
    y = min_size[1]
    X = max_size[0]
    Y = max_size[1]
    width = cast(FloatLike, abs(X - x))
    height = cast(FloatLike, abs(Y - y))
    center_x = x + (X - x) * 0.5
    center_y = y + (Y - y) * 0.5
    samples = poisson_disc_samples(width, height, r)
    np_samples = np.array(samples)
    ra = np_samples[:, 0] - (width * 0.5)
    dec = np_samples[:, 1] - (height * 0.5)
    ra = ra + center_x
    dec = dec + center_y
    np_samples = np.vstack((ra, dec)).transpose()
    flux = np.random.random((len(samples), 1)) * (flux_max - flux_min) + flux_min
    sky_array = np.hstack((np_samples, flux))
    return sky_array


#
def long_lat_to_cartesian(lat: NPFloatLike, lon: NPFloatLike) -> NDArray[np.float_]:
    lat_, lon_ = np.deg2rad(lat), np.deg2rad(lon)
    x = R * cos(lat_) * cos(lon_)
    y = R * cos(lat_) * sin(lon_)
    z = R * sin(lat_)
    out = cast(
        NDArray[np.float_], np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
    )
    return out


#
#
# def cartesian_to_long_lat(cart: [float, float, float]):
#     lat = np.degrees(np.arcsin(cart[2]))
#     lon = np.degrees(np.arctan2(cart[1], cart[0]))
#     return np.array([lon, lat])


R = 6360000  # earth radius


def cartesian_to_ll(
    x: FloatLike,
    y: FloatLike,
    z: int = 0,
) -> Tuple[float, float]:
    # does not use `z`
    r = math.sqrt(x**2 + y**2)
    long = 180 * math.atan2(y, x) / math.pi
    lat = 180 * math.acos(r / R) / math.pi
    return lat, long
