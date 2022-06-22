from random import random
from math import cos, sin, floor, sqrt, pi, ceil

import numpy as np


def euclidean_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)


def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=random):
    tau = 2 * pi
    cellsize = r / sqrt(2)

    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    def grid_coords(p):
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(p, gx, gy):
        yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in yrange:
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if distance(p, g) <= r:
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
    return [p for p in grid if p is not None]


def get_poisson_disk_sky(min_size: (float, float),
                         max_size: (float, float),
                         flux_min: float,
                         flux_max: float,
                         r=10):
    x, y = min_size
    X, Y = max_size
    width = abs(X - x)
    height = abs(Y - y)
    center_x = x + (X - x) * 0.5
    center_y = y + (Y - y) * 0.5
    samples = poisson_disc_samples(width, height, r)
    np_samples = np.array(samples)
    ra = np_samples[:, 0] - (width * 0.5)
    dec = np_samples[:, 1] - (height * 0.5)
    ra = ra + center_x
    dec = dec + center_y
    np_samples = np.vstack((ra, dec)).transpose()
    flux = np.random.random((len(samples), 1)) * (flux_max + 1 - flux_min) + flux_min
    sky_array = np.hstack((np_samples, flux))
    return sky_array
