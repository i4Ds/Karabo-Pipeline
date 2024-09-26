# type: ignore
"""
Source: https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels
Copyright: SKAO
License: Apache License 2.0

MeasurementSets V2 Reference Codes Based on Python-casacore
"""

import numpy

__version__ = "0.1"
__revision__ = "$Rev$"
__all__ = ["STOKES_CODES", "NUMERIC_STOKES"]

STOKES_CODES = {
    "I": 1,
    "Q": 2,
    "U": 3,
    "V": 4,
    "RR": 5,
    "RL": 6,
    "LR": 7,
    "LL": 8,
    "XX": 9,
    "XY": 10,
    "YX": 11,
    "YY": 12,
}

NUMERIC_STOKES = {
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
}


def geo_to_ecef(lat, lon, elev):
    """
    Convert latitude (rad), longitude (rad), elevation (m) to earth-
    centered, earth-fixed coordinates.
    """

    wgs84_a = 6378137.00000000
    wgs84_b = 6356752.31424518
    north = wgs84_a**2 / numpy.sqrt(
        wgs84_a**2 * numpy.cos(lat) ** 2 + wgs84_b**2 * numpy.sin(lat) ** 2
    )
    x_coord = (north + elev) * numpy.cos(lat) * numpy.cos(lon)
    y_coord = (north + elev) * numpy.cos(lat) * numpy.sin(lon)
    z_coord = ((wgs84_b**2 / wgs84_a**2) * north + elev) * numpy.sin(lat)

    return (x_coord, y_coord, z_coord)


def get_eci_transform(lat, lon):
    """
    Return a 3x3 transformation matrix that converts a baseline in
    [east, north, elevation] to earth-centered inertial coordinates
    for that baseline [x, y, z].
    """
    return numpy.array(
        [
            [
                -numpy.sin(lon),
                -numpy.sin(lat) * numpy.cos(lon),
                numpy.cos(lat) * numpy.cos(lon),
            ],
            [
                numpy.cos(lon),
                -numpy.sin(lat) * numpy.sin(lon),
                numpy.cos(lat) * numpy.sin(lon),
            ],
            [0.0, numpy.cos(lat), numpy.sin(lat)],
        ]
    )


def merge_baseline(ant1, ant2, shift=16):
    """
    Merge two stand ID numbers into a single baseline using the specified bit
    shift size.
    """
    return (ant1 << shift) | ant2


def split_baseline(baseline, shift=16):
    """
    Given a baseline, split it into it consistent stand ID numbers.
    """

    part = 2**shift - 1
    return (baseline >> shift) & part, baseline & part
