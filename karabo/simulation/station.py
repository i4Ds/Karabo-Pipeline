from typing import Dict, List, Mapping, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from karabo.simulation.coordinate_helper import east_north_to_long_lat
from karabo.simulation.east_north_coordinate import EastNorthCoordinate

SEFDCurves = Dict[str, NDArray[np.float64]]


class Station:
    def __init__(
        self,
        position: EastNorthCoordinate,
        parent_longitude: float = 0.0,
        parent_latitude: float = 0.0,
        parent_altitude: float = 0.0,
        *,
        name: Optional[str] = None,
        dish_diameter_m: Optional[float] = None,
        sefd: Optional[Mapping[str, ArrayLike]] = None,
    ):
        """
        :param position: Position of station in relation to the telescope.png centre
        :param name: Station identifier, e.g. `M000` or `SKA001`.
        :param dish_diameter_m: Dish diameter, or aperture-array station diameter,
            in metres.
        :param sefd: SEFD curves per receiver band, see `Station.sefd`.
        """

        self.position: EastNorthCoordinate = position
        self.antennas: List[EastNorthCoordinate] = []
        long, lat = east_north_to_long_lat(
            position.x,
            position.y,
            parent_longitude,
            parent_latitude,
            alt=parent_altitude,
            up=position.z,
        )
        self.longitude: float = long
        self.latitude: float = lat
        self.altitude: float = parent_altitude + position.z
        self.name: Optional[str] = name
        self.dish_diameter_m = dish_diameter_m
        self._sefd = validated_sefd(sefd)

    @property
    def dish_diameter_m(self) -> Optional[float]:
        return self._dish_diameter_m

    @dish_diameter_m.setter
    def dish_diameter_m(self, value: Optional[float]) -> None:
        if value is not None and not (np.isfinite(value) and value > 0):
            raise ValueError(f"dish_diameter_m must be finite and > 0, got {value}")
        self._dish_diameter_m = None if value is None else float(value)

    @property
    def sefd(self) -> Optional[SEFDCurves]:
        """SEFD curves per receiver band, as `{band: array}`.

        Each array has shape (N, 2) with columns (frequency_hz, sefd_jy) and
        strictly increasing frequencies. The SEFD is per station and per
        polarisation. Curves of different bands must not be interpolated
        across each other. The arrays are read-only; assign new curves instead.
        """
        return self._sefd

    @sefd.setter
    def sefd(self, value: Optional[SEFDCurves]) -> None:
        self._sefd = validated_sefd(value)

    def add_station_antenna(self, antenna: EastNorthCoordinate) -> None:
        self.antennas.append(antenna)


def validated_sefd(value: Optional[Mapping[str, ArrayLike]]) -> Optional[SEFDCurves]:
    """Validates SEFD curves and returns them as read-only arrays per band."""
    if value is None:
        return None
    curves: SEFDCurves = {}
    for band, samples in value.items():
        curve = np.array(samples, dtype=np.float64)
        if curve.ndim != 2 or curve.shape[0] == 0 or curve.shape[1] != 2:
            raise ValueError(
                f"SEFD curve of band {band!r} must have shape (N, 2), "
                f"got {curve.shape}"
            )
        if not np.all(np.isfinite(curve)) or np.any(curve <= 0):
            raise ValueError(f"SEFD curve of band {band!r} must be finite and > 0")
        if np.any(np.diff(curve[:, 0]) <= 0):
            raise ValueError(
                f"SEFD curve of band {band!r} must have strictly "
                "increasing frequencies"
            )
        curve.setflags(write=False)
        curves[str(band)] = curve
    return curves
