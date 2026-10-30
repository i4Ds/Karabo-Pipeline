from typing import List

from karabo.simulation.coordinate_helper import east_north_to_long_lat
from karabo.simulation.east_north_coordinate import EastNorthCoordinate


class Station:
    def __init__(
        self,
        position: EastNorthCoordinate,
        parent_longitude: float = 0.0,
        parent_latitude: float = 0.0,
        parent_altitude: float = 0.0,
    ):
        """Initialize a station.

        Args:
            position (EastNorthCoordinate): Position of station in relation to the
                telescope centre.
            parent_longitude (float, optional): Longitude of the telescope centre
                in degrees. Defaults to 0.0.
            parent_latitude (float, optional): Latitude of the telescope centre in
                degrees. Defaults to 0.0.
            parent_altitude (float, optional): Altitude of the telescope centre in
                meters. Defaults to 0.0.
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

    def add_station_antenna(self, antenna: EastNorthCoordinate) -> None:
        self.antennas.append(antenna)
