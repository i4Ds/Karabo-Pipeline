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
        """
        :param position: Position of station in relation to the telescope.png centre
        """

        self.position: EastNorthCoordinate = position
        self.antennas: List[EastNorthCoordinate] = []
        long, lat = east_north_to_long_lat(
            position.x, position.y, parent_longitude, parent_latitude
        )
        self.longitude: float = long
        self.latitude: float = lat
        self.altitude: float = position.z

    def add_station_antenna(self, antenna: EastNorthCoordinate) -> None:
        self.antennas.append(antenna)
