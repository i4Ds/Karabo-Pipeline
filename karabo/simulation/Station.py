from karabo.simulation.CoordinateHelpers import east_north_to_long_lat
from karabo.simulation.EastNorthCoordinate import EastNorthCoordinate
import numpy as np


class Station:

    def __init__(self, position: EastNorthCoordinate,
                 parent_longitude: float = 0,
                 parent_latitude: float = 0,
                 parent_altitude: float = 0):
        """
        :param position: Position of station in relation to the telescope centre
        """

        self.position: EastNorthCoordinate = position
        self.antennas: [EastNorthCoordinate] = []
        long, lat = east_north_to_long_lat(position.x, position.y, parent_longitude, parent_latitude)
        self.longitude: float = long
        self.latitude: float = lat
        self.altitude: float = parent_altitude

    def add_station_antenna(self, antenna: EastNorthCoordinate):
        self.antennas.append(antenna)
