import os
import tempfile

import numpy as np
import oskar.telescope as os_telescope

import karabo.error
from karabo.simulation.coordinate_helper import east_north_to_long_lat
from karabo.simulation.east_north_coordinate import EastNorthCoordinate
from karabo.simulation.station import Station


class Telescope:
    def __init__(self, longitude: float, latitude: float, altitude: float = 0):
        """
        WGS84 longitude and latitude and altitude in metres centre of the telescope.png centre
        """

        self.centre_longitude: float = longitude
        self.centre_latitude: float = latitude
        self.centre_altitude: float = altitude
        """
        Telescope Layout format
        Each row is one station
        [[  horizontal x (east), horizontal y (north), horizontal z (up) = 0,
            horizontal y (east) error = 0, horizontal y (north) error = 0, horizontal z (up) error ],
         [ ... ]]
        """
        self.stations: [Station] = []

    def add_station(self, horizontal_x: float, horizontal_y: float, horizontal_z: float = 0,
                    horizontal_x_coordinate_error: float = 0, horizontal_y_coordinate_error: float = 0,
                    horizontal_z_coordinate_error: float = 0):
        """
        Specify the stations as relative to the centre position
        :param horizontal_x: east coordinate relative to centre
        :param horizontal_y: north coordinate relative to centre
        :param horizontal_z: up coordinate
        :param horizontal_x_coordinate_error: east coordinate error
        :param horizontal_y_coordinate_error: north coordinate error
        :param horizontal_z_coordinate_error: up coordinate error
        """
        self.stations.append(Station(EastNorthCoordinate(horizontal_x,
                                                         horizontal_y,
                                                         horizontal_z,
                                                         horizontal_x_coordinate_error,
                                                         horizontal_y_coordinate_error,
                                                         horizontal_z_coordinate_error), self.centre_longitude,
                                     self.centre_latitude, self.centre_altitude))

    def add_antenna_to_station(self, station_index: int, horizontal_x: float, horizontal_y: float,
                               horizontal_z: float = 0,
                               horizontal_x_coordinate_error: float = 0, horizontal_y_coordinate_error: float = 0,
                               horizontal_z_coordinate_error: float = 0) -> None:
        """
        Add a new antenna to an existing station

        :param station_index: Index of station to add antenna to
        :param horizontal_x: east coordinate relative to the station center in metres
        :param horizontal_y: north coordinate relative to the station center in metres
        :param horizontal_z: altitude of antenna
        :param horizontal_x_coordinate_error: east coordinate error relative to the station center in metres
        :param horizontal_y_coordinate_error: north coordinate error relative to the station center in metres
        :param horizontal_z_coordinate_error: altitude of antenna error
        :return:
        """
        if station_index < len(self.stations):
            station = self.stations[station_index]
            station.add_station_antenna(
                EastNorthCoordinate(horizontal_x, horizontal_y, horizontal_z, horizontal_x_coordinate_error,
                                    horizontal_y_coordinate_error, horizontal_z_coordinate_error))

    def plot_telescope(self) -> None:
        """
        Plot the telescope and all its stations and antennas with longitude altitude
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        antenna_x = []
        antenna_y = []
        station_x = []
        station_y = []
        for station in self.stations:
            station_x.append(station.longitude)
            station_y.append(station.latitude)

            for antenna in station.antennas:
                long, lat = east_north_to_long_lat(antenna.x,
                                                   antenna.y,
                                                   station.longitude,
                                                   station.latitude)
                antenna_x.append(long)
                antenna_y.append(lat)

        ax.scatter(antenna_x, antenna_y, label="Antennas")
        ax.scatter(station_x, station_y, label="Stations")

        x = np.array([self.centre_longitude])
        y = np.array([self.centre_latitude])

        ax.scatter(x, y, label="Centre")
        ax.ticklabel_format(useOffset=False)
        ax.legend(loc='upper left', shadow=False, fontsize='medium')
        plt.show()

    def get_OSKAR_telescope(self) -> os_telescope:
        """
        Retrieve the OSKAR Telescope object from the karabo.Telescope object.
        :return: OSKAR Telescope object
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            self.__create_telescope_tm_file(temp_dir)
            tel = os_telescope.Telescope()
            tel.load(temp_dir)
            return tel

    def __create_telescope_tm_file(self, path: str) -> None:
        """
        Create .tm telescope configuration at the specified path
        :param path: directory in which the configuration will be saved in.
        """
        self.__write_position_txt(f"{path}/position.txt")
        self.__write_layout_txt(f"{path}/layout.txt", [station.position for station in self.stations])
        i = 0
        for station in self.stations:
            station_path = f"{path}/station{'{:03d}'.format(i)}"
            os.mkdir(station_path)
            self.__write_layout_txt(f"{station_path}/layout.txt", station.antennas)
            i += 1

    def __write_position_txt(self, position_file_path: str) -> None:
        position_file = open(position_file_path, "a")
        position_file.write(f"{self.centre_longitude} {self.centre_latitude} {self.centre_altitude} \n")
        position_file.close()

    def __write_layout_txt(self, layout_path: str, elements: [EastNorthCoordinate]):
        layout_file = open(layout_path, "a")
        for element in elements:
            layout_file.write(
                f"{element.x}, {element.y}, {element.z}, {element.x_error}, {element.y_error}, {element.z_error} \n")
        layout_file.close()

    # def get_SKA_LOW(self):
    #
    # def get_SKA_MID(self):

    # @staticmethod
    # def get_MEERKAT_Array():
    #     return Telescope(0, 0)


def get_OSKAR_Example_Telescope():
    path_elements = os.path.abspath(karabo.__file__).split('/')
    path_elements.pop()
    path = f"{'/'.join(path_elements)}/data/telescope.tm"
    return read_OSKAR_tm_file(path)


def read_OSKAR_tm_file(path: str) -> Telescope:
    files = []
    dirs = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.append(filenames)
        dirs.append(dirnames)

    if "position.txt" not in files[0]:
        raise karabo.error.KaraboException("Missing crucial position.txt file")

    if "layout.txt" not in files[0]:
        raise karabo.error.KaraboException(
            "Only Layout.txt is support. layout_ecef.txt and layout_wgs84.txt support is on its way.")

    telescope = None

    position_file = open(path + "/position.txt")
    lines = position_file.readlines()
    for line in lines:
        long_lat = line.split(" ")
        if len(long_lat) > 3:
            raise karabo.error.KaraboException("Too many values in position.txt")
        long = float(long_lat[0])
        lat = float(long_lat[1])
        alt = 0
        if len(long_lat) == 3:
            alt = float(long_lat[2])
        telescope = Telescope(long, lat, alt)

    if Telescope is None:
        raise karabo.error.KaraboException("Could not create Telescope from position.txt file.")

    position_file.close()

    station_positions = __read_layout_txt(f"{path}/layout.txt")
    for station_position in station_positions:
        telescope.add_station(station_position[0], station_position[1],
                              station_position[2], station_position[3],
                              station_position[4], station_position[5])

    station_dirs = list(filter(lambda directory: "station" in directory, dirs[0]))

    if len(station_dirs) != len(telescope.stations):
        raise karabo.error.KaraboException(f"Not all {len(telescope.stations)} stations have a station directory.")

    for station_dir, station in zip(station_dirs, telescope.stations):
        antenna_positions = __read_layout_txt(f"{path}/{station_dir}/layout.txt")
        for antenna_pos in antenna_positions:
            station.add_station_antenna(EastNorthCoordinate(antenna_pos[0],
                                                            antenna_pos[1],
                                                            antenna_pos[2],
                                                            antenna_pos[3],
                                                            antenna_pos[4],
                                                            antenna_pos[5]))

    return telescope


def __read_layout_txt(path) -> [[float]]:
    positions: [[float]] = []
    layout_file = open(path)
    lines = layout_file.readlines()
    for line in lines:
        station_position = line.split(",")
        values = np.zeros(6)
        i = 0
        for pos in station_position:
            values[i] = float(pos)
            i += 1
        positions.append([values[0], values[1], values[2], values[3], values[4], values[5]])
    layout_file.close()
    return positions
