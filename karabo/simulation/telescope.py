import logging
import os
import re
import shutil
from typing import List

import numpy as np
import oskar.telescope as os_telescope
from math import comb
import karabo.error
from karabo.karabo_resource import KaraboResource
from karabo.simulation.coordinate_helper import east_north_to_long_lat
from karabo.simulation.east_north_coordinate import EastNorthCoordinate
from karabo.simulation.station import Station
from karabo.simulation.telescope_versions import (
    ALMAVersions,
    ATCAVersions,
    CARMAVersions,
    NGVLAVersions,
    PDBIVersions,
    SMAVersions,
    VLAVersions,
    ACAVersions,
)
from karabo.util.FileHandle import FileHandle
from karabo.util.data_util import get_module_absolute_path
from karabo.util.math_util import long_lat_to_cartesian
from astropy import units
from astropy.stats import gaussian_fwhm_to_sigma


class Telescope(KaraboResource):
    def __init__(self, longitude: float, latitude: float, altitude: float = 0):
        """
        WGS84 longitude and latitude and altitude in metres centre of the telescope.png centre
        """
        self.temp_dir = None
        self.path = None  # hotfix #59
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
        self.stations: List[Station] = []

    def add_station(
        self,
        horizontal_x: float,
        horizontal_y: float,
        horizontal_z: float = 0,
        horizontal_x_coordinate_error: float = 0,
        horizontal_y_coordinate_error: float = 0,
        horizontal_z_coordinate_error: float = 0,
    ):
        """
        Specify the stations as relative to the centre position
        :param horizontal_x: east coordinate relative to centre
        :param horizontal_y: north coordinate relative to centre
        :param horizontal_z: up coordinate
        :param horizontal_x_coordinate_error: east coordinate error
        :param horizontal_y_coordinate_error: north coordinate error
        :param horizontal_z_coordinate_error: up coordinate error
        """
        self.stations.append(
            Station(
                EastNorthCoordinate(
                    horizontal_x,
                    horizontal_y,
                    horizontal_z,
                    horizontal_x_coordinate_error,
                    horizontal_y_coordinate_error,
                    horizontal_z_coordinate_error,
                ),
                self.centre_longitude,
                self.centre_latitude,
                self.centre_altitude,
            )
        )

    def add_antenna_to_station(
        self,
        station_index: int,
        horizontal_x: float,
        horizontal_y: float,
        horizontal_z: float = 0,
        horizontal_x_coordinate_error: float = 0,
        horizontal_y_coordinate_error: float = 0,
        horizontal_z_coordinate_error: float = 0,
    ) -> None:
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
                EastNorthCoordinate(
                    horizontal_x,
                    horizontal_y,
                    horizontal_z,
                    horizontal_x_coordinate_error,
                    horizontal_y_coordinate_error,
                    horizontal_z_coordinate_error,
                )
            )

    def plot_telescope(self, file: str = None) -> None:
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
                long, lat = east_north_to_long_lat(
                    antenna.x, antenna.y, station.longitude, station.latitude
                )
                antenna_x.append(long)
                antenna_y.append(lat)

        ax.scatter(antenna_x, antenna_y, label="Antennas")
        ax.scatter(station_x, station_y, label="Stations")

        x = np.array([self.centre_longitude])
        y = np.array([self.centre_latitude])

        ax.scatter(x, y, label="Centre")
        ax.ticklabel_format(useOffset=False)
        ax.legend(loc="upper left", shadow=False, fontsize="medium")

        if file is not None:
            plt.savefig(file)
            plt.close(fig)
        else:
            plt.show()

    def get_OSKAR_telescope(self) -> os_telescope:
        """
        Retrieve the OSKAR Telescope object from the karabo.Telescope object.
        :return: OSKAR Telescope object
        """
        self.temp_dir = FileHandle(is_dir=True)
        self.__create_telescope_tm_file(self.temp_dir.path)
        tel = os_telescope.Telescope()
        tel.load(self.temp_dir.path)
        self.path = self.temp_dir.path
        return tel

    def __create_telescope_tm_file(self, path: str) -> None:
        """
        Create .tm telescope configuration at the specified path
        :param path: directory in which the configuration will be saved in.
        """
        self.__write_position_txt(f"{path}/position.txt")
        self.__write_layout_txt(
            f"{path}/layout.txt", [station.position for station in self.stations]
        )
        i = 0
        for station in self.stations:
            station_path = f"{path}/station{'{:03d}'.format(i)}"
            os.mkdir(station_path)
            self.__write_layout_txt(f"{station_path}/layout.txt", station.antennas)
            i += 1

    def __write_position_txt(self, position_file_path: str) -> None:
        position_file = open(position_file_path, "a")
        position_file.write(
            f"{self.centre_longitude} {self.centre_latitude} {self.centre_altitude} \n"
        )
        position_file.close()

    def __write_layout_txt(self, layout_path: str, elements: List[EastNorthCoordinate]):
        layout_file = open(layout_path, "a")
        for element in elements:
            layout_file.write(
                f"{element.x}, {element.y}, {element.z}, {element.x_error}, {element.y_error}, {element.z_error} \n"
            )
        layout_file.close()

    def write_to_file(self, path: str) -> None:
        shutil.copytree(self.path.path, path)

    def get_cartesian_position(self):
        return long_lat_to_cartesian(self.centre_latitude, self.centre_longitude)

    @classmethod
    def read_from_file(cls, path: str) -> any:
        if path.endswith(".tm"):
            logging.info("Supplied file is a .tm file. Read as OSKAR Telescope file.")
            cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_MEERKAT_Telescope(cls):
        path = f"{get_module_absolute_path()}/data/meerkat.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_ACA_Telescope(cls, version: ACAVersions):
        path = f"{get_module_absolute_path()}/data/aca.{version.value}.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_ALMA_Telescope(cls, version: ALMAVersions):
        path = f"{get_module_absolute_path()}/data/alma.{version.value}.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_ASKAP_Telescope(cls):
        path = f"{get_module_absolute_path()}/data/askap.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_ATCA_Telescope(cls, version: ATCAVersions):
        path = f"{get_module_absolute_path()}/data/atca.{version.value}.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_CARMA_Telescope(cls, version: CARMAVersions):
        path = f"{get_module_absolute_path()}/data/carma.{version.value}.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_LOFAR_Telescope(cls):
        path = f"{get_module_absolute_path()}/data/lofar.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_MKATPLUS_Telescope(cls):
        path = f"{get_module_absolute_path()}/data/mkatplus.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_NG_VLA_Telescope(cls, version: NGVLAVersions):
        path = f"{get_module_absolute_path()}/data/ngvla-{version.value}.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_PDBI_Telescope(cls, version: PDBIVersions):
        path = f"{get_module_absolute_path()}/data/pdbi-{version.value}.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_SKA1_LOW_Telescope(cls):
        path = f"{get_module_absolute_path()}/data/ska1low.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_SKA1_MID_Telescope(cls):
        path = f"{get_module_absolute_path()}/data/ska1mid.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_SMA_Telescope(cls, version: SMAVersions):
        path = f"{get_module_absolute_path()}/data/sma.{version.value}.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_VLA_Telescope(cls, version: VLAVersions):
        path = f"{get_module_absolute_path()}/data/vla.{version.value}.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_VLBA_Telescope(cls):
        path = f"{get_module_absolute_path()}/data/vlba.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_WSRT_Telescope(cls):
        path = f"{get_module_absolute_path()}/data/WSRT.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def get_OSKAR_Example_Telescope(cls):
        path = f"{get_module_absolute_path()}/data/telescope.tm"
        return cls.read_OSKAR_tm_file(path)

    @classmethod
    def read_OSKAR_tm_file(cls, path: str) -> any:
        abs_station_dir_paths = []
        station_position_file = None
        station_layout_file = None
        for file_or_dir in os.listdir(path):
            if file_or_dir.startswith("position"):
                station_position_file = os.path.abspath(os.path.join(path, file_or_dir))
            if file_or_dir.startswith("layout"):
                station_layout_file = os.path.abspath(os.path.join(path, file_or_dir))
            if file_or_dir.startswith("station"):
                abs_station_dir_paths.append(
                    os.path.abspath(os.path.join(path, file_or_dir))
                )

        if station_position_file is None:
            raise karabo.error.KaraboError("Missing crucial position.txt file_or_dir")

        if station_layout_file is None:
            raise karabo.error.KaraboError(
                "Missing layout.txt file in station directory. Only Layout.txt is support. "
                "The layout_ecef.txt and layout_wgs84.txt as "
                "defined in the OSKAR Telescope .tm specification are not currently supported."
            )

        telescope = None

        position_file = open(station_position_file)
        lines = position_file.readlines()
        for line in lines:
            long_lat = line.split(" ")
            if len(long_lat) > 3:
                raise karabo.error.KaraboError("Too many values in position.txt")
            long = float(long_lat[0])
            lat = float(long_lat[1])
            alt = 0
            if len(long_lat) == 3:
                alt = float(long_lat[2])
            telescope = Telescope(long, lat, alt)

        if Telescope is None:
            raise karabo.error.KaraboError(
                "Could not create Telescope from position.txt file_or_dir."
            )

        position_file.close()

        station_positions = cls.__read_layout_txt(station_layout_file)
        for station_position in station_positions:
            telescope.add_station(
                station_position[0],
                station_position[1],
                station_position[2],
                station_position[3],
                station_position[4],
                station_position[5],
            )

        if len(abs_station_dir_paths) != len(telescope.stations):
            raise karabo.error.KaraboError(
                f"There are {len(telescope.stations)} stations "
                f"but {len(abs_station_dir_paths)} "
                f"station directories."
            )

        for station_dir, station in zip(abs_station_dir_paths, telescope.stations):
            antenna_positions = cls.__read_layout_txt(
                os.path.join(station_dir, "layout.txt")
            )
            for antenna_pos in antenna_positions:
                station.add_station_antenna(
                    EastNorthCoordinate(
                        antenna_pos[0],
                        antenna_pos[1],
                        antenna_pos[2],
                        antenna_pos[3],
                        antenna_pos[4],
                        antenna_pos[5],
                    )
                )

        telescope.path = path
        return telescope

    @classmethod
    def __read_layout_txt(cls, path) -> List[List[float]]:
        positions: List[List[float]] = []
        layout_file = open(path)
        lines = layout_file.readlines()
        for line in lines:
            station_position = re.split("[\\s,]+", line)
            values = np.zeros(6)
            i = 0
            for pos in station_position:
                values[i] = cls.__float_try_parse(pos)
                i += 1
            positions.append(
                [values[0], values[1], values[2], values[3], values[4], values[5]]
            )
        layout_file.close()
        return positions

    @classmethod
    def __float_try_parse(cls, value):
        try:
            return float(value)
        except ValueError:
            return 0.0


def compute_distance(i, j, station_x, station_y):
    return np.sqrt(
        (station_x[i] - station_x[j]) ** 2 + (station_y[i] - station_y[j]) ** 2
    )


def create_baseline_cut_telelescope(lcut, hcut, tel):
    stations = np.loadtxt(tel.path + "/layout.txt")
    station_x = stations[:, 0]
    station_y = stations[:, 1]
    nb = comb(stations.shape[0], 2)
    k = 0
    baseline = np.zeros(nb)
    baseline_x = np.zeros(nb)
    baseline_y = np.zeros(nb)
    for i in range(stations.shape[0]):
        for j in range(i):
            baseline[k] = compute_distance(i, j, station_x, station_y)
            baseline_x[k] = i
            baseline_y[k] = j
            k = k + 1
    cut_idx = np.where((baseline > lcut) & (baseline < hcut))
    cut_baseline_x = baseline_x[cut_idx]
    cut_baseline_y = baseline_y[cut_idx]
    cut_station_list = np.unique(np.hstack((cut_baseline_x, cut_baseline_y)))
    output_path = tel.path.split("data/")[0] + "data/" + "tel_baseline_cut.tm"
    os.system("rm -rf " + output_path)
    os.system("mkdir " + output_path)
    l = 0
    for ns in cut_station_list:
        os.system("cp -r " + tel.path + "/station0" + str(int(ns)) + " " + output_path)
        os.system(
            "mv "
            + output_path
            + "/station0"
            + str(int(ns))
            + " "
            + output_path
            + "/station0"
            + "%02d" % (int(l))
        )
        l = l + 1
    cut_stations = stations[cut_station_list.astype(int)]
    os.system("cp -r " + tel.path + "/position.txt " + output_path)
    np.savetxt(output_path + "/layout.txt", cut_stations)
    return output_path
