from __future__ import annotations

import enum
import glob
import logging
import os
import re
import shutil
from itertools import combinations
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    overload,
)

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import EarthLocation
from numpy.typing import NDArray
from oskar.telescope import Telescope as OskarTelescope
from scipy.spatial.distance import pdist
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.configuration.config_model import Configuration
from typing_extensions import assert_never

import karabo.error
from karabo.error import KaraboError
from karabo.simulation.coordinate_helper import wgs84_to_cartesian
from karabo.simulation.east_north_coordinate import EastNorthCoordinate
from karabo.simulation.station import Station
from karabo.simulation.telescope_versions import (
    ACAVersions,
    ALMAVersions,
    ATCAVersions,
    CARMAVersions,
    NGVLAVersions,
    PDBIVersions,
    SKALowAA0Point5Versions,
    SKALowAA1Versions,
    SKALowAA2Versions,
    SKALowAA4Versions,
    SKALowAAStarVersions,
    SKAMidAA0Point5Versions,
    SKAMidAA1Versions,
    SKAMidAA2Versions,
    SKAMidAA4Versions,
    SKAMidAAStarVersions,
    SMAVersions,
    VLAVersions,
)
from karabo.simulator_backend import SimulatorBackend
from karabo.util._types import DirPathType, NPFloatLike
from karabo.util.data_util import get_module_absolute_path
from karabo.util.file_handler import FileHandler, write_dir
from karabo.util.math_util import long_lat_to_cartesian

OSKARTelescopesWithVersionType = Literal[
    "ACA",
    "ALMA",
    "ATCA",
    "CARMA",
    "NGVLA",
    "PDBI",
    "SKA-LOW-AA0.5",
    "SKA-LOW-AA1",
    "SKA-LOW-AA2",
    "SKA-LOW-AA4",
    "SKA-LOW-AAstar",
    "SKA-MID-AA0.5",
    "SKA-MID-AA1",
    "SKA-MID-AA2",
    "SKA-MID-AA4",
    "SKA-MID-AAstar",
    "SMA",
    "VLA",
]
OSKARTelescopesWithoutVersionType = Literal[
    "EXAMPLE",
    "MeerKAT",
    "ASKAP",
    "LOFAR",
    "MKATPlus",
    "SKA1LOW",
    "SKA1MID",
    "VLBA",
    "WSRT",
]
# RASCIL Telescopes based on:
# https://developer.skatelescope.org/projects/ska-sdp-datamodels/en/latest/_modules/ska_sdp_datamodels/configuration/config_create.html#create_named_configuration # noqa: E501
RASCILTelescopes = Literal[
    "LOWBD2",
    "LOWBD2-CORE",
    "LOW",
    "LOWR3",
    "LOWR4",
    "LOW-AA0.5",
    "MID",
    "MIDR5",
    "MID-AA0.5",
    "MEERKAT+",
    "ASKAP",
    "LOFAR",
    "VLAA",
    "VLAA_north",
]

OSKAR_TELESCOPE_TO_FILENAMES: Dict[
    Union[OSKARTelescopesWithVersionType, OSKARTelescopesWithoutVersionType],
    str,
] = {
    "EXAMPLE": "telescope.tm",
    "MeerKAT": "meerkat.tm",
    "ACA": "aca.{0}.tm",
    "ALMA": "alma.{0}.tm",
    "ASKAP": "askap.tm",
    "ATCA": "atca_{0}.tm",
    "CARMA": "carma.{0}.tm",
    "LOFAR": "lofar.tm",
    "MKATPlus": "mkatplus.tm",
    "NGVLA": "ngvla-{0}.tm",
    "PDBI": "pdbi-{0}.tm",
    "SKA-LOW-AA0.5": "SKA-LOW-AA0.5.{0}.tm",
    "SKA-LOW-AA1": "SKA-LOW-AA1.{0}.tm",
    "SKA-LOW-AA2": "SKA-LOW-AA2.{0}.tm",
    "SKA-LOW-AA4": "SKA-LOW-AA4.{0}.tm",
    "SKA-LOW-AAstar": "SKA-LOW-AAstar.{0}.tm",
    "SKA-MID-AA0.5": "SKA-MID-AA0.5.{0}.tm",
    "SKA-MID-AA1": "SKA-MID-AA1.{0}.tm",
    "SKA-MID-AA2": "SKA-MID-AA2.{0}.tm",
    "SKA-MID-AA4": "SKA-MID-AA4.{0}.tm",
    "SKA-MID-AAstar": "SKA-MID-AAstar.{0}.tm",
    "SKA1LOW": "ska1low.tm",
    "SKA1MID": "ska1mid.tm",
    "SMA": "sma.{0}.tm",
    "VLA": "vla.{0}.tm",
    "VLBA": "vlba.tm",
    "WSRT": "WSRT.tm",
}
OSKAR_TELESCOPE_TO_VERSIONS: Dict[OSKARTelescopesWithVersionType, Type[enum.Enum]] = {
    "ACA": ACAVersions,
    "ALMA": ALMAVersions,
    "ATCA": ATCAVersions,
    "CARMA": CARMAVersions,
    "NGVLA": NGVLAVersions,
    "PDBI": PDBIVersions,
    "SKA-LOW-AA0.5": SKALowAA0Point5Versions,
    "SKA-LOW-AA1": SKALowAA1Versions,
    "SKA-LOW-AA2": SKALowAA2Versions,
    "SKA-LOW-AA4": SKALowAA4Versions,
    "SKA-LOW-AAstar": SKALowAAStarVersions,
    "SKA-MID-AA0.5": SKAMidAA0Point5Versions,
    "SKA-MID-AA1": SKAMidAA1Versions,
    "SKA-MID-AA2": SKAMidAA2Versions,
    "SKA-MID-AA4": SKAMidAA4Versions,
    "SKA-MID-AAstar": SKAMidAAStarVersions,
    "SMA": SMAVersions,
    "VLA": VLAVersions,
}


class Telescope:
    """Telescope

    WGS84 longitude and latitude and altitude in metres centre of the telescope.png
    centre. A telescope is described as follows:

    Each row represents one station, with the elements being the horizontal x (east),
    horizontal y (north), and horizontal z (up) coordinates,
    followed by the errors in horizontal y (east), horizontal y (north),
    and horizontal z (up).
    Example: [[x, y, z, error_x, error_y, error_z], [...]]

    centre_longitude : float
        WGS84 longitude at the center of the telescope.
    centre_latitude : float
        WGS84 latitude at the center of the telescope.
    centre_altitude : float
        Altitude (in meters) at the center of the telescope.
    """

    def __init__(
        self, longitude: float, latitude: float, altitude: float = 0.0
    ) -> None:
        """__init__ method

        Parameters
        ----------
        longitude : float
            WGS84 longitude at the center of the telescope.
        latitude : float
            WGS84 latitude at the center of the telescope.
        altitude : float, optional
            Altitude (in meters) at the center of the telescope, default is 0.
        """
        self.path: Optional[DirPathType] = None
        self._name: Optional[str] = None
        self.centre_longitude = longitude
        self.centre_latitude = latitude
        self.centre_altitude = altitude

        self.stations: List[Station] = []

        self.backend: SimulatorBackend = SimulatorBackend.OSKAR

        self.RASCIL_configuration: Optional[Configuration] = None

    @overload
    @classmethod
    def constructor(
        cls,
        name: OSKARTelescopesWithVersionType,
        version: enum.Enum,
        backend: Literal[SimulatorBackend.OSKAR] = SimulatorBackend.OSKAR,
    ) -> Telescope:
        ...

    @overload
    @classmethod
    def constructor(
        cls,
        name: OSKARTelescopesWithoutVersionType,
        version: Literal[None] = None,
        backend: Literal[SimulatorBackend.OSKAR] = SimulatorBackend.OSKAR,
    ) -> Telescope:
        ...

    @overload
    @classmethod
    def constructor(
        cls,
        name: RASCILTelescopes,
        version: Literal[None] = None,
        backend: Literal[SimulatorBackend.RASCIL] = SimulatorBackend.RASCIL,
    ) -> Telescope:
        ...

    @classmethod
    def constructor(
        cls,
        name: Union[
            RASCILTelescopes,
            OSKARTelescopesWithVersionType,
            OSKARTelescopesWithoutVersionType,
        ],
        version: Optional[enum.Enum] = None,
        backend: SimulatorBackend = SimulatorBackend.OSKAR,
    ) -> Telescope:
        """Main constructor to obtain a pre-configured telescope instance.
        :param name: Name of the desired telescope configuration.
            This name, together with the backend, is used as the key
            to look up the correct telescope specification file.
        :param version: Version details required for some
            telescope configurations. Defaults to None.
        :param backend: Underlying package to be used for the telescope configuration,
            since each package stores the arrays in a different format.
            Defaults to OSKAR.
        :raise ValueError: If the combination of input parameters is invalid.
            Specifically, if the requested telescope requires a version,
            but an invalid version (or no version) is provided,
            or if the requested telescope name is not
            supported by the requested backend.

        :return: Telescope instance
        """
        if backend is SimulatorBackend.OSKAR:
            # Explicitly cast name depending on whether it requires a telescope version
            # This should no longer be necessary when mypy starts supporting
            # type narrowing with get_args.
            # https://github.com/python/mypy/issues/12535
            if name in get_args(OSKARTelescopesWithVersionType):
                name = cast(OSKARTelescopesWithVersionType, name)
                data_path = OSKAR_TELESCOPE_TO_FILENAMES[name]
                accepted_versions = OSKAR_TELESCOPE_TO_VERSIONS[name]
                assert (
                    version is not None
                ), f"version is a required field for telescope {name}, \
but was not provided. Please provide a value for the version field."
                assert (
                    version in accepted_versions
                ), f"""{version = } is not one of the accepted versions.
                List of accepted versions: {accepted_versions}"""
                data_path = data_path.format(version.value)
            elif name in get_args(OSKARTelescopesWithoutVersionType):
                name = cast(OSKARTelescopesWithoutVersionType, name)
                data_path = OSKAR_TELESCOPE_TO_FILENAMES[name]
                assert (
                    version is None
                ), f"""version is not a required field
                    for telescope {name}, but {version} was provided.
                    Please do not provide a value for the version field."""
            else:
                raise TypeError(
                    f"""
                    {name = } is not an accepted telescope name for this backend.
                """
                )

            path = os.path.join(get_module_absolute_path(), "data", data_path)
            return cls.read_OSKAR_tm_file(path)
        elif backend is SimulatorBackend.RASCIL:
            if version is not None:
                logging.warning(
                    f"""The version parameter is not supported
    by the backend {backend}.
    The version value {version} provided will be ignored."""
                )
            assert name in get_args(RASCILTelescopes)
            try:
                telescope: Telescope = cls.__convert_to_karabo_telescope(name)
            except ValueError as e:
                raise ValueError(
                    f"""Requested telescope {name} is not supported by this backend.
                    For more details, see
    https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/blob/d6dcce6288a7bf6d9ce63ab16e799977723e7ae5/src/ska_sdp_datamodels/configuration/config_create.py"""  # noqa
                ) from e

            # Function like _get_station_infos() and
            # create_baseline_cut_telescope() need access to an OSKAR telescope
            # model (.tm). This is not available for RASCIL datasets.
            # Thus, we create a temporary one.
            disk_cache = FileHandler().get_tmp_dir(
                prefix="telescope-constructor-rascil-",
                mkdir=False,
            )
            tm_path = os.path.join(disk_cache, f"{name}.tm")

            telescope.write_to_disk(tm_path)
            telescope.path = tm_path
            return telescope
        else:
            assert_never(backend)

    @classmethod
    def __convert_to_karabo_telescope(cls, instr_name: str) -> Telescope:
        """
        Converts a site saved in RASCIl data format into a Karabo Telescope.
            This function acts as an adapter to make the functionality in Telescope
            class work for a RASCIL telescope. Namely the functions max_baseline()
            and get_baseline_lengths().
            It derives the necessary data structures from the RASCIL_configuration
            and fits them into those of the Telescope class. The resuting class is
            a SimulatorBackend.RASCIL but has the stations: List[Station]
            list filled as well. Nevertheless, it should only be used as a RASCIL
            telescope class.

        :param instr_name: The name of the instrument to convert.
        :raise ValueError: If instr_name is not a valid RASCIL telescope
        :returns: An instance of Karabo Telescope.
        :rtype: karabo.simulation.telescope.Telecope

        """
        config = create_named_configuration(instr_name)

        site_location_gc: EarthLocation = config.location
        # this conversion returns complex type with unit
        # lon,lat,alt = site_location_gc.geodetic
        longitude = site_location_gc.lon.to("deg").value
        latitude = site_location_gc.lat.to("deg").value
        altitude = site_location_gc.height.to("m").value

        telescope = Telescope(longitude, latitude, altitude)
        # This is used in some inteferometer simulations
        telescope.RASCIL_configuration = config

        station_coords = config.xyz.data
        for i, coord in enumerate(station_coords):
            telescope.add_station(
                horizontal_x=coord[0],
                horizontal_y=coord[1],
                horizontal_z=coord[2],
            )

            # there are only stations in the rascil files no antennas.
            # we add a dummy antenna in order to avoid the creation
            # of an empty file. This matches other files. See
            # karabo/data/aca.all.tm/station000/layout.txt for example.
            # Reason: Value not set to 0 probably to compensate
            # for dish diameter. (see comment for PR #631)
            telescope.add_antenna_to_station(i, 0.1, 0.1)
        telescope.backend = SimulatorBackend.RASCIL
        return telescope

    @property
    def name(self) -> Optional[str]:
        """Gets the telescope name (if available).

        It's just the file-name of the referred telescope-file without the ending.

        Returns:
            Telescope name or `None`.
        """
        if self._name is not None:
            return self._name
        if self.path is None:
            return None
        return os.path.split(self.path)[-1].split(".")[0]

    @name.setter
    def name(self, value: str) -> None:
        """Sets the name of the telescope. Usually, this is the name
        of the telescope model file w/o ending
        """
        self._name = value

    def get_backend_specific_information(self) -> Union[DirPathType, Configuration]:
        if self.backend is SimulatorBackend.OSKAR:
            return self.path
        if self.backend is SimulatorBackend.RASCIL:
            return self.RASCIL_configuration

        raise ValueError(
            f"""Unexpected: current backend is set to {self.backend},
        but expected one of {SimulatorBackend}.
        Verify the construction of this Telescope instance."""
        )

    def add_station(
        self,
        horizontal_x: float,
        horizontal_y: float,
        horizontal_z: float = 0.0,
        horizontal_x_coordinate_error: float = 0.0,
        horizontal_y_coordinate_error: float = 0.0,
        horizontal_z_coordinate_error: float = 0.0,
    ) -> None:
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
        :param horizontal_x_coordinate_error: east coordinate error
            relative to the station center in metres
        :param horizontal_y_coordinate_error: north coordinate error
            relative to the station center in metres
        :param horizontal_z_coordinate_error: altitude of antenna error

        :return:
        """
        if station_index >= 0 and station_index < len(self.stations):
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
        else:
            raise IndexError(
                "You tried to add an antenna to a station that doesn't exist.\n"
                f"station_index must be between 0 and {len(self.stations)-1}"
            )

    def plot_telescope(self, file: Optional[str] = None) -> None:
        """
        Plot the telescope according to which backend is being used,
        and save the resulting image into a file, if any is provided.
        """
        if self.backend is SimulatorBackend.OSKAR:
            self.plot_telescope_OSKAR(file)
        elif self.backend is SimulatorBackend.RASCIL:
            # we can use plot_telescope_OSKAR here because we converted
            # the RASCIl setup into an OSKAR setup when constructing it.
            self.plot_telescope_OSKAR(file)
        else:
            logging.warning(
                f"""Backend {self.backend} is not valid.
            Proceeding without any further actions."""
            )
            return

    def plot_telescope_OSKAR(
        self, file: Optional[str] = None, block: bool = False
    ) -> None:
        """
        Plot the telescope and all its stations and antennas with longitude altitude
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        station_x = []
        station_y = []
        for station in self.stations:
            station_x.append(station.longitude)
            station_y.append(station.latitude)

        # we set the colour manually in order to keep the colour scheme.
        ax.scatter(station_x, station_y, label="Stations", c="tab:orange")

        x = np.array([self.centre_longitude])
        y = np.array([self.centre_latitude])

        ax.scatter(x, y, label="Centre", c="tab:green")
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(f"{self.name} Overview")
        ax.legend(loc="upper left", shadow=False, fontsize="medium")

        if file is not None:
            plt.savefig(file)
            plt.close(fig)
        else:
            plt.show(block=block)
            plt.pause(1)

    def get_OSKAR_telescope(self) -> OskarTelescope:
        """
        Retrieve the OSKAR Telescope object from the karabo.Telescope object.

        :return: OSKAR Telescope object
        """
        tmp_dir = FileHandler().get_tmp_dir(
            prefix="telescope-",
            purpose="telescope disk-cache",
            unique=self,
            mkdir=False,
        )
        tmp_dir = os.path.join(tmp_dir, "oskar-telescope.tm")
        self.write_to_disk(dir_name=tmp_dir, overwrite=True)
        tel = OskarTelescope()
        tel.load(tmp_dir)
        self.path = tmp_dir
        return tel

    def write_to_disk(self, dir_name: DirPathType, *, overwrite: bool = False) -> None:
        """Write `dir_path` to disk (must have .tm ending).

        :param dir: directory in which the configuration will be saved in.
        :param overwrite: If True an existing directory is overwritten if exists. Be
            careful to put the correct dir as input because the old one can get removed!
        """
        if not str(dir_name).endswith(
            ".tm"
        ):  # for OSKAR & `overwrite` security purpose
            err_msg = f"{dir_name=} has to end with a `.tm`, but doesn't."
            raise RuntimeError(err_msg)
        with write_dir(dir=dir_name, overwrite=overwrite) as wd:
            self.__write_position_txt(os.path.join(wd, "position.txt"))
            self.__write_layout_txt(
                os.path.join(wd, "layout.txt"),
                [station.position for station in self.stations],
            )
            for i, station in enumerate(self.stations):
                station_path = f"{wd}{os.path.sep}station{'{:03d}'.format(i)}"
                os.mkdir(station_path)
                self.__write_layout_txt(
                    os.path.join(station_path, "layout.txt"),
                    station.antennas,
                )

    def __write_position_txt(self, position_file_path: str) -> None:
        position_file = open(position_file_path, "a")

        position_file.write(
            f"{self.centre_longitude} {self.centre_latitude} {self.centre_altitude}\n"
        )
        position_file.close()

    def __write_layout_txt(
        self, layout_path: str, elements: List[EastNorthCoordinate]
    ) -> None:
        layout_file = open(layout_path, "a")
        for element in elements:
            layout_file.write(
                f"{element.x} {element.y} {element.z} {element.x_error} "
                + f"{element.y_error} {element.z_error}\n"
            )
        layout_file.close()

    def get_cartesian_position(self) -> NDArray[np.float_]:
        return long_lat_to_cartesian(self.centre_latitude, self.centre_longitude)

    @classmethod
    def read_OSKAR_tm_file(cls, path: DirPathType) -> Telescope:
        """Reads an OSKAR telescope model from disk and
           returns an object of karabo.simulation.telescope.Telescope

        :param path: Path to a valid telescope model (extemsion * .tm)
        :return: A karabo.simulation.telescope.Telescope object. Importantn:
           The object has the backend set to SimulatorBackend.OSKAR.
        :raises: A karabo.error.KaraboError if the path does not exit,
           or the data in the file cannot be read.
        """
        path_ = str(path)
        abs_station_dir_paths = []
        center_position_file = None
        station_layout_file = None
        for file_or_dir in os.listdir(path_):
            if file_or_dir.startswith("position"):
                center_position_file = os.path.abspath(os.path.join(path_, file_or_dir))
            if file_or_dir.startswith("layout"):
                station_layout_file = os.path.abspath(os.path.join(path_, file_or_dir))
            if file_or_dir.startswith("station"):
                abs_station_dir_paths.append(
                    os.path.abspath(os.path.join(path_, file_or_dir))
                )

        if center_position_file is None:
            raise karabo.error.KaraboError(
                f"Missing crucial position.txt in {file_or_dir}"
            )

        if station_layout_file is None:
            raise karabo.error.KaraboError(
                "Missing layout.txt file in station directory. "
                "Only layout.txt is supported. "
                "The layout_ecef.txt and layout_wgs84.txt as "
                "defined in the OSKAR Telescope .tm specification are not "
                "supported currently."
            )

        telescope = None

        position_file = open(center_position_file)
        lines = position_file.readlines()
        for line in lines:
            match = re.match(
                r"^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)?\s*$",  # noqa: E501
                line.strip(),
            )  # one line with two or three numbers
            if match:
                numbers = [float(num) for num in match.group().split()]
                long = numbers[0]
                lat = numbers[1]
                alt = 0.0
                if len(numbers) == 3:
                    alt = float(numbers[2])
                telescope = Telescope(long, lat, alt)
                break

        if telescope is None:
            raise karabo.error.KaraboError(
                "Could not create Telescope from position.txt file_or_dir. "
                + "It must contain one line with two or three numbers."
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
        telescope.backend = SimulatorBackend.OSKAR
        return telescope

    @classmethod
    def __read_layout_txt(cls, path: str) -> List[List[float]]:
        positions: List[List[float]] = []
        layout_file = open(path)
        lines = layout_file.readlines()
        for line in lines:
            line = line.rstrip()
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
    def __float_try_parse(cls, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return 0.0

    @classmethod
    def _get_station_infos(cls, tel_path: DirPathType) -> pd.DataFrame:
        """Creates a pd.DataFrame with telescope-station infos.

        - "station-nr": Station-number inside the .tm file.
        - "station-path": Path of the according station.
        - "x": x-position
        - "y": y-position

        Args:
            tel_path: .tm dir-path to get infos from.

        Returns:
            pd.DataFrame with the according infos.
        """
        station_paths = glob.glob(f"{tel_path}{os.path.sep}station[0-9]*")
        if len(station_paths) <= 0:
            raise FileNotFoundError(f"No stations found in {tel_path}")
        station_numbers = list()
        for station_path in station_paths:
            station_number = os.path.split(station_path)[-1].split("station")[1]
            station_numbers.append(int(station_number))
        df_tel = (
            pd.DataFrame(
                {
                    "station-nr": station_numbers,
                    "station-path": station_paths,
                }
            )
            .sort_values(by="station-nr")
            .reset_index(drop=True)
        )
        if not np.all(df_tel["station-nr"].to_numpy() == np.arange(0, df_tel.shape[0])):
            raise KaraboError(
                f"Stations found in {tel_path} are not ascending from station<0 - n>. "
            )
        stations = np.array(cls.__read_layout_txt(os.path.join(tel_path, "layout.txt")))
        if (n_stations_layout := stations.shape[0]) != (n_stations := df_tel.shape[0]):
            raise KaraboError(
                f"Number of stations mismatch of {n_stations_layout=} & {n_stations=}"
            )
        df_tel["x"] = stations[:, 0]
        df_tel["y"] = stations[:, 1]
        return df_tel

    @classmethod
    def create_baseline_cut_telescope(
        cls,
        lcut: NPFloatLike,
        hcut: NPFloatLike,
        tel: Telescope,
        tm_path: Optional[DirPathType] = None,
    ) -> Tuple[DirPathType, Dict[str, str]]:
        """
        Returns a telescope model for telescope `tel` with baseline lengths
            only between `lcut` and `hcut` metres.

        :param lcut: Lower cut
        :param hcut: Higher cut
        :param tel: Telescope to cut off
        :param tm_path: .tm file-path to save the cut-telescope.
            `tm_path` will get overwritten if it already exists.

        :return: .tm file-path & station-name
            conversion (e.g. station055 -> station009)
        """
        if tel.path is None:
            raise KaraboError(
                "`tel.path` None indicates that there is not telescope.tm file "
                + "available for `tel`, which is not allowed here."
            )
        if tm_path is not None and not str(tm_path).endswith(".tm"):
            raise KaraboError(f"{tm_path=} must end with '.tm'.")
        df_tel = Telescope._get_station_infos(tel_path=tel.path)
        n_stations = df_tel.shape[0]
        baseline_idx: List[Tuple[int, int]] = list(combinations(range(n_stations), 2))

        n_baselines = len(baseline_idx)

        station_x = df_tel["x"].to_numpy()
        station_y = df_tel["y"].to_numpy()
        baseline_dist = np.zeros(n_baselines)
        for i, (x, y) in enumerate(baseline_idx):
            baseline_dist[i] = np.linalg.norm(station_x[x] - station_y[y])
        cut_idx = np.where((baseline_dist > lcut) & (baseline_dist < hcut))
        cut_station_list = np.unique(np.array(baseline_idx)[cut_idx])
        df_tel = df_tel[df_tel["station-nr"].isin(cut_station_list)].reset_index(
            drop=True
        )

        if cut_station_list.shape[0] == 0:
            raise KaraboError("All telescope-stations were cut off.")

        if tm_path is None:
            disk_cache = FileHandler().get_tmp_dir(
                prefix="telescope-baseline-cut-",
                mkdir=False,
            )
            tm_path = os.path.join(disk_cache, "telescope-baseline-cut.tm")
        else:
            if os.path.exists(tm_path):
                shutil.rmtree(tm_path)
        os.makedirs(tm_path, exist_ok=False)

        conversions: Dict[str, str] = dict()
        for i in range(df_tel.shape[0]):
            source_path = df_tel.iloc[i]["station-path"]
            number_str = str(i).zfill(3)
            target_station = f"station{number_str}"
            target_path = os.path.join(tm_path, target_station)
            source_station = os.path.split(source_path)[-1]
            conversions[source_station] = target_station
            shutil.copytree(src=source_path, dst=target_path)

        shutil.copyfile(
            src=os.path.join(tel.path, "position.txt"),
            dst=os.path.join(tm_path, "position.txt"),
        )
        cut_stations = df_tel[["x", "y"]].to_numpy()
        np.savetxt(os.path.join(tm_path, "layout.txt"), cut_stations, delimiter=" ")
        return tm_path, conversions

    def get_stations_wgs84(self) -> NDArray[np.float64]:
        """Gets the coordinates of the interferometer stations in WGS84.

        This function assumes that `self.stations` provides WGS84 coordinates.

        Returns:
            Stations lon[deg]/lat[deg]/alt[m] (nx3).
        """
        return np.array(
            [
                [station.longitude, station.latitude, station.altitude]
                for station in self.stations
            ]
        )

    @classmethod
    def get_baseline_lengths(
        cls,
        stations_wgs84: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Gets the interferometer baselines distances in meters.

        It's euclidean distance (aka geocentric), not geodesic.

        Args:
            baselines_wgs84: nx3 wgs84 baselines.

        Returns:
            Interferometer baselines dists in meters.
        """
        lon, lat, alt = (
            stations_wgs84[:, 0],
            stations_wgs84[:, 1],
            stations_wgs84[:, 2],
        )

        cart_coords: NDArray[np.float64] = wgs84_to_cartesian(lon, lat, alt)

        return cast(NDArray[np.float64], pdist(cart_coords))

    def max_baseline(self) -> np.float64:
        """Gets the longest baseline in meters.

        Returns:
            Length of longest baseline.
        """
        dists = self.get_baseline_lengths(stations_wgs84=self.get_stations_wgs84())
        max_distance = np.max(dists)
        return max_distance

    @classmethod
    def ang_res(cls, freq: float, b: float) -> float:
        """Calculates angular resolution in arcsec.

        Angular resolution: θ=λ/B
        Wavelength: λ=c/f
        B: Max baseline in meters

        Args:
            freq: Frequency [Hz].
            b: Max baseline in meters (e.g. from `max_baseline`).

        Returns:
            Angular resolution in arcsec.
        """
        ang_res = (const.c.value / freq) / b * u.deg
        ang_res_arcsec: float = ang_res.to(u.arcsec).value
        return ang_res_arcsec
