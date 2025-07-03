""" Radio Frequency Interference (RFI) signal simulation """
import os
import subprocess
from shutil import copyfile
from typing import Dict, Optional, Union

import yaml

from karabo.simulation.observation import ObservationAbstract
from karabo.simulation.telescope import Telescope
from karabo.util._types import FilePathType, IntFloat
from karabo.util.data_util import get_module_absolute_path
from karabo.util.file_handler import FileHandler

_TABSIM_BINARY = "sim-vis"
TABSIM_DATA_DIR = os.path.join("data", "tabsim")


"""
tab-sim expects lists in the 'flow style' format, i.e.
sat_names: [navstar, sputnik] instead of
sat_names:
  - navstar
  - sputnik

The following lines configure the yaml package to do so.
"""


class FlowStyleList(list):
    pass


def represent_flow_style_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


class RFISignal:
    """Base type for RFI simulations"""

    def __init__(self) -> None:
        """
        Initializes the RFISignal class.
        """

        self.G0_mean: float = 1.0
        """Mean of the Gaussian distribution for the starting gain amplitude."""

        self.G0_std: float = 0.0
        """Standard deviation of the Gaussian distribution for the starting gain amplitude."""  # noqa: E501

        self.Gt_std_amp: float = 0.0
        """Standard deviation of the Gaussian distribution for the variability of the gain amplitude [%]."""  # noqa: E501

        self.Gt_std_phase: float = 0.0
        """Gt_std_phase: Standard deviation of the Gaussian distribution for the variability of the gain phase [deg]."""  # noqa: E501

        self.Gt_corr_amp: float = 0.0
        """Correlation time of the Gaussian process for the time variability of the gain amplitude [hr]."""  # noqa: E501

        self.Gt_corr_phase: float = 0.0
        """Correlation time of the Gaussian process for the time variability of the gain phase [hr]."""  # noqa: E501

        self.random_seed: int = 999
        """Random seed. Set it it to a predefined value to get reproducible results."""  # noqa: E501

        """tabsim diagnostic output
            rfi_seps: Plots the separation of the RFI sources from the pointing
                direction over time. Set to True.
            src_alt: Plots the altitude of the pointing over time. Set to False
                because this can be obtained from other output.
            uv_cov: Plots the UV coverage of the observation. Set to False because
                this can be obtained from the visibilities.
        """
        self.rfi_seps: bool = True
        self.src_alt: bool = False
        self.uv_cov: bool = False

        """tabsim output settings
            zarr_output: Write simulation output as zarr file. Set to true, beacuse
                sim-vis needs zarr file output.
            ms_output: Write simulation output as measurement set file.
                This is the preferred output. Set to true.
            overwrite_output: Overwrite putput files if they exist. Set to true
        """
        self.zarr_output: bool = True
        self.ms_output: bool = True
        self.overwrite_output: bool = True

        self._observation: ObservationAbstract
        self._telescope: Telescope
        self._sat_names: list = FlowStyleList(["navstar"])  # adding 1 default satellite
        self._norad_ids: list = FlowStyleList([])

        # sim-vis downloads satellite data from space-tracks service. These file
        # will be cached here. On Linux, they can be found in
        # ~/.cache/karabo-LTM-<username>-<unique_id>/tabsim-files-<unique_id>
        self.cache_dir = FileHandler().get_tmp_dir(
            prefix="tabsim-files-",
            purpose="file cache for sim-vis",
            term="long",
            mkdir=True,
        )

        # temporary work directory for sim-vis. We put intermediate scripts and
        # output files here. On Linux, the location is probably
        # /tmp/karabo-STM-<username>-<unique_id>/tabsim-files-<unique_id>
        self.tmp_dir = FileHandler().get_tmp_dir(
            prefix="tabsim-files-",
            purpose="working directoryfor sim-vis",
            term="short",
            mkdir=True,
        )

    def __set_basic_properties(
        self,
    ) -> Dict[str, Dict[str, Union[str, IntFloat, bool, None]]]:
        """
        Sets basic properties for sim-vis. Most important parameters are:
        - output.zarr: When set to true write out data as zarr file
        - output.ms: Set to true (default) to get a measurement set

        It produces a lot of data, because we set both 'output.zarr' and
        'output.ms' to True. Adjust the parameters accordingly if you don't need
        both files.

        Returns:
            dict: A dictionary containing the basic properties.
        """
        return {
            "output": {
                "path": self.tmp_dir,
                "prefix": self._telescope.name,
                "overwrite": self.overwrite_output,
                "zarr": self.zarr_output,
                "ms": self.ms_output,
            },
            "gains": {
                "G0_mean": self.G0_mean,
                "G0_std": self.G0_std,
                "Gt_std_amp": self.Gt_std_amp,
                "Gt_std_phase": self.Gt_std_phase,
                "Gt_corr_amp": self.Gt_corr_amp,
                "Gt_corr_phase": self.Gt_corr_phase,
                "random_seed": self.random_seed,
            },
            "diagnostics": {
                "rfi_seps": self.rfi_seps,
                "src_alt": self.src_alt,
                "uv_cov": self.uv_cov,
            },
            "dask": {"max_chunk_MB": 100},
        }

    def __set_observation_properties(
        self,
    ) -> Dict[str, Dict[str, Union[str, IntFloat, bool, None]]]:
        """Get the properties of the observation.

        Returns:
            dict: A dictionary containing the properties of the observation.
        """
        return {
            "observation": {
                "target_name": f"pointing_{self._observation.phase_centre_ra_deg:.2f}_{self._observation.phase_centre_dec_deg:.2f}",  # noqa: E501
                "ra": self._observation.phase_centre_ra_deg,
                "dec": self._observation.phase_centre_dec_deg,
                "start_time_isot": self._observation.start_date_and_time.isoformat(),
                "int_time": self._observation.length.total_seconds()
                / self._observation.number_of_time_steps,
                "n_time": self._observation.number_of_time_steps,
                "n_int": 1024,  # default for tab_sim
                "start_freq": float(self._observation.start_frequency_hz),
                "chan_width": float(self._observation.frequency_increment_hz),
                "n_freq": self._observation.number_of_channels,
                "SEFD": 420,  # system equivivalent flux density (system noise) in Jy
                "auto_corrs": False,
                "no_w": False,
                "random_seed": 12345,
            },
        }

    def __set_telescope_properties(
        self,
    ) -> Dict[str, Dict[str, Union[str, IntFloat, bool]]]:
        """
        Get the properties of the telescope.
        The properties are used by the tab_sim command to run the simulation.
        In the .tm files used by Karabo the coordinates of the stations are
        given in ENU (east-north) coordinates. Thus, we create a file in
        this format.

        Returns:
            dict: A dictionary containing the properties of the observation.
        """
        stations = self._telescope.stations
        enu_file_path = os.path.join(self.cache_dir, "stations.enu.txt")
        with open(enu_file_path, "w") as enu_file:
            for station in stations:
                # Write the station name and its ENU coordinates to the file.
                enu_file.write(
                    f"{station.position.x:.4f} "
                    f"{station.position.y:.4f} "
                    f"{station.position.z:.4f}\n"
                )

        copyfile(enu_file_path, "stations.enu.txt")
        # need to do this because name is Optional[str] in Telescope
        telescope_name = self._telescope.name
        if telescope_name is None:
            telescope_name = "unknown_telescope"

        return {
            "telescope": {
                "name": telescope_name,  # self._telescope.name,
                "latitude": self._telescope.centre_latitude,
                "longitude": self._telescope.centre_longitude,
                "elevation": self._telescope.centre_altitude,
                "dish_d": 13.5,
                "enu_path": enu_file_path,
                # "itrf_path": "", # not used with Karabo .tm files.
                "n_ant": len(self._telescope.stations),
            }
        }

    def __set_satellite_properties(self, credentials_file: FilePathType) -> Dict:
        path_to_data_dir = os.path.join(get_module_absolute_path(), TABSIM_DATA_DIR)
        tle_file_cache = os.path.join(self.cache_dir, "tles")
        return {
            "rfi_sources": {
                "tle_satellite": {
                    "sat_names": FlowStyleList(self._sat_names),
                    "norad_ids": FlowStyleList(self._norad_ids),
                    "spacetrack_path": credentials_file,
                    "tle_dir": tle_file_cache,
                    "norad_spec_model": os.path.join(
                        path_to_data_dir, "norad_satellite.rfimodel"
                    ),
                    "power_scale": 1e-2,
                    "max_ang_sep": 30,  # degrees
                    "min_alt": 0,  # degrees
                    "vis_step": 1,  # minutes
                }
            }
        }

    def _write_property_file(
        self, filename: FilePathType, credentials_file: FilePathType
    ) -> None:
        """Write the properties of the RFISignal to a YAML file.

        Args:
            filename (FilePathType): The name of the file to write the properties to.
            credentials_file (FilePathType): The name of the file containing the
                credentials for the spacetrack service. This file is mandatory.
        """

        with open(filename, "w") as file:
            file.write("# RFI Signal Properties\n\n")

        properties = {}
        properties.update(self.__set_telescope_properties())
        properties.update(self.__set_observation_properties())
        properties.update(self.__set_satellite_properties(credentials_file))
        properties.update(self.__set_basic_properties())

        # configure yaml to write lists in flow style
        yaml.add_representer(FlowStyleList, represent_flow_style_list)

        with open(filename, "a") as file:
            yaml.dump(properties, file)

    def run_simulation(
        self,
        observation: ObservationAbstract,
        telescope: Telescope,
        *,
        credentials_filename: FilePathType,
        property_filename: Optional[FilePathType] = None,
    ) -> None:
        """
        Simulates the RFI signal. We call `tab_sim` to run the simulation.
        `tab-sim` relies on information from the 'Space-Track' service to get the
        TLEs of the satellites. You need a login to use this service. The login
        is free, but you need to register at https://www.space-track.org/auth/login.
        You need to provide the username and password in a YAML file, which must
        be passed as `credentials_filename` to this method.

        Args:
            observation (ObservationAbstract): The observation object containing
            the observation details.
            site (Telescope): The telescope object containing the telescope details.
            credentials_filename (FilePathType): The name of the file containing
                the credentials for the spacetrack service. This file is mandatory.
            property_filename (Optional[FilePathType]): `sim-vis` reads the
                simulation properties from a .yaml file. Set the file name here if
                you want to keep this file. Otherwise Karabo creates a temporary file.
        """

        if not os.path.isfile(credentials_filename):
            raise FileNotFoundError(
                f"Credentials file '{credentials_filename}' does not exist."
            )

        tmp_property_filename = os.path.join(
            self.cache_dir, "sim_target_properties.yaml"
        )

        self._observation = observation
        self._telescope = telescope

        self._write_property_file(tmp_property_filename, credentials_filename)

        # user requested to keep the file
        if property_filename is not None:
            copyfile(
                tmp_property_filename,
                property_filename,
            )

        command = [
            _TABSIM_BINARY,
            "-c",
            tmp_property_filename,
            "-st",
            credentials_filename,
        ]

        if self.overwrite_output:
            command.append("-o")

        completed_process = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            text=True,
            # Raises exception on return code != 0
            check=True,
        )
        print(f"sim-vis output:\n[{completed_process.stdout}]")
