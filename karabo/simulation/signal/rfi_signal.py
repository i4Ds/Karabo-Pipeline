"""Radio Frequency Interference (RFI) signal simulation"""

import logging
import os
import subprocess
from shutil import copyfile
from typing import Dict, Optional, Union

import numpy as np
import yaml

from karabo.simulation.observation import ObservationAbstract
from karabo.simulation.telescope import Telescope
from karabo.util._types import FilePathType, IntFloat
from karabo.util.data_util import get_module_absolute_path
from karabo.util.file_handler import FileHandler
from karabo.simulation.visibility import Visibility

# from karabo.simulation.interferometer import InterferometerSimulation

import xarray as xr

_TABSIM_BINARY = "/home/karabo/miniconda3/envs/tabsim/bin/sim-vis"
_TABSIM_CONFIG_WRITE_BINARY = "/home/karabo/miniconda3/envs/tabsim/bin/write-config"
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
    """
    Converts lists to compact flow style format [item1, item2, item3] instead of
    the default block style format with line breaks and dashes. This is required
    by tab-sim which expects satellite names and NORAD IDs in flow style format.

    Args:
        dumper: The YAML dumper object
        data: The list data to be represented

    Returns:
        A YAML sequence representation with flow_style=True
    """
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


class RFISignal:
    """Base type for RFI simulations"""

    def __init__(self, credentials_filename: FilePathType) -> None:
        """
        Initializes the RFISignal class and set the credentials filename for space-track service authentication.

        Args:
            credentials_filename (FilePathType): Path to the YAML file containing
                space-track.org credentials (username and password)
        """

        self.logger = logging.getLogger("karabo.simulation.signal.rfi_signal")
        self.logger.info("Initializing RFISignal class")

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
        self.rfi_seps: bool = False
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

        """tabsim satellite filter options defaults
            max_n_sats: Maximum number of satellites to include
            min_altitude_deg: Minimum angle above horizon
            max_angular_sep_deg: Maximum angular separation between satellites and phase centre
        """
        self._max_n_sats = None
        self._min_altitude_deg = 0.0
        self._max_angular_sep_deg = 30.0

        self._flux_scale = 1e-3

        self._observation: ObservationAbstract
        self._telescope: Telescope
        # self._simulation: InterferometerSimulation
        self._sat_names: list = FlowStyleList([])
        self._norad_ids: list = FlowStyleList([])
        self._credentials_filename = credentials_filename

        # specifies if the tabsim files are kept or deleted after.
        # The files can be quite big. If you do not plan to process
        # them further (outside of Karabo) it's save to delete it.
        self.keep_sim = True

        # takes the file path to an MS file and then adds the RFI visibilities to it
        self.accumulate_ms: Optional[FilePathType] = None

        # sim-vis downloads satellite data from space-tracks service. These file
        # will be cached here. On Linux, they can be found in
        # ~/.cache/karabo-LTM-<username>-<unique_id>/tabsim-files-<unique_id>
        self.cache_dir = FileHandler().get_tmp_dir(
            prefix="tabsim-files-",
            purpose="file cache for sim-vis",
            term="long",
            mkdir=True,
        )
        self.logger.debug(f"Cache directory created: {self.cache_dir}")

        # temporary work directory for sim-vis. We put intermediate scripts and
        # output files here. On Linux, the location is probably
        # /tmp/karabo-STM-<username>-<unique_id>/tabsim-files-<unique_id>
        self.tmp_dir = FileHandler().get_tmp_dir(
            prefix="tabsim-files-",
            purpose="working directory for sim-vis",
            term="short",
            mkdir=True,
        )
        self.logger.debug(f"Temporary directory created: {self.tmp_dir}")

    def set_satellites(
        self,
        sat_names: Optional[list[str]] = None,
        norad_ids: Optional[list[int]] = None,
    ) -> None:
        """
        Set the satellite names and NORAD IDs for RFI simulation.

        Args:
            sat_names (list[str]): List of satellite names (e.g., ["navstar", "galileo"])
            norad_ids (list[int]): List of NORAD catalog numbers (e.g., [12345, 67890])
        """
        if sat_names is not None:
            self._sat_names = FlowStyleList(sat_names)
            self.logger.debug(f"Satellite names set to: {sat_names}")
        if norad_ids is not None:
            self._norad_ids = FlowStyleList(norad_ids)
            self.logger.debug(f"NORAD IDs set to: {norad_ids}")

    def filter_satellites(
        self,
        max_n_sats: Optional[int] = None,
        min_altitude_deg: Optional[float] = None,
        max_angular_sep_deg: Optional[float] = None,
    ):
        """
        Filter the satellites to include in the RFI simulation.

        Args:
            max_n_sats (int): Maximum number of satellites
            min_altitude_deg (float): Minimum altitude (angle above horizon) of the satellites in degrees
            max_angular_sep_deg (float): Maximum angular separation between the satellites and the phase centre in degrees
        """
        if max_n_sats is not None:
            self._max_n_sats = int(max_n_sats)
        if min_altitude_deg is not None:
            self._min_altitude_deg = float(min_altitude_deg)
        if max_angular_sep_deg is not None:
            self._max_angular_sep_deg = float(max_angular_sep_deg)

    def set_rfi_flux_scale(self, flux_scale: float):

        self._flux_scale = float(flux_scale)

    def __read_properties_from_ms(
        self, ms_path: str, tel_name: str, save_dir: str
    ) -> dict:
        """
        This will read the following properties from the MS file.
        telescope: [latitude, longitude, elevation, itrf_path, n_ant, dish_d]
        observation: [start_time_isot, int_time, n_time, start_freq, chan_width, n_freq]

        Args:
            ms_path (str): Path to the MS file generate by OSKAR
            tel_name (str): Telescope name.
            save_dir (str): Path to the save directory for the itrf coordinates and generated config file.

        Returns:
            sim_config (dict): Configuration parameters read from the MS file.
        """

        command = [
            _TABSIM_CONFIG_WRITE_BINARY,
            "-ms",
            ms_path,
            "-t",
            tel_name,
            "-s",
            save_dir,
        ]

        try:
            completed_process = subprocess.run(
                command,
                # shell=True,
                # executable=bash_path,
                capture_output=True,
                text=True,
                check=False,
            )

            stdout = completed_process.stdout
            stderr = completed_process.stderr

            # print(stdout)
            # print(stderr)

            config_path = stdout.rstrip().split(" : ")[-1]
            sim_config = yaml.load(open(config_path, "r"), yaml.Loader)

            return sim_config

        except:
            raise IOError("Could not write tabsim config file from MS file.")

    def __set_basic_properties(
        self,
    ) -> Dict[str, Dict[str, Union[str, FilePathType, IntFloat, bool, None]]]:
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
        self.logger.debug(f"Output path: {self.tmp_dir}")
        self.logger.debug(
            f"Output settings - zarr: {self.zarr_output}, "
            f"ms: {self.ms_output}, "
            f"overwrite: {self.overwrite_output}"
        )
        return {
            "output": {
                "path": self.tmp_dir,
                "prefix": self._telescope.name,
                "overwrite": self.overwrite_output,
                "zarr": self.zarr_output,
                "ms": self.ms_output,
                "keep_sim": self.keep_sim,
                "accumulate_ms": self.accumulate_ms,
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
                "ra": self._observation.phase_centre_ra_deg,  # OSKAR MS does not set this
                "dec": self._observation.phase_centre_dec_deg,  # OSKAR MS does not set this
                "n_int": 10,  # default for tab_sim
                "SEFD": 0,  # system equivivalent flux density (system noise) in Jy
                "auto_corrs": self._telescope.backend.value == "RASCIL",
                # "no_w": self._simulation.ignore_w_components,
                "no_w": False,
                "random_seed": 12345,
                # "start_time_isot": Set by MS file
                # "int_time": Set by MS file
                # "n_time": Set by MS file
                # "start_freq": Set by MS file
                # "chan_width": Set by MS file
                # "n_freq": Set by MS file
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
        # need to do this because name is Optional[str] in Telescope
        telescope_name = self._telescope.name
        if telescope_name is None:
            telescope_name = "unknown_telescope"

        return {
            "telescope": {
                "name": telescope_name,  # self._telescope.name,
                "dish_d": 1.0,  # OSKAR defaults to 1 metre
                # "latitude": Set by MS file
                # "longitude": Set by MS file
                # "elevation": Set by MS file
                # "itrf_path":  Set by MS file
                # "n_ant": Set by MS file
            }
        }

    def __set_satellite_properties(self, credentials_file: FilePathType) -> Dict:
        path_to_data_dir = os.path.join(get_module_absolute_path(), TABSIM_DATA_DIR)
        tle_file_cache = os.path.join(self.cache_dir, "tles")
        self.logger.debug(f"TLE cache directory: {tle_file_cache}")
        self.logger.debug(f"Using credentials file: {credentials_file}")
        return {
            "rfi_sources": {
                "tle_satellite": {
                    "max_n_sat": self._max_n_sats,
                    "sat_names": FlowStyleList(self._sat_names),
                    "norad_ids": FlowStyleList(self._norad_ids),
                    "spacetrack_path": credentials_file,
                    "tle_dir": tle_file_cache,
                    "norad_spec_model": os.path.join(
                        path_to_data_dir, "norad_satellite.rfimodel"
                    ),
                    "power_scale": self._flux_scale,
                    "max_ang_sep": self._max_angular_sep_deg,  # degrees
                    "min_alt": self._min_altitude_deg,  # degrees
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

        try:
            with open(filename, "w") as file:
                file.write("# RFI Signal Properties\n\n")
        except IOError as e:
            self.logger.error(f"Failed to write property file header: {str(e)}")
            raise

        save_dir = os.path.split(filename)[0]

        properties = {}
        properties.update(self.__set_telescope_properties())
        properties.update(self.__set_observation_properties())
        properties.update(self.__set_satellite_properties(credentials_file))
        properties.update(self.__set_basic_properties())
        ms_sim_config = self.__read_properties_from_ms(
            properties["output"]["accumulate_ms"],
            properties["telescope"]["name"],
            save_dir,
        )
        properties["telescope"].update(ms_sim_config["telescope"])
        properties["observation"].update(ms_sim_config["observation"])

        self.properties = properties

        # configure yaml to write lists in flow style
        yaml.add_representer(FlowStyleList, represent_flow_style_list)

        try:
            with open(filename, "a") as file:
                yaml.dump(properties, file)
            self.logger.debug(f"Property file written successfully: {filename}")
        except IOError as e:
            self.logger.error(f"Failed to write property file content: {str(e)}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"YAML serialization error: {str(e)}")
            raise

    def _set_dataset(self, zarr_path: str) -> None:

        if os.path.exists(zarr_path):
            self.xds = xr.open_zarr(zarr_path)
        else:
            raise IOError("")

    def _get_paths_from_stdout(self, stdout: str) -> tuple[str, str]:

        for line in stdout.split(os.linesep):
            if "Writing data to :" in line:
                sim_dir = line.split(" : ")[-1]
                sim_name = os.path.split(sim_dir)[-1]
                zarr_path = os.path.join(sim_dir, sim_name + ".zarr")
                ms_path = os.path.join(sim_dir, sim_name + ".ms")
                return zarr_path, ms_path

        return "", ""

    def _xyz_to_radec(self, xyz: np.ndarray) -> np.ndarray:
        if xyz.ndim == 2:
            xyz = xyz[None, :, :]

        xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
        radec = np.zeros((*xyz.shape[:2], 2))
        radec[:, :, 0] = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])
        radec[:, :, 1] = np.arcsin(xyz[:, :, 2])

        return np.rad2deg(radec)

    def get_satellite_radec(self) -> np.ndarray:

        xyz = (
            self.xds.rfi_tle_sat_xyz.data[:, :: self.xds.n_int_samples]
            - self.xds.ants_xyz.data.mean(axis=1)[None, :: self.xds.n_int_samples, :]
        )

        radec = self._xyz_to_radec(xyz.compute())

        return radec

    def run_simulation(
        self,
        observation: ObservationAbstract,
        telescope: Telescope,
        *,
        property_filename: Optional[FilePathType] = None,
    ) -> Visibility:
        """
        Simulates the RFI signal. We call `tab-sim` to run the simulation.
        `tab-sim` relies on information from the 'Space-Track' service to get the
        TLEs of the satellites. You need a login to use this service. The login
        is free, but you need to register at https://www.space-track.org/auth/login.
        You need to provide the username and password in a YAML file, which must
        be set using the set_credentials_filename() method before calling this method.

        Args:
            observation (ObservationAbstract): The observation object containing
            the observation details.
            site (Telescope): The telescope object containing the telescope details.
            property_filename (Optional[FilePathType]): `sim-vis` reads the
                simulation properties from a .yaml file. Set the file name here if
                you want to keep this file. Otherwise Karabo creates a temporary file.
        """

        if len(self._norad_ids) == 0 and len(self._sat_names) == 0:
            return Visibility(self.accumulate_ms)
        elif self._max_n_sats == 0:
            return Visibility(self.accumulate_ms)

        self.logger.info("Starting RFI signal simulation")

        if self._credentials_filename is None:
            self.logger.error(
                "Credentials filename not set. Use set_credentials_filename() method."
            )
            raise ValueError(
                "Credentials filename not set. Use set_credentials_filename() method."
            )

        if not os.path.isfile(self._credentials_filename):
            self.logger.error(
                f"Credentials file '{self._credentials_filename}' does not exist"
            )
            raise FileNotFoundError(
                f"Credentials file '{self._credentials_filename}' does not exist."
            )

        tmp_property_filename = os.path.join(
            self.cache_dir, "sim_target_properties.yaml"
        )

        self._observation = observation
        self._telescope = telescope

        self.logger.debug(f"Creating temporary property file: {tmp_property_filename}")
        self._write_property_file(tmp_property_filename, self._credentials_filename)
        self.logger.debug("Property file created successfully")

        # user requested to keep the file
        if property_filename is not None:
            self.logger.info(f"Copying property file to: {property_filename}")
            copyfile(
                tmp_property_filename,
                property_filename,
            )

        print(tmp_property_filename)

        command = [
            _TABSIM_BINARY,
            "-c",
            tmp_property_filename,
            "-st",
            self._credentials_filename,
        ]

        if self.overwrite_output:
            command.append("-o")

        self.logger.info(f"Executing sim-vis command: {' '.join(command)}")

        # import shutil

        # bash_path = shutil.which("bash")
        try:
            completed_process = subprocess.run(
                command,
                # shell=True,
                # executable=bash_path,
                capture_output=True,
                text=True,
                # sim-vis retuns 1 on success. Don't need an excception here
                check=False,
            )

            stdout = completed_process.stdout
            stderr = completed_process.stderr
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
            self.zarr_path, self.ms_path = self._get_paths_from_stdout(stdout)

            if self.properties["output"]["keep_sim"] and self.zarr_path:
                self._set_dataset(self.zarr_path)
                if self.properties["output"]["ms"]:
                    return Visibility(self.ms_path)

            # If tabsim deletes its data then return the original MS path with RFI added
            return Visibility(self.accumulate_ms)

        except Exception as e:
            self.logger.error(f"Error running sim-vis: {str(e)}")
            raise
