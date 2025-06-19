""" Radio Frequency Interference (RFI) signal simulation """
import os
import subprocess
from typing import Optional

import yaml

from karabo.simulation.observation import ObservationAbstract
from karabo.simulation.telescope import Telescope
from karabo.util._types import FilePathType
from karabo.util.file_handler import FileHandler

_TABSIM_BINARY = "sim-vis"


class RFISignal:
    """Base type for RFI simulations"""

    def __init__(self) -> None:
        """
        Initializes the RFISignal class.

        Args:
            observation: The observation object containing the observation details.
            site: The telescope object containing the telescope details.
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

        """tab_sim diagnostic output
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

    # write properties to disk as yaml file

    def _write_property_file(self, filename: FilePathType) -> None:
        """Write the properties of the RFISignal to a YAML file.

        Args:
            filename (str): The name of the file to write the properties to.
        """
        with open(filename, "w") as file:
            file.write("# RFI Signal Properties\n\n")

        properties = {
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
        `tab-sim` relies in information from the 'Space-Track' service to get the
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
                simulation proerties from a .yaml file. Set the file name here if
                you want to keep this file. Otherwise Karabo creates a temporay file.
        """

        if not os.path.isfile(credentials_filename):
            raise FileNotFoundError(
                f"Credentials file '{credentials_filename}' does not exist."
            )
        if property_filename is None:
            tmp_dir = FileHandler().get_tmp_dir(
                prefix="sim-vis-files-",
                purpose="temporary file with tab-sim properties",
            )
            property_filename = os.path.join(tmp_dir, "sim_target_properties.yaml")
            os.makedirs(tmp_dir, exist_ok=True)

        print(property_filename)
        self._write_property_file(property_filename)

        command = [
            _TABSIM_BINARY,
            "-c",
            property_filename,
            "-st",
            credentials_filename,
        ]

        completed_process = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            text=True,
            # Raises exception on return code != 0
            check=True,
        )
        print(f"WSClean output:\n[{completed_process.stdout}]")
