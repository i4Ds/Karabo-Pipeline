from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Union

from karabo.error import KaraboError
from karabo.imaging.image import Image
from karabo.imaging.imager_base import Imager, ImagerConfig
from karabo.util.file_handler import FileHandler


@dataclass
class WscleanImagerConfig(ImagerConfig):
    niter: int = 20

    @classmethod
    # TODO test this
    def from_imager_config(cls, imager_config: ImagerConfig) -> WscleanImagerConfig:
        return cls(
            ms_file_path=imager_config.ms_file_path,
            imaging_npixel=imager_config.imaging_npixel,
            imaging_cellsize=imager_config.imaging_cellsize,
        )


class WscleanImager(Imager):
    WSCLEAN_BINARY = "wsclean"

    TMP_PREFIX_DIRTY = "WSClean-dirty-"
    TMP_PURPOSE_DIRTY = "Disk cache for WSClean dirty images"
    TMP_PREFIX_CLEANED = "WSClean-cleaned-"
    TMP_PURPOSE_CLEANED = "Disk cache for WSClean cleaned images"
    TMP_PREFIX_CUSTOM = "WSClean-custom-"
    TMP_PURPOSE_CUSTOM = "Disk cache for WSClean custom command images"

    OUTPUT_FITS_DIRTY = "wsclean-dirty.fits"
    OUTPUT_FITS_DECONVOLVED = "wsclean-image.fits"
    OUTPUT_FITS_RESTORED = "wsclean-image.fits"
    OUTPUT_FITS_RESIDUAL = "wsclean-residual.fits"

    def create_dirty_image(self, config: ImagerConfig) -> Image:
        # TODO combine_across_frequencies
        # If config is an ImagerConfig (base class) instance, convert to
        # WscleanImagerConfig using default values for WSClean-specific configuration.
        if not isinstance(config, WscleanImagerConfig):
            config = WscleanImagerConfig.from_imager_config(config)
        tmp_dir = FileHandler().get_tmp_dir(
            prefix=self.TMP_PREFIX_DIRTY,
            purpose=self.TMP_PURPOSE_DIRTY,
        )
        command = self._get_command_prefix(tmp_dir) + (
            f"{self.WSCLEAN_BINARY} "
            f"-size {config.imaging_npixel} {config.imaging_npixel} "
            f"-scale {math.degrees(config.imaging_cellsize)}deg "
            f"{config.ms_file_path}"
        )
        print(f"WSClean command: [{command}]")
        completed_process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            # Raises exception on return code != 0
            check=True,
        )
        print(f"WSClean output:\n[{completed_process.stdout}]")

        return Image(path=os.path.join(tmp_dir, self.OUTPUT_FITS_DIRTY))

    def create_cleaned_image(self, config: ImagerConfig) -> Tuple[Image, Image, Image]:
        # If config is an ImagerConfig (base class) instance, convert to
        # WscleanImagerConfig using default values for WSClean-specific configuration.
        if not isinstance(config, WscleanImagerConfig):
            config = WscleanImagerConfig.from_imager_config(config)
        tmp_dir = FileHandler().get_tmp_dir(
            prefix=self.TMP_PREFIX_CLEANED,
            purpose=self.TMP_PURPOSE_CLEANED,
        )
        command = self._get_command_prefix(tmp_dir) + (
            f"{self.WSCLEAN_BINARY} "
            f"-size {config.imaging_npixel} {config.imaging_npixel} "
            f"-scale {math.degrees(config.imaging_cellsize)}deg "
            f"-niter {config.niter} "
            f"{config.ms_file_path}"
        )
        print(f"WSClean command: [{command}]")
        completed_process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            # Raises exception on return code != 0
            check=True,
        )
        print(f"WSClean output:\n[{completed_process.stdout}]")

        return (
            # TODO What exactly is the RASCIL deconvolved_image?
            Image(path=os.path.join(tmp_dir, self.OUTPUT_FITS_DECONVOLVED)),
            Image(path=os.path.join(tmp_dir, self.OUTPUT_FITS_RESTORED)),
            Image(path=os.path.join(tmp_dir, self.OUTPUT_FITS_RESIDUAL)),
        )

    def create_image_custom_command(
        self,
        command: str,
        output_filenames: Union[str, List[str]] = "wsclean-image.fits",
    ) -> Union[Image, List[Image]]:
        tmp_dir = FileHandler().get_tmp_dir(
            prefix=self.TMP_PREFIX_CUSTOM,
            purpose=self.TMP_PURPOSE_CUSTOM,
        )
        expected_command_prefix = f"{self.WSCLEAN_BINARY} "
        if not command.startswith(expected_command_prefix):
            raise KaraboError(
                "Unexpected command. Expecting command to start with "
                f'"{expected_command_prefix}".'
            )
        command = self._get_command_prefix(tmp_dir) + command
        print(f"WSClean command: [{command}]")
        completed_process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            # Raises exception on return code != 0
            check=True,
        )
        print(f"WSClean output:\n[{completed_process.stdout}]")

        if isinstance(output_filenames, str):
            return Image(path=os.path.join(tmp_dir, output_filenames))
        else:
            return [
                Image(path=os.path.join(tmp_dir, output_filename))
                for output_filename in output_filenames
            ]

    @staticmethod
    def _get_command_prefix(tmp_dir: str) -> str:
        return (
            f"cd {tmp_dir} && "
            # Avoids the following wsclean error:
            # This software was linked to a multi-threaded version of OpenBLAS.
            # OpenBLAS multi-threading interferes with other multi-threaded parts of
            # the code, which has a severe impact on performance. Please disable
            # OpenBLAS multi-threading by setting the environment variable
            # OPENBLAS_NUM_THREADS to 1.
            "OPENBLAS_NUM_THREADS=1 "
        )
