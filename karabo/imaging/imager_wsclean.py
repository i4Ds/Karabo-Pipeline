from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from typing import Tuple

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
    def create_dirty_image(self, config: ImagerConfig) -> Image:
        # TODO combine_across_frequencies
        # If config is an ImagerConfig (base class) instance, convert to
        # WscleanImagerConfig using default values for WSClean-specific configuration.
        if not isinstance(config, WscleanImagerConfig):
            config = WscleanImagerConfig.from_imager_config(config)
        tmp_dir = FileHandler().get_tmp_dir(
            prefix="WSClean-dirty-",
            purpose="Disk cache for WSClean dirty images",
        )
        command = (
            f"cd {tmp_dir} && "
            # Avoids the following wsclean error:
            # This software was linked to a multi-threaded version of OpenBLAS.
            # OpenBLAS multi-threading interferes with other multi-threaded parts of
            # the code, which has a severe impact on performance. Please disable
            # OpenBLAS multi-threading by setting the environment variable
            # OPENBLAS_NUM_THREADS to 1.
            "OPENBLAS_NUM_THREADS=1 "
            "wsclean "
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

        return Image(path=os.path.join(tmp_dir, "wsclean-dirty.fits"))

    def create_cleaned_image(self, config: ImagerConfig) -> Tuple[Image, Image, Image]:
        # If config is an ImagerConfig (base class) instance, convert to
        # WscleanImagerConfig using default values for WSClean-specific configuration.
        if not isinstance(config, WscleanImagerConfig):
            config = WscleanImagerConfig.from_imager_config(config)
        tmp_dir = FileHandler().get_tmp_dir(
            prefix="WSClean-cleaned-",
            purpose="Disk cache for WSClean cleaned images",
        )
        command = (
            f"cd {tmp_dir} && "
            # Avoids the following wsclean error:
            # This software was linked to a multi-threaded version of OpenBLAS.
            # OpenBLAS multi-threading interferes with other multi-threaded parts of
            # the code, which has a severe impact on performance. Please disable
            # OpenBLAS multi-threading by setting the environment variable
            # OPENBLAS_NUM_THREADS to 1.
            "OPENBLAS_NUM_THREADS=1 "
            "wsclean "
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
            Image(path=os.path.join(tmp_dir, "wsclean-image.fits")),
            Image(path=os.path.join(tmp_dir, "wsclean-image.fits")),
            Image(path=os.path.join(tmp_dir, "wsclean-residual.fits")),
        )
