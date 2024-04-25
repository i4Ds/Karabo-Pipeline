from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from typing import List, Union

from karabo.error import KaraboError
from karabo.imaging.image import Image
from karabo.imaging.imager_base import (
    DirtyImager,
    DirtyImagerConfig,
    ImageCleaner,
    ImageCleanerConfig,
)
from karabo.util.file_handler import FileHandler

WSCLEAN_BINARY = "wsclean"


def _get_command_prefix(tmp_dir: str) -> str:
    return (
        # wsclean always uses the current directory as the working directory
        f"cd {tmp_dir} && "
        # Avoids the following wsclean error:
        # This software was linked to a multi-threaded version of OpenBLAS.
        # OpenBLAS multi-threading interferes with other multi-threaded parts of
        # the code, which has a severe impact on performance. Please disable
        # OpenBLAS multi-threading by setting the environment variable
        # OPENBLAS_NUM_THREADS to 1.
        "OPENBLAS_NUM_THREADS=1 "
    )


class WscleanDirtyImager(DirtyImager):
    TMP_PREFIX_DIRTY = "WSClean-dirty-"
    TMP_PURPOSE_DIRTY = "Disk cache for WSClean dirty images"

    OUTPUT_FITS_DIRTY = "wsclean-dirty.fits"

    def create_dirty_image(self, config: DirtyImagerConfig) -> Image:
        # TODO combine_across_frequencies
        tmp_dir = FileHandler().get_tmp_dir(
            prefix=self.TMP_PREFIX_DIRTY,
            purpose=self.TMP_PURPOSE_DIRTY,
        )
        command = _get_command_prefix(tmp_dir) + (
            f"{WSCLEAN_BINARY} "
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


@dataclass
class WscleanImageCleanerConfig(ImageCleanerConfig):
    niter: int = 50000
    mgain: float = 0.8
    auto_threshold: int = 3

    # TODO test this
    def __post_init__(self) -> None:
        if not (self.ms_file_path is not None and self.dirty_fits_path is None):
            raise KaraboError(
                "This class starts from the measurement set, "
                "not the dirty image, when cleaning. "
                "Please pass ms_file_path and do not pass dirty_fits_path."
            )

    @classmethod
    # TODO test this
    def from_image_cleaner_config(
        cls, image_cleaner_config: ImageCleanerConfig
    ) -> WscleanImageCleanerConfig:
        return cls(
            imaging_npixel=image_cleaner_config.imaging_npixel,
            imaging_cellsize=image_cleaner_config.imaging_cellsize,
            ms_file_path=image_cleaner_config.ms_file_path,
            dirty_fits_path=image_cleaner_config.dirty_fits_path,
        )


class WscleanImageCleaner(ImageCleaner):
    TMP_PREFIX_CLEANED = "WSClean-cleaned-"
    TMP_PURPOSE_CLEANED = "Disk cache for WSClean cleaned images"
    TMP_PREFIX_CUSTOM = "WSClean-custom-"
    TMP_PURPOSE_CUSTOM = "Disk cache for WSClean custom command images"

    OUTPUT_FITS_CLEANED = "wsclean-image.fits"

    def create_cleaned_image(self, config: ImageCleanerConfig) -> Image:
        # If config is an ImageCleanerConfig (base class) instance, convert to
        # WscleanImageCleanerConfig using default values
        # for WSClean-specific configuration.
        if not isinstance(config, WscleanImageCleanerConfig):
            config = WscleanImageCleanerConfig.from_image_cleaner_config(config)
        tmp_dir = FileHandler().get_tmp_dir(
            prefix=self.TMP_PREFIX_CLEANED,
            purpose=self.TMP_PURPOSE_CLEANED,
        )
        # There is a flag -reuse-dirty <prefix> to start from an existing
        # dirty image, but I currently don't see a clean way of
        # using it with our temporary directories since wsclean
        # always uses the current directory as the working directory
        # and it doesn't seem to be possible to pass a path to a dirty
        # image, only a name prefix for a file in the working directory.
        command = _get_command_prefix(tmp_dir) + (
            f"{WSCLEAN_BINARY} "
            f"-size {config.imaging_npixel} {config.imaging_npixel} "
            f"-scale {math.degrees(config.imaging_cellsize)}deg "
            f"-niter {config.niter} "
            f"-mgain {config.mgain} "
            f"-auto-threshold {config.auto_threshold} "
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

        return Image(path=os.path.join(tmp_dir, self.OUTPUT_FITS_CLEANED))

    def create_image_custom_command(
        self,
        command: str,
        output_filenames: Union[str, List[str]] = "wsclean-image.fits",
    ) -> Union[Image, List[Image]]:
        tmp_dir = FileHandler().get_tmp_dir(
            prefix=self.TMP_PREFIX_CUSTOM,
            purpose=self.TMP_PURPOSE_CUSTOM,
        )
        expected_command_prefix = f"{WSCLEAN_BINARY} "
        if not command.startswith(expected_command_prefix):
            raise KaraboError(
                "Unexpected command. Expecting command to start with "
                f'"{expected_command_prefix}".'
            )
        command = _get_command_prefix(tmp_dir) + command
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
