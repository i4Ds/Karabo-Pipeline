from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

from ska_sdp_datamodels.visibility import Visibility as RASCILVisibility

from karabo.imaging.image import Image
from karabo.simulation.visibility import Visibility
from karabo.util._types import FilePathType


@dataclass
class DirtyImagerConfig:
    # TODO was nullable before
    imaging_npixel: int
    # TODO was nullable before
    imaging_cellsize: float
    combine_across_frequencies: bool = True


class DirtyImager(ABC):
    def __init__(self, config: DirtyImagerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def create_dirty_image(
        self,
        # TODO find better solution
        # use general format throughout pipeline,
        # convert to specific format where it's necessary.
        # https://github.com/i4Ds/Karabo-Pipeline/issues/421
        visibility: Union[Visibility, RASCILVisibility],
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        ...


@dataclass
class ImageCleanerConfig:
    # TODO was nullable before
    imaging_npixel: int
    # TODO was nullable before
    imaging_cellsize: float


class ImageCleaner(ABC):
    def __init__(self, config: ImageCleanerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def create_cleaned_image(
        self,
        ms_file_path: Optional[FilePathType] = None,
        dirty_fits_path: Optional[FilePathType] = None,
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        ...
