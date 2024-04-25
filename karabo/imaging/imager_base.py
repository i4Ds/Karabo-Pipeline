from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from operator import xor
from typing import Optional

from karabo.error import KaraboError
from karabo.imaging.image import Image
from karabo.util._types import FilePathType


@dataclass
class DirtyImagerConfig:
    # TODO was nullable before
    imaging_npixel: int
    # TODO was nullable before
    imaging_cellsize: float
    ms_file_path: FilePathType


class DirtyImager(ABC):
    @abstractmethod
    def create_dirty_image(self, config: DirtyImagerConfig) -> Image:
        ...


@dataclass
class ImageCleanerConfig:
    # TODO was nullable before
    imaging_npixel: int
    # TODO was nullable before
    imaging_cellsize: float
    ms_file_path: Optional[FilePathType] = None
    dirty_fits_path: Optional[FilePathType] = None

    # TODO test this
    def __post_init__(self) -> None:
        if not xor(self.ms_file_path is None, self.dirty_fits_path is None):
            raise KaraboError(
                "Please pass ms_file_path or dirty_fits_path, "
                "not both and not none of them."
            )


class ImageCleaner(ABC):
    @abstractmethod
    def create_cleaned_image(self, config: ImageCleanerConfig) -> Image:
        ...
