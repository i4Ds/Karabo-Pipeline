from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from karabo.imaging.image import Image


@dataclass
class ImagerConfig:
    ms_file_path: str
    # TODO was nullable before
    imaging_npixel: int
    # TODO was nullable before
    imaging_cellsize: float


class Imager(ABC):
    @abstractmethod
    def create_dirty_image(self, config: ImagerConfig) -> Image:
        ...

    @abstractmethod
    def create_cleaned_image(self, config: ImagerConfig) -> Tuple[Image, Image, Image]:
        ...
