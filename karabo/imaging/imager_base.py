"""Base classes for dirty imagers and image cleaners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from karabo.imaging.image import Image
from karabo.simulation.visibility import Visibility
from karabo.util._types import FilePathType


# TODO Set kw_only=True after update to Python 3.10
# Right now, if one inherited superclass has a default-argument, you have to set
# defaults for all your attributes as well.
@dataclass
class DirtyImagerConfig:
    """Base class for the config / parameters of a dirty imager.

    Contains basic parameters common across all dirty imagers.
    Inherit and add parameters specific to a dirty imager implementation.

    Attributes:
        imaging_npixel (int): Image size
        imaging_cellsize (float): Scale of a pixel in radians
        combine_across_frequencies (bool): Whether or not to combine images
            across all frequency channels into one image. Defaults to True.

    """

    imaging_npixel: int
    imaging_cellsize: float
    combine_across_frequencies: bool = True


class DirtyImager(ABC):
    """Abstract base class for a dirty imager.

    A dirty imager creates dirty images from visibilities.

    """

    config: DirtyImagerConfig

    @abstractmethod
    def create_dirty_image(
        self,
        visibility: Visibility,
        /,
        *,
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        """Creates a dirty image from a visibility.

        Args:
            visibility: Visibility object
                from which to create the dirty image. Contains the visibilities
                of an observation.
            output_fits_path: Path to write the dirty image to.
                Example: /tmp/dirty.fits.
                If None, will be set to a temporary directory and a default file name.
                Defaults to None.

        Returns:
            Image: Dirty image

        """

        ...


# TODO Set kw_only=True after update to Python 3.10
# Right now, if one inherited superclass has a default-argument, you have to set
# defaults for all your attributes as well.
@dataclass
class ImageCleanerConfig:
    """Base class for the config / parameters of an image cleaner.

    Contains basic parameters common across all image cleaners.
    Inherit and add parameters specific to an image cleaner implementation.

    Attributes:
        imaging_npixel (int): Image size
        imaging_cellsize (float): Scale of a pixel in radians

    """

    imaging_npixel: int
    imaging_cellsize: float


class ImageCleaner(ABC):
    """Abstract base class for an image cleaner.

    An image cleaner creates clean images from dirty images
    or directly from visibilities, in that case including the
    dirty imaging process.

    """

    @abstractmethod
    def create_cleaned_image(
        self,
        visibility: Visibility,
        /,
        *,
        dirty_fits_path: Optional[FilePathType] = None,
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        """Creates a clean image from a dirty image or from visibilities.

        Args:
            visibility: Visibility from which a clean image should be created.
            dirty_fits_path: Path to dirty image FITS file that
                should be reused to create a clean image. If None, dirty image will be
                created first from the visibilities. Defaults to None.
            output_fits_path: Path to write the clean image to.
                Example: /tmp/restored.fits.
                If None, will be set to a temporary directory and a default file name.
                Defaults to None.

        Returns:
            Image: Clean image

        """

        ...
