"""Base classes for dirty imagers and image cleaners."""

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

    Attributes:
        config (DirtyImagerConfig): Config containing parameters for
            dirty imaging. May be a DirtyImagerConfig object
            or an object of a class derived from DirtyImagerConfig
            if the dirty imager implementation needs additional
            parameters.
    """

    def __init__(self, config: DirtyImagerConfig) -> None:
        """Initializes the instance with a config.

        Args:
            config (DirtyImagerConfig): see config attribute
        """

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
        """Creates a dirty image from a visibility.

        Args:
            visibility (Union[Visibility, RASCILVisibility]): Visibility object
                from which to create the dirty image. Contains the visibilities
                of an observation.
            output_fits_path (Optional[FilePathType]): Path to write the dirty image to.
                Example: /tmp/dirty.fits.
                If None, will be set to a temporary directory and a default file name.
                Defaults to None.

        Returns:
            Image: Dirty image
        """

        ...


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

    Attributes:
        config (ImageCleanerConfig): Config containing parameters for
            image cleaning. May be an ImageCleanerConfig object
            or an object of a class derived from ImageCleanerConfig
            if the image cleaner implementation needs additional
            parameters.
    """

    def __init__(self, config: ImageCleanerConfig) -> None:
        """Initializes the instance with a config.

        Args:
            config (ImageCleanerConfig): see config attribute
        """

        super().__init__()
        self.config = config

    @abstractmethod
    def create_cleaned_image(
        self,
        ms_file_path: Optional[FilePathType] = None,
        dirty_fits_path: Optional[FilePathType] = None,
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        """Creates a clean image from a dirty image or from visibilities.

        Args:
            ms_file_path (Optional[FilePathType]): Path to measurement set from which
                a clean image should be created. Provide either ms_file_path or
                dirty_fits_path, not both, not none. Defaults to None.
            dirty_fits_path (Optional[FilePathType]): Path to dirty image FITS file from
                which to create a clean image. Provide either ms_file_path or
                dirty_fits_path, not both, not none. Defaults to None.
            output_fits_path (Optional[FilePathType]): Path to write the clean image to.
                Example: /tmp/restored.fits.
                If None, will be set to a temporary directory and a default file name.
                Defaults to None.

        Returns:
            Image: Clean image
        """

        ...