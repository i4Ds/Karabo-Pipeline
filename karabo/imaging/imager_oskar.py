from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import oskar
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.visibility import Visibility as RASCILVisibility
from typing_extensions import override

from karabo.imaging.image import Image
from karabo.imaging.imager_base import DirtyImager, DirtyImagerConfig
from karabo.simulation.visibility import Visibility
from karabo.util._types import FilePathType
from karabo.util.file_handler import FileHandler


# TODO Set kw_only=True after update to Python 3.10
# Right now, if one inherited superclass has a default-argument, you have to set
# defaults for all your attributes as well.
@dataclass
class OskarDirtyImagerConfig(DirtyImagerConfig):
    """Config / parameters of an OskarDirtyImager.

    Adds parameters specific to OskarDirtyImager.

    Attributes:
        imaging_npixel (int): see DirtyImagerConfig
        imaging_cellsize (float): see DirtyImagerConfig
        combine_across_frequencies (bool): see DirtyImagerConfig
        imaging_phase_centre (Optional[str]): Phase centre (in SkyCoord string format).
            Defaults to None.
    """

    imaging_phase_centre: Optional[str] = None


class OskarDirtyImager(DirtyImager):
    """Dirty imager based on the OSKAR library.

    Attributes:
        config (OskarDirtyImagerConfig): Config containing parameters for
            OSKAR dirty imaging.
    """

    def __init__(self, config: OskarDirtyImagerConfig) -> None:
        """Initializes the instance with a config.

        Args:
            config (OskarDirtyImagerConfig): see config attribute
        """
        super().__init__()
        self.config: OskarDirtyImagerConfig = config

    @override
    def create_dirty_image(
        self,
        visibility: Union[Visibility, RASCILVisibility],
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        if isinstance(visibility, RASCILVisibility):
            raise NotImplementedError(
                """OSKAR Imager applied to
                RASCIL Visibilities is currently not supported.
                For RASCIL Visibilities please use the RASCIL Imager."""
            )

        # Validate requested filepath
        if output_fits_path is None:
            tmp_dir = FileHandler().get_tmp_dir(
                prefix="Imager-Dirty-",
                purpose="disk-cache for dirty.fits",
            )
            output_fits_path = os.path.join(tmp_dir, "dirty.fits")

        if self.config.combine_across_frequencies is False:
            raise NotImplementedError(
                """For the OSKAR backend, the dirty image will
                always have intensities added across all frequency channels.
                Therefore, combine_across_frequencies==False
                is not currently supported.
                """
            )
        imager = oskar.Imager()

        # Use VIS file path by default. If it does not exist, switch to MS file path.
        # visibility should have at least one valid path by construction
        input_file = visibility.vis_path
        if not Path(input_file).exists():
            input_file = visibility.ms_file_path

        imager.set(
            input_file=input_file,
            output_root=output_fits_path,
            cellsize_arcsec=3600 * np.degrees(self.config.imaging_cellsize),
            image_size=self.config.imaging_npixel,
        )
        if self.config.imaging_phase_centre is not None:
            phase_centre = SkyCoord(self.config.imaging_phase_centre, frame="icrs")
            ra = phase_centre.ra.degree
            dec = phase_centre.dec.degree

            imager.set_vis_phase_centre(ra, dec)

        imager.run(return_images=1)

        # OSKAR adds _I.fits to the fits_path set by the user
        os.rename(f"{output_fits_path}_I.fits", output_fits_path)

        image = Image(path=output_fits_path)

        # OSKAR Imager always produces one image by
        # combining all frequency channels.
        # To maintain the same data shape when compared to other imagers (e.g. RASCIL),
        # We add an axis for frequency, and modify the header accordingly
        assert image.data.ndim == 4

        image.header["NAXIS"] = 4
        image.header["NAXIS4"] = 1

        # This card is not set correctly by OSKAR. It sets the value to 0.0 which
        # prevents the calculation of the world coordinate system later on.
        # Using the value from RASCIL imager, which sets it correctly.
        image.header.set("CDELT3", 1.0, "Coordinate increment at reference point")
        
        image.write_to_file(path=output_fits_path, overwrite=True)

        return image
