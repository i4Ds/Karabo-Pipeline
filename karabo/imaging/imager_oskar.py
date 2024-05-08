from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, cast

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


@dataclass
class OskarDirtyImagerConfig(DirtyImagerConfig):
    imaging_phasecentre: Optional[str] = None

    @classmethod
    def from_dirty_imager_config(
        cls, dirty_imager_config: DirtyImagerConfig
    ) -> OskarDirtyImagerConfig:
        return cls(
            imaging_npixel=dirty_imager_config.imaging_npixel,
            imaging_cellsize=dirty_imager_config.imaging_cellsize,
            combine_across_frequencies=dirty_imager_config.combine_across_frequencies,
        )


class OskarDirtyImager(DirtyImager):
    def __init__(self, config: DirtyImagerConfig) -> None:
        # If config is a DirtyImagerConfig (base class) instance, convert to
        # OskarDirtyImagerConfig using default values
        # for OSKAR-specific configuration.
        if not isinstance(config, OskarDirtyImagerConfig):
            config = OskarDirtyImagerConfig.from_dirty_imager_config(config)
        super().__init__(config)

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

        config: OskarDirtyImagerConfig = cast(OskarDirtyImagerConfig, self.config)

        # Validate requested filepath
        if output_fits_path is None:
            tmp_dir = FileHandler().get_tmp_dir(
                prefix="Imager-Dirty-",
                purpose="disk-cache for dirty.fits",
            )
            output_fits_path = os.path.join(tmp_dir, "dirty.fits")

        if config.combine_across_frequencies is False:
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
            cellsize_arcsec=3600 * np.degrees(config.imaging_cellsize),
            image_size=config.imaging_npixel,
        )
        if config.imaging_phasecentre is not None:
            phase_centre = SkyCoord(config.imaging_phasecentre, frame="icrs")
            ra = phase_centre.ra.degree
            dec = phase_centre.dec.degree

            imager.set_vis_phase_centre(ra, dec)

        imager.run(return_images=1)

        # OSKAR adds _I.fits to the fits_path set by the user
        image = Image(path=f"{output_fits_path}_I.fits")

        # OSKAR Imager always produces one image by
        # combining all frequency channels.
        # To maintain the same data shape when compared to other imagers (e.g. RASCIL),
        # We add an axis for frequency, and modify the header accordingly
        assert image.data.ndim == 4

        image.header["NAXIS"] = 4
        image.header["NAXIS4"] = 1

        image.write_to_file(path=f"{output_fits_path}_I.fits", overwrite=True)

        return image
