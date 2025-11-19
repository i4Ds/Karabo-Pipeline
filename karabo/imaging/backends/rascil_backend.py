from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from karabo.imaging.image import Image
from karabo.imaging.imager_interface import Imager, ImageSpec
from karabo.imaging.imager_rascil import RascilDirtyImager, RascilDirtyImagerConfig
from karabo.simulation.visibility import Visibility
from karabo.util.file_handler import FileHandler


@dataclass
class RascilBackendConfig:
    combine_across_frequencies: bool = True


class RascilBackendImager(Imager):
    """Adapter exposing the legacy RASCIL imaging stack."""

    def __init__(self, config: Optional[RascilBackendConfig] = None) -> None:
        self.config = config or RascilBackendConfig()

    def _config_for_spec(self, spec: ImageSpec) -> RascilDirtyImagerConfig:
        return RascilDirtyImagerConfig(
            imaging_npixel=spec.npix,
            imaging_cellsize=spec.cellsize_radians,
            combine_across_frequencies=self.config.combine_across_frequencies,
        )

    def _create_dirty_imager(self, spec: ImageSpec) -> RascilDirtyImager:
        return RascilDirtyImager(self._config_for_spec(spec))

    def invert(self, vis: Visibility, image_spec: ImageSpec) -> tuple[Image, Image]:
        dirty_imager = self._create_dirty_imager(image_spec)

        tmp_dir = FileHandler().get_tmp_dir(
            prefix="rascil-imager-",
            purpose="dirty and psf fits storage",
        )
        dirty_path = os.path.join(tmp_dir, "dirty.fits")
        psf_path = os.path.join(tmp_dir, "psf.fits")
        dirty_image, psf_image = dirty_imager.create_dirty_and_psf(
            vis,
            dirty_output_fits_path=dirty_path,
            psf_output_fits_path=psf_path,
        )
        return dirty_image, psf_image

    def restore(self, model_or_dirty: Image, psf: Image) -> Image:  # noqa: ARG002
        return model_or_dirty
