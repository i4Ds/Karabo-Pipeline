from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, cast

from karabo.imaging.image import Image
from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_interface import Imager, ImageSpec
from karabo.imaging.imager_wsclean import (
    WscleanDirtyImager,
    WscleanImageCleaner,
    WscleanImageCleanerConfig,
    create_image_custom_command,
)
from karabo.simulation.visibility import Visibility


@dataclass
class WscleanBackendConfig:
    combine_across_frequencies: bool = True
    clean_niter: int = 100
    clean_mgain: float = 0.8
    clean_auto_threshold: int = 3


class WscleanBackendImager(Imager):
    """Adapter exposing WSClean through the backend-agnostic imager API."""

    def __init__(self, config: Optional[WscleanBackendConfig] = None) -> None:
        self.config = config or WscleanBackendConfig()
        self.last_visibility: Optional[Visibility] = None
        self.last_image_spec: Optional[ImageSpec] = None
        self.last_dirty_image: Optional[Image] = None
        self.last_psf_image: Optional[Image] = None

    def _dirty_config_for_spec(self, spec: ImageSpec) -> DirtyImagerConfig:
        return DirtyImagerConfig(
            imaging_npixel=spec.npix,
            imaging_cellsize=spec.cellsize_radians,
            combine_across_frequencies=self.config.combine_across_frequencies,
        )

    def _cleaner_config_for_spec(self, spec: ImageSpec) -> WscleanImageCleanerConfig:
        return WscleanImageCleanerConfig(
            imaging_npixel=spec.npix,
            imaging_cellsize=spec.cellsize_radians,
            niter=self.config.clean_niter,
            mgain=self.config.clean_mgain,
            auto_threshold=self.config.clean_auto_threshold,
        )

    def _create_psf_image(self, vis: Visibility, spec: ImageSpec) -> Image:
        psf_image = create_image_custom_command(
            "wsclean "
            f"-size {spec.npix} {spec.npix} "
            f"-scale {math.degrees(spec.cellsize_radians)}deg "
            "-make-psf "
            f"{vis.path}",
            "wsclean-psf.fits",
        )
        return cast(Image, psf_image)

    def invert(self, vis: Visibility, image_spec: ImageSpec) -> tuple[Image, Image]:
        dirty_imager = WscleanDirtyImager(self._dirty_config_for_spec(image_spec))
        dirty_image = dirty_imager.create_dirty_image(vis)
        psf_image = self._create_psf_image(vis, image_spec)

        self.last_visibility = vis
        self.last_image_spec = image_spec
        self.last_dirty_image = dirty_image
        self.last_psf_image = psf_image

        return dirty_image, psf_image

    def restore(self, model_or_dirty: Image, psf: Image) -> Image:  # noqa: ARG002
        if self.last_visibility is None or self.last_image_spec is None:
            raise RuntimeError(
                "WSClean restore requires a previous invert(...) call so the "
                "adapter can reuse the visibility."
            )

        cleaner = WscleanImageCleaner(
            self._cleaner_config_for_spec(self.last_image_spec)
        )
        return cleaner.create_cleaned_image(
            self.last_visibility,
            dirty_fits_path=model_or_dirty.path,
        )
