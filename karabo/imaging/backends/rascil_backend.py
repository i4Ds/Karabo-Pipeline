from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from karabo.data.casa import MSSpectralWindowTable
from karabo.imaging.image import Image
from karabo.imaging.imager_interface import Imager, ImageSpec
from karabo.imaging.imager_rascil import (
    CleanAlgorithmType,
    CleanRestoredOutputType,
    RascilDirtyImager,
    RascilDirtyImagerConfig,
    RascilImageCleaner,
    RascilImageCleanerConfig,
)
from karabo.simulation.visibility import Visibility
from karabo.util.file_handler import FileHandler


@dataclass
class RascilBackendConfig:
    combine_across_frequencies: bool = True
    clean_algorithm: CleanAlgorithmType = "hogbom"
    clean_nmajor: int = 1
    clean_niter: int = 100
    clean_gain: float = 0.1
    clean_threshold: float = 1e-4
    clean_fractional_threshold: float = 0.3
    clean_scales: Tuple[int, ...] = (0,)
    clean_nmoment: int = 4
    clean_psf_support: int = 256
    clean_restored_output: CleanRestoredOutputType = "list"


class RascilBackendImager(Imager):
    """Adapter exposing the legacy RASCIL imaging stack."""

    def __init__(self, config: Optional[RascilBackendConfig] = None) -> None:
        self.config = config or RascilBackendConfig()
        self.last_visibility: Optional[Visibility] = None
        self.last_image_spec: Optional[ImageSpec] = None
        self.last_dirty_image: Optional[Image] = None
        self.last_psf_image: Optional[Image] = None
        self.last_model_image: Optional[Image] = None
        self.last_residual_image: Optional[Image] = None

    def _config_for_spec(self, spec: ImageSpec) -> RascilDirtyImagerConfig:
        return RascilDirtyImagerConfig(
            imaging_npixel=spec.npix,
            imaging_cellsize=spec.cellsize_radians,
            combine_across_frequencies=self.config.combine_across_frequencies,
        )

    def _create_dirty_imager(self, spec: ImageSpec) -> RascilDirtyImager:
        return RascilDirtyImager(self._config_for_spec(spec))

    def _infer_visibility_nchan(self, vis: Visibility) -> int:
        num_chan = MSSpectralWindowTable.get_col(vis.path, "NUM_CHAN")
        try:
            num_chan_values = [int(value) for value in num_chan]
        except TypeError:
            num_chan_values = [int(num_chan)]
        unique_values = set(num_chan_values)
        if len(unique_values) != 1:
            raise NotImplementedError(
                "RASCIL backend restore currently supports Measurement Sets with "
                "one channel count across spectral windows."
            )
        return unique_values.pop()

    def _cleaner_config_for_spec(
        self,
        vis: Visibility,
        spec: ImageSpec,
    ) -> RascilImageCleanerConfig:
        return RascilImageCleanerConfig(
            imaging_npixel=spec.npix,
            imaging_cellsize=spec.cellsize_radians,
            ingest_vis_nchan=self._infer_visibility_nchan(vis),
            imaging_nchan=spec.nchan,
            clean_algorithm=self.config.clean_algorithm,
            clean_nmajor=self.config.clean_nmajor,
            clean_niter=self.config.clean_niter,
            clean_gain=self.config.clean_gain,
            clean_threshold=self.config.clean_threshold,
            clean_fractional_threshold=self.config.clean_fractional_threshold,
            clean_scales=list(self.config.clean_scales),
            clean_nmoment=self.config.clean_nmoment,
            clean_psf_support=self.config.clean_psf_support,
            clean_restored_output=self.config.clean_restored_output,
        )

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
        self.last_visibility = vis
        self.last_image_spec = image_spec
        self.last_dirty_image = dirty_image
        self.last_psf_image = psf_image
        return dirty_image, psf_image

    def restore(self, model_or_dirty: Image, psf: Image) -> Image:  # noqa: ARG002
        if self.last_visibility is None or self.last_image_spec is None:
            raise RuntimeError(
                "RASCIL restore requires a previous invert(...) call so the "
                "adapter can reuse the visibility."
            )

        cleaner = RascilImageCleaner(
            self._cleaner_config_for_spec(self.last_visibility, self.last_image_spec)
        )
        tmp_dir = FileHandler().get_tmp_dir(
            prefix="rascil-clean-",
            purpose="model/restored/residual fits storage",
        )
        (
            self.last_model_image,
            restored_image,
            self.last_residual_image,
        ) = cleaner.create_cleaned_image_variants(
            self.last_visibility,
            deconvolved_fits_path=os.path.join(tmp_dir, "model.fits"),
            restored_fits_path=os.path.join(tmp_dir, "restored.fits"),
            residual_fits_path=os.path.join(tmp_dir, "residual.fits"),
        )
        return restored_image
