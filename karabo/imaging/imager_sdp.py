from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import xarray as xr
from ska_sdp_func_python.imaging import create_image_from_visibility, invert_visibility
from ska_sdp_func_python.visibility import convert_visibility_to_stokesI

try:  # pragma: no cover - optional dependency path
    from ska_sdp_func_python.visibility.ms import read_ms as _read_ms
except ModuleNotFoundError:  # pragma: no cover
    _read_ms = None

from rascil.processing_components import create_visibility_from_ms

from karabo.imaging.image import Image
from karabo.imaging.imager_interface import Imager, ImageSpec
from karabo.simulation.visibility import Visibility
from karabo.util.file_handler import FileHandler


@dataclass
class SdpImagerConfig:
    combine_across_frequencies: bool = True
    weighting: str = "natural"
    context: str = "2d"


class SdpImager(Imager):
    def __init__(self, config: SdpImagerConfig | None = None) -> None:
        self.config = config or SdpImagerConfig()

    def _load_visibility(self, vis: Visibility):
        if vis.format != "MS":
            raise NotImplementedError(
                "SDP imager currently supports Measurement Set inputs only."
            )
        if _read_ms is not None:
            block = _read_ms(str(vis.path))
            if isinstance(block, list):
                if len(block) != 1:
                    raise NotImplementedError(
                        "SdpImager currently supports a single visibility per call."
                    )
                block = block[0]
        else:
            block_visibilities = create_visibility_from_ms(str(vis.path))
            if len(block_visibilities) != 1:
                raise NotImplementedError(
                    "SdpImager currently supports a single visibility per call."
                )
            block = block_visibilities[0]
        block = convert_visibility_to_stokesI(block)
        block = self._flag_autocorrelations(block)
        return block

    def _flag_autocorrelations(self, visibility):  # type: ignore[no-untyped-def]
        autocorr_mask = visibility.antenna1 == visibility.antenna2
        mask_da = xr.DataArray(autocorr_mask, dims=["baselines"])
        expanded_mask = mask_da.broadcast_like(visibility.flags)
        visibility["flags"] = visibility.flags.where(~expanded_mask, other=1)
        return visibility

    def invert(self, vis: Visibility, image_spec: ImageSpec) -> tuple[Image, Image]:
        sdp_visibility = self._load_visibility(vis)

        model = create_image_from_visibility(
            sdp_visibility,
            npixel=image_spec.npix,
            cellsize=image_spec.cellsize_radians,
            override_cellsize=False,
        )

        dirty, _ = invert_visibility(
            sdp_visibility,
            model,
            context=self.config.context,
            dopsf=False,
            weighting=self.config.weighting,
        )
        psf, _ = invert_visibility(
            sdp_visibility,
            model,
            context=self.config.context,
            dopsf=True,
            weighting=self.config.weighting,
        )

        return self._export_images(dirty, psf)

    # type: ignore[no-untyped-def]
    def _export_images(self, dirty, psf) -> tuple[Image, Image]:
        tmp_dir = FileHandler().get_tmp_dir(
            prefix="sdp-imager-", purpose="dirty and psf storage"
        )
        dirty_path = os.path.join(tmp_dir, "dirty.fits")
        psf_path = os.path.join(tmp_dir, "psf.fits")

        for path, image in ((dirty_path, dirty), (psf_path, psf)):
            if os.path.exists(path):
                os.remove(path)
            image.image_acc.export_to_fits(fits_file=path)

        dirty_image = Image(path=dirty_path)
        psf_image = Image(path=psf_path)

        if self.config.combine_across_frequencies:
            for img, path in ((dirty_image, dirty_path), (psf_image, psf_path)):
                if img.data.ndim == 4 and img.data.shape[0] > 1:
                    img.header["NAXIS4"] = 1
                    img.data = np.array([np.sum(img.data, axis=0)])
                    img.write_to_file(path, overwrite=True)

        return dirty_image, psf_image

    def restore(self, model_or_dirty: Image, psf: Image) -> Image:  # noqa: ARG002
        """Return the dirty image until CLEAN-style restore is available."""
        return model_or_dirty
