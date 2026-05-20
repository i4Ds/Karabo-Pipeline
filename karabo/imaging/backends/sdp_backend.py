from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from ska_sdp_func_python.image.deconvolution import deconvolve_cube, restore_cube
from ska_sdp_func_python.imaging import create_image_from_visibility, invert_visibility
from ska_sdp_func_python.visibility import convert_visibility_to_stokesI

try:  # pragma: no cover - optional dependency path
    from ska_sdp_func_python.visibility.ms import read_ms as _read_ms
except ModuleNotFoundError:  # pragma: no cover
    _read_ms = None

from rascil.processing_components import create_visibility_from_ms
from rascil.processing_components.image.operations import import_image_from_fits

from karabo.imaging.image import Image
from karabo.imaging.imager_interface import Imager, ImageSpec
from karabo.imaging.util import guess_beam_parameters
from karabo.simulation.visibility import Visibility
from karabo.util.file_handler import FileHandler


@dataclass
class SdpImagerConfig:
    combine_across_frequencies: bool = True
    weighting: str = "natural"
    context: str = "2d"
    # CLEAN / deconvolution
    clean_algorithm: str = "hogbom"
    clean_niter: int = 100
    clean_gain: float = 0.1
    clean_threshold: float = 0.0
    clean_preserve_facets: bool = False
    save_model_and_residual: bool = True
    clean_fractional_threshold: float | None = None
    clean_scales: tuple[int, ...] | None = None


class SdpImager(Imager):
    def __init__(self, config: SdpImagerConfig | None = None) -> None:
        self.config = config or SdpImagerConfig()

    def _load_visibility(self, vis: Visibility) -> Any:
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

    def _flag_autocorrelations(self, visibility: Any) -> Any:
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

    def _export_images(self, dirty: Any, psf: Any) -> tuple[Image, Image]:
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

    def _import_native_image(self, img: Image) -> Any:
        """Convert Karabo Image (FITS on disk) back to ska-sdp image object."""
        return import_image_from_fits(img.path)

    def _export_native_image(self, tmp_dir: str, fname: str, native_img: Any) -> Image:
        """Write ska-sdp image object to FITS and wrap as Karabo Image."""
        path = os.path.join(tmp_dir, fname)
        if os.path.exists(path):
            os.remove(path)
        native_img.image_acc.export_to_fits(fits_file=path)
        karabo_img = Image(path=path)
        if (
            self.config.combine_across_frequencies
            and karabo_img.data.ndim == 4
            and karabo_img.data.shape[0] > 1
        ):
            karabo_img.header["NAXIS4"] = 1
            karabo_img.data = np.array([np.sum(karabo_img.data, axis=0)])
            karabo_img.write_to_file(path, overwrite=True)
        return karabo_img

    def restore(self, model_or_dirty: Image, psf: Image) -> Image:
        """
        Perform CLEAN-style deconvolution + restore using ska-sdp-func-python.

        Returns a restored image; model and residual are written to disk as a side
        effect for inspection (paths stored on instance attributes).
        """
        tmp_dir = FileHandler().get_tmp_dir(
            prefix="sdp-clean-", purpose="model/residual/restored storage"
        )

        dirty_native = self._import_native_image(model_or_dirty)
        psf_native = self._import_native_image(psf)

        # Normalise PSF peak to 1.0 for SDP CLEAN expectations
        # and scale dirty accordingly
        psf_peak = float(np.nanmax(psf_native["pixels"].data))
        if psf_peak == 0.0:
            raise ValueError("PSF peak is zero; cannot deconvolve.")
        dirty_native = dirty_native.copy(deep=True)
        psf_native = psf_native.copy(deep=True)
        dirty_native["pixels"].data /= psf_peak
        psf_native["pixels"].data /= psf_peak

        comp_native, residual_native = deconvolve_cube(
            dirty_native,
            psf_native,
            algorithm=self.config.clean_algorithm,
            niter=self.config.clean_niter,
            gain=self.config.clean_gain,
            threshold=self.config.clean_threshold,
            fractional_threshold=(
                self.config.clean_fractional_threshold
                if self.config.clean_fractional_threshold is not None
                else 0.01
            ),
            scales=(
                list(self.config.clean_scales)
                if self.config.clean_scales is not None
                else None
            ),
        )

        # Guess a clean beam from the PSF (arcsec → degrees for SDP restore)
        beam_arcsec = guess_beam_parameters(psf)
        clean_beam = {
            "bmaj": beam_arcsec["bmaj"] / 3600.0,
            "bmin": beam_arcsec["bmin"] / 3600.0,
            "bpa": beam_arcsec["bpa"],
        }

        restored_native = restore_cube(
            comp_native,
            psf=psf_native,
            residual=residual_native,
            clean_beam=clean_beam,
        )

        # Restore original flux scaling
        comp_native["pixels"].data *= psf_peak
        residual_native["pixels"].data *= psf_peak
        restored_native["pixels"].data *= psf_peak

        self.last_model_image = self._export_native_image(
            tmp_dir, "model.fits", comp_native
        )
        self.last_residual_image = self._export_native_image(
            tmp_dir, "residual.fits", residual_native
        )
        restored_image = self._export_native_image(
            tmp_dir, "restored.fits", restored_native
        )

        return restored_image
