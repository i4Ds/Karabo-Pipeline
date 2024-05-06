from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from distributed import Client
from rascil.processing_components import create_visibility_from_ms
from rascil.workflows import (
    continuum_imaging_skymodel_list_rsexecute_workflow,
    create_visibility_from_ms_rsexecute,
)
from rascil.workflows.rsexecute.execution_support import rsexecute
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility import Visibility as RASCILVisibility
from ska_sdp_func_python.image import image_gather_channels
from ska_sdp_func_python.imaging import (
    create_image_from_visibility,
    invert_visibility,
    remove_sumwt,
)
from ska_sdp_func_python.visibility import convert_visibility_to_stokesI

from karabo.error import KaraboError
from karabo.imaging.image import Image
from karabo.imaging.imager_base import (
    DirtyImager,
    DirtyImagerConfig,
    ImageCleaner,
    ImageCleanerConfig,
)
from karabo.simulation.visibility import Visibility
from karabo.util._types import FilePathType
from karabo.util.dask import DaskHandler
from karabo.util.file_handler import FileHandler, assert_valid_ending


@dataclass
class RascilDirtyImagerConfig(DirtyImagerConfig):
    override_cellsize: bool = False

    @classmethod
    def from_dirty_imager_config(
        cls, dirty_imager_config: DirtyImagerConfig
    ) -> RascilDirtyImagerConfig:
        return cls(
            imaging_npixel=dirty_imager_config.imaging_npixel,
            imaging_cellsize=dirty_imager_config.imaging_cellsize,
            combine_across_frequencies=dirty_imager_config.combine_across_frequencies,
        )


class RascilDirtyImager(DirtyImager):
    def __init__(self, config: DirtyImagerConfig) -> None:
        # If config is a DirtyImagerConfig (base class) instance, convert to
        # RascilDirtyImagerConfig using default values
        # for RASCIL-specific configuration.
        if not isinstance(config, RascilDirtyImagerConfig):
            config = RascilDirtyImagerConfig.from_dirty_imager_config(config)
        super().__init__(config)

    def create_dirty_image(
        self,
        visibility: Union[Visibility, RASCILVisibility],
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        config: RascilDirtyImagerConfig = cast(RascilDirtyImagerConfig, self.config)

        # Validate requested filepath
        if output_fits_path is None:
            tmp_dir = FileHandler().get_tmp_dir(
                prefix="Imager-Dirty-",
                purpose="disk-cache for dirty.fits",
            )
            output_fits_path = os.path.join(tmp_dir, "dirty.fits")

        if isinstance(visibility, Visibility):
            # Convert OSKAR Visibility to RASCIL-compatible format
            block_visibilities = create_visibility_from_ms(str(visibility.ms_file_path))

            if len(block_visibilities) != 1:
                raise ValueError("Visibilities are too large")
            visibility = block_visibilities[0]

        # Compute dirty image from visibilities
        model = create_image_from_visibility(
            visibility,
            npixel=config.imaging_npixel,
            cellsize=config.imaging_cellsize,
            override_cellsize=config.override_cellsize,
        )
        dirty, _ = invert_visibility(visibility, model, context="2d")
        if os.path.exists(output_fits_path):
            os.remove(output_fits_path)
        dirty.image_acc.export_to_fits(fits_file=output_fits_path)

        image = Image(path=output_fits_path)

        # By default, RASCIL Imager produces a 4D Image object, with shape
        # corresponding to (frequency channels, polarisations, pixels_x, pixels_y).
        # If requested, we combine images across all frequency channels into one image,
        # and modify the header information accordingly
        if config.combine_across_frequencies is True:
            image.header["NAXIS4"] = 1

            assert len(image.data.shape) == 4
            image.data = np.array([np.sum(image.data, axis=0)])

            image.write_to_file(path=output_fits_path, overwrite=True)

        return image


ImageContextType = Literal["awprojection", "2d", "ng", "wg"]
CleanAlgorithmType = Literal["hogbom", "msclean", "mmclean"]
CleanBeamInDegType = Literal["bmaj", "bmin", "bpa"]
CleanComponentMethodType = Literal["fit", "extract"]
CleanTaperType = Literal["none", "linear", "tukey"]
CleanRestoredOutputType = Literal["taylor", "list", "integrated"]


def _create_clean_scales_default_value() -> List[int]:
    return [0]


def _create_ingest_dd_default_value() -> List[int]:
    return [0]


@dataclass
class RascilImageCleanerConfig(ImageCleanerConfig):
    ingest_dd: List[int] = field(default_factory=_create_ingest_dd_default_value)
    ingest_vis_nchan: Optional[int] = None
    ingest_chan_per_vis: int = 1
    imaging_nchan: int = 1
    imaging_w_stacking: Union[bool, str] = True
    imaging_flat_sky: Union[bool, str] = False
    override_cellsize: bool = False
    imaging_uvmax: Optional[float] = None
    imaging_uvmin: float = 0
    imaging_dft_kernel: Optional[
        str
    ] = None  # DFT kernel: cpu_looped | cpu_numba | gpu_raw
    client: Optional[Client] = None
    use_dask: bool = False
    n_threads: int = 1
    use_cuda: bool = False
    img_context: ImageContextType = "ng"
    clean_algorithm: CleanAlgorithmType = "hogbom"
    clean_beam: Optional[Dict[CleanBeamInDegType, float]] = None
    clean_scales: List[int] = field(default_factory=_create_clean_scales_default_value)
    clean_nmoment: int = 4
    clean_nmajor: int = 5
    clean_niter: int = 1000
    clean_psf_support: int = 256
    clean_gain: float = 0.1
    clean_threshold: float = 1e-4
    clean_component_threshold: Optional[float] = None
    clean_component_method: CleanComponentMethodType = "fit"
    clean_fractional_threshold: float = 0.3
    clean_facets: int = 1
    clean_overlap: int = 32
    clean_taper: CleanTaperType = "tukey"
    clean_restore_facets: int = 1
    clean_restore_overlap: int = 32
    clean_restore_taper: CleanTaperType = "tukey"
    clean_restored_output: CleanRestoredOutputType = "list"

    @classmethod
    def from_image_cleaner_config(
        cls, image_cleaner_config: ImageCleanerConfig
    ) -> RascilImageCleanerConfig:
        return cls(
            imaging_npixel=image_cleaner_config.imaging_npixel,
            imaging_cellsize=image_cleaner_config.imaging_cellsize,
        )


class RascilImageCleaner(ImageCleaner):
    def __init__(self, config: ImageCleanerConfig) -> None:
        # If config is an ImageCleanerConfig (base class) instance, convert to
        # RascilImageCleanerConfig using default values
        # for RASCIL-specific configuration.
        if not isinstance(config, RascilImageCleanerConfig):
            config = RascilImageCleanerConfig.from_image_cleaner_config(config)
        super().__init__(config)

    def create_cleaned_image(
        self,
        ms_file_path: Optional[FilePathType] = None,
        dirty_fits_path: Optional[FilePathType] = None,
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        if not (ms_file_path is not None and dirty_fits_path is None):
            raise KaraboError(
                "This class starts from the measurement set, "
                "not the dirty image, when cleaning. "
                "Please pass ms_file_path and do not pass dirty_fits_path."
            )

        config: RascilImageCleanerConfig = cast(RascilImageCleanerConfig, self.config)

        _, restored, _ = self._compute(ms_file_path, config)

        if output_fits_path is not None:
            assert_valid_ending(path=output_fits_path, ending=".fits")
        else:
            tmp_dir = FileHandler().get_tmp_dir(
                prefix="Imaging-Rascil-",
                purpose="disk-cache for non-specified .fits files.",
            )
            output_fits_path = os.path.join(tmp_dir, "restored.fits")

        if isinstance(restored, list):
            restored = image_gather_channels(restored)
        if os.path.exists(output_fits_path):
            os.remove(output_fits_path)
        restored.image_acc.export_to_fits(fits_file=str(output_fits_path))
        restored_image = Image(path=output_fits_path)

        return restored_image

    def create_cleaned_image_variants(
        self,
        ms_file_path: FilePathType,
        deconvolved_fits_path: Optional[FilePathType] = None,
        restored_fits_path: Optional[FilePathType] = None,
        residual_fits_path: Optional[FilePathType] = None,
    ) -> Tuple[Image, Image, Image]:
        config: RascilImageCleanerConfig = cast(RascilImageCleanerConfig, self.config)

        residual, restored, skymodel = self._compute(ms_file_path, config)

        deconvolved_fits_path = deconvolved_fits_path
        restored_fits_path = restored_fits_path
        residual_fits_path = residual_fits_path
        if deconvolved_fits_path is not None:
            assert_valid_ending(path=deconvolved_fits_path, ending=".fits")
        if restored_fits_path is not None:
            assert_valid_ending(path=restored_fits_path, ending=".fits")
        if residual_fits_path is not None:
            assert_valid_ending(path=residual_fits_path, ending=".fits")
        if (
            deconvolved_fits_path is None
            or restored_fits_path is None
            or residual_fits_path is None
        ):
            tmp_dir = FileHandler().get_tmp_dir(
                prefix="Imaging-Rascil-",
                purpose="disk-cache for non-specified .fits files.",
            )
            if deconvolved_fits_path is None:
                deconvolved_fits_path = os.path.join(tmp_dir, "deconvolved.fits")
            if restored_fits_path is None:
                restored_fits_path = os.path.join(tmp_dir, "restored.fits")
            if residual_fits_path is None:
                residual_fits_path = os.path.join(tmp_dir, "residual.fits")

        deconvolved = [sm.image for sm in skymodel]
        deconvolved_image_rascil = image_gather_channels(deconvolved)

        if isinstance(restored, list):
            restored = image_gather_channels(restored)

        residual = remove_sumwt(residual)
        if isinstance(residual, list):
            residual = image_gather_channels(residual)

        if os.path.exists(deconvolved_fits_path):
            os.remove(deconvolved_fits_path)
        deconvolved_image_rascil.image_acc.export_to_fits(
            fits_file=str(deconvolved_fits_path)
        )
        if os.path.exists(restored_fits_path):
            os.remove(restored_fits_path)
        restored.image_acc.export_to_fits(fits_file=str(restored_fits_path))
        if os.path.exists(residual_fits_path):
            os.remove(residual_fits_path)
        residual.image_acc.export_to_fits(fits_file=str(residual_fits_path))

        deconvolved_image = Image(path=deconvolved_fits_path)
        restored_image = Image(path=restored_fits_path)
        residual_image = Image(path=residual_fits_path)

        return deconvolved_image, restored_image, residual_image

    def _compute(
        self,
        ms_file_path: FilePathType,
        config: RascilImageCleanerConfig,
    ) -> Any:
        if config.client and not config.use_dask:
            raise RuntimeError("Client passed but use_dask is False")
        if config.use_dask:
            if not config.client:
                client = DaskHandler.get_dask_client()
            print(client.cluster)
            rsexecute.set_client(client=client, use_dask=config.use_dask, use_dlg=False)
        else:  # set use_dask through `set_client` to False,
            # because it's the only way to disable dask for `rsexecute` singleton
            rsexecute.set_client(client=None, use_dask=False, use_dlg=False)
        # Set CUDA parameters
        img_context = config.img_context
        if config.use_cuda:
            if img_context != "wg":
                print(
                    f"Changing img_context` from '{img_context}' "
                    + f"to 'wg' because {config.use_cuda=}"
                )
            img_context = "wg"

        if config.ingest_vis_nchan is None:
            raise ValueError("`self.ingest_vis_nchan` is None but must set.")

        blockviss = create_visibility_from_ms_rsexecute(
            msname=str(ms_file_path),
            nchan_per_vis=config.ingest_chan_per_vis,
            nout=config.ingest_vis_nchan // config.ingest_chan_per_vis,
            dds=config.ingest_dd,
            average_channels=True,
        )

        blockviss = [
            rsexecute.execute(convert_visibility_to_stokesI)(bv) for bv in blockviss
        ]

        models = [
            rsexecute.execute(create_image_from_visibility)(
                bvis,
                npixel=config.imaging_npixel,
                nchan=config.imaging_nchan,
                cellsize=config.imaging_cellsize,
                override_cellsize=config.override_cellsize,
                polarisation_frame=PolarisationFrame("stokesI"),
            )
            for bvis in blockviss
        ]

        result = continuum_imaging_skymodel_list_rsexecute_workflow(
            vis_list=blockviss,
            model_imagelist=models,
            context=img_context,
            threads=config.n_threads,
            wstacking=config.imaging_w_stacking == "True",
            niter=config.clean_niter,
            nmajor=config.clean_nmajor,
            algorithm=config.clean_algorithm,
            gain=config.clean_gain,
            scales=config.clean_scales,
            fractional_threshold=config.clean_fractional_threshold,
            threshold=config.clean_threshold,
            nmoment=config.clean_nmoment,
            psf_support=config.clean_psf_support,
            restored_output=config.clean_restored_output,
            deconvolve_facets=config.clean_facets,
            deconvolve_overlap=config.clean_overlap,
            deconvolve_taper=config.clean_taper,
            restore_facets=config.clean_restore_facets,
            restore_overlap=config.clean_restore_overlap,
            restore_taper=config.clean_restore_taper,
            dft_compute_kernel=config.imaging_dft_kernel,
            component_threshold=config.clean_component_threshold,
            component_method=config.clean_component_method,
            flat_sky=config.imaging_flat_sky,
            clean_beam=config.clean_beam,
            clean_algorithm=config.clean_algorithm,
            imaging_uvmax=config.imaging_uvmax,
            imaging_uvmin=config.imaging_uvmin,
        )

        result = rsexecute.compute(result, sync=True)

        return result
