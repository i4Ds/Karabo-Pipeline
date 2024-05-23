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
from typing_extensions import override

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
    """Config / parameters of a RascilDirtyImager.

    Adds parameters specific to RascilDirtyImager.

    Attributes:
        imaging_npixel (int): see DirtyImagerConfig
        imaging_cellsize (float): see DirtyImagerConfig
        combine_across_frequencies (bool): see DirtyImagerConfig
        override_cellsize (bool): Override the cellsize if it is
            above the critical cellsize. Defaults to False.
    """

    override_cellsize: bool = False

    @classmethod
    def from_dirty_imager_config(
        cls, dirty_imager_config: DirtyImagerConfig
    ) -> RascilDirtyImagerConfig:
        """Creates a RascilDirtyImagerConfig from a DirtyImagerConfig.

        Adopts basic parameters from a DirtyImagerConfig.
        Uses default values for RascilDirtyImagerConfig-specific parameters.

        Args:
            dirty_imager_config (DirtyImagerConfig): basic dirty imager config

        Returns:
            RascilDirtyImagerConfig: RascilDirtyImager-specific config
        """
        return cls(
            imaging_npixel=dirty_imager_config.imaging_npixel,
            imaging_cellsize=dirty_imager_config.imaging_cellsize,
            combine_across_frequencies=dirty_imager_config.combine_across_frequencies,
        )


class RascilDirtyImager(DirtyImager):
    """Dirty imager based on the RASCIL library.

    Attributes:
        config (RascilDirtyImagerConfig): Config containing parameters for
            RASCIL dirty imaging.
    """

    def __init__(self, config: DirtyImagerConfig) -> None:
        """Initializes the instance with a config.

        If config is a DirtyImagerConfig (base class) instance, converts it to
        a RascilDirtyImagerConfig using the
        RascilDirtyImagerConfig.from_dirty_imager_config method.

        Args:
            config (DirtyImagerConfig): see config attribute
        """

        if not isinstance(config, RascilDirtyImagerConfig):
            config = RascilDirtyImagerConfig.from_dirty_imager_config(config)
        super().__init__(config)

    @override
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

            assert image.data.ndim == 4
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
    """Config / parameters of a RascilImageCleaner.

    Adds parameters specific to RascilImageCleaner.

    Attributes:
        imaging_npixel (int): see ImageCleanerConfig
        imaging_cellsize (float): see ImageCleanerConfig
        ingest_dd (List[int]): Data descriptors in MS to read (all must have the same
            number of channels). Defaults to [0].
        ingest_vis_nchan (Optional[int]): Number of channels in a single data
            descriptor in the MS. Defaults to None.
        ingest_chan_per_vis (int): Number of channels per blockvis (before any average).
            Defaults to 1.
        imaging_nchan (int): Number of channels per image. Defaults to 1.
        imaging_w_stacking (Union[bool, str]): Use the improved w stacking method
            in Nifty Gridder?. Defaults to True.
        imaging_flat_sky (Union[bool, str]): If using a primary beam, normalise to
        flat sky? Defaults to False.
        override_cellsize (bool): Override the cellsize if it is above
            the critical cellsize? Defaults to False.
        imaging_uvmax (Optional[float]): TODO. Defaults to None.
        imaging_uvmin (float): TODO. Defaults to 0.
        imaging_dft_kernel (Optional[str]): TODO.
            DFT kernel: cpu_looped | cpu_numba | gpu_raw. Defaults to None.
        client (Optional[Client]): Dask client. Defaults to None.
        use_dask (bool): Use dask? Defaults to False.
        n_threads (int): n_threads per worker. Defaults to 1.
        use_cuda (bool): Use CUDA for Nifty Gridder? Defaults to False.
        img_context (ImageContextType): Which nifty gridder to use. Defaults to "ng".
        clean_algorithm (CleanAlgorithmType): Deconvolution algorithm
            (hogbom or msclean or mmclean). Defaults to "hogbom".
        clean_beam (Optional[Dict[CleanBeamInDegType, float]]): major axis, minor axis,
            position angle (deg). Defaults to None.
        clean_scales (List[int]): Scales for multiscale clean (pixels) e.g. [0, 6, 10].
            Defaults to [0].
        clean_nmoment (int): Number of frequency moments in mmclean
            (1=constant, 2=linear). Defaults to 4.
        clean_nmajor (int): Number of major cycles in cip or ical. Defaults to 5.
        clean_niter (int): Number of minor cycles in CLEAN. Defaults to 1000.
        clean_psf_support (int): Half-width of psf used in cleaning (pixels).
            Defaults to 256.
        clean_gain (float): Clean loop gain. Defaults to 0.1.
        clean_threshold (float): Clean stopping threshold (Jy/beam). Defaults to 1e-4.
        clean_component_threshold (Optional[float]): Sources with absolute flux
            > this level (Jy) are fit or extracted using skycomponents.
            Defaults to None.
        clean_component_method (CleanComponentMethodType): Method to convert sources
            in image to skycomponents: "fit" in frequency or "extract" actual values.
            Defaults to "fit".
        clean_fractional_threshold (float): Fractional stopping threshold for major
            cycle. Defaults to 0.3.
        clean_facets (int) Number of overlapping facets in faceted clean along each
            axis. Defaults to 1.
        clean_overlap (int): Overlap of facets in clean (pixels). Defaults to 32.
        clean_taper (CleanTaperType): Type of interpolation between facets in
            deconvolution: none or linear or tukey. Defaults to "tukey".
        clean_restore_facets (int): Number of overlapping facets in restore step
            along each axis. Defaults to 1.
        clean_restore_overlap (int): Overlap of facets in restore step (pixels).
            Defaults to 32.
        clean_restore_taper (CleanTaperType): Type of interpolation between facets in
            restore step (none, linear or tukey). Defaults to "tukey".
        clean_restored_output (CleanRestoredOutputType): Type of restored image output:
            taylor, list, or integrated. Defaults to "list".
    """

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
        """Creates a RascilImageCleanerConfig from an ImageCleanerConfig.

        Adopts basic parameters from an ImageCleanerConfig.
        Uses default values for RascilImageCleanerConfig-specific parameters.

        Args:
            image_cleaner_config (ImageCleanerConfig): basic image cleaner config

        Returns:
            RascilImageCleanerConfig: RascilImageCleaner-specific config
        """

        return cls(
            imaging_npixel=image_cleaner_config.imaging_npixel,
            imaging_cellsize=image_cleaner_config.imaging_cellsize,
        )


class RascilImageCleaner(ImageCleaner):
    """Image cleaner based on the RASCIL library.

    Attributes:
        config (RascilImageCleanerConfig): Config containing parameters for
            RASCIL image cleaning.
    """

    def __init__(self, config: ImageCleanerConfig) -> None:
        """Initializes the instance with a config.

        If config is an ImageCleanerConfig (base class) instance, converts it to
        a RascilImageCleanerConfig using the
        RascilImageCleanerConfig.from_image_cleaner_config method.

        Args:
            config (ImageCleanerConfig): see config attribute
        """

        if not isinstance(config, RascilImageCleanerConfig):
            config = RascilImageCleanerConfig.from_image_cleaner_config(config)
        super().__init__(config)

    @override
    def create_cleaned_image(
        self,
        ms_file_path: FilePathType,
        dirty_fits_path: Optional[FilePathType] = None,
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        if dirty_fits_path is not None:
            raise NotImplementedError(
                "Creating a cleaned image from an existing dirty image is not "
                "currently supported by this ImageCleaner."
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
        # TODO Improve description deconvolved vs restored
        """Creates a clean image from visibilities.

        Args:
            ms_file_path (FilePathType): Path to measurement set from which
                a clean image should be created
            deconvolved_fits_path (Optional[FilePathType], optional): Path to write the
                deconvolved image to. Example: /tmp/deconvolved.fits.
                If None, will be set to a temporary directory and a default file name.
                Defaults to None.
            restored_fits_path (Optional[FilePathType], optional): Path to write the
                restored image to. Example: /tmp/restored.fits.
                If None, will be set to a temporary directory and a default file name.
                Defaults to None.
            residual_fits_path (Optional[FilePathType], optional): Path to write the
                residual image to. Example: /tmp/residual.fits.
                If None, will be set to a temporary directory and a default file name.
                Defaults to None.

        Returns:
            Tuple[Image, Image, Image]: Tuple of deconvolved, restored, residual images
        """

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
