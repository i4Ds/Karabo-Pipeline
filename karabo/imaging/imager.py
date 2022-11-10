from typing import List, Tuple, Union, Dict

from distributed import Client
from rascil.data_models import PolarisationFrame
from rascil.processing_components import (
    create_blockvisibility_from_ms,
    create_image_from_visibility,
    invert_blockvisibility,
    export_image_to_fits,
    image_gather_channels,
    remove_sumwt,
    convert_blockvisibility_to_stokesI,
)
from rascil.workflows import (
    create_blockvisibility_from_ms_rsexecute,
    continuum_imaging_skymodel_list_rsexecute_workflow,
)
from rascil.workflows.rsexecute.execution_support import rsexecute

from karabo.util.dask import get_global_client
from karabo.imaging.image import Image
from karabo.simulation.visibility import Visibility


class Imager:
    """Imager class provides imaging functionality using the visibilities of an observation with the help of RASCIL.
    In addition, it provides the calculation of the pixel coordinates of point sources.
    
    Parameters
    ----------
    visibility : Visibility, required
        Visibility object containing the visibilities of an observation.
    logfile : str, default=None,
        Name of logfile (default is to construct one from msname)
    performance_file : str, default=None
        Name of json file to contain performance information
    ingest_dd : List[int], default=[0],
        Data descriptors in MS to read (all must have the same number of channels)
    ingest_vis_nchan : int, default=None,
        Number of channels in a single data descriptor in the MS
    ingest_chan_per_blockvis : int, defualt=1,
        Number of channels per blockvis (before any average)
    ingest_average_blockvis : Union[bool, str], default=False,
        Average all channels in blockvis. 
    imaging_phasecentre : str, default=None
        Phase centre (in SkyCoord string format)
    imaging_pol : str, default="stokesI"
        RASCIL polarisation frame for image
    imaging_nchan : int, default=1,
        Number of channels per image
    imaging_ng_threads : int, default=4,
        Number of Nifty Gridder threads to use (4 is a good choice)
    imaging_w_stacking : Union[bool, str], default=True
        Use the improved w stacking method in Nifty Gridder?
    imaging_flat_sky : Union[bool, str], default=False
        If using a primary beam, normalise to flat sky?
    imaging_npixel : int, default=None
        Number of pixels in ra, dec: Should be a composite of 2, 3, 5
    imaging_cellsize : float, default=None
        Cellsize (radians). Default is to calculate
    imaging_weighting : str, default="uniform"
        Type of weighting uniform or robust or natural
    imaging_robustness : float, default=0.0
        Robustness for robust weighting
    imaging_gaussian_taper : float, default=None
        Size of Gaussian smoothing, implemented as taper in weights (rad)
    imaging_dopsf : Union[bool, str], default=False
        Make the PSF instead of the dirty image?
    imaging_dft_kernel : str, default=None
        DFT kernel: cpu_looped | cpu_numba | gpu_raw
    Attributes
    ----------
    TODO: Add attributes
    -----
    See [Rascil](https://gitlab.com/ska-telescope/external/rascil) for more information.
    Examples
    --------
    >>> imager = Imager()
    >>> convolved, restored, residual = imager.imaging_rascil()
    ...
    >>> SourceDetectionResult.detect_sources_in_image(restored)
    """
    def __init__(
        self,
        visibility: Visibility,
        logfile: str = None,
        performance_file: str = None, 
        ingest_dd: List[int] = [0],
        ingest_vis_nchan: int = None,
        ingest_chan_per_blockvis: int = 1,
        ingest_average_blockvis: Union[
            bool, str
        ] = False,
        imaging_phasecentre: str = None,
        imaging_pol: str = "stokesI",
        imaging_nchan: int = 1,
        imaging_context: str = "ng",
        imaging_ng_threads: int = 4,
        imaging_w_stacking: Union[bool, str] = True, 
        imaging_flat_sky: Union[bool, str] = False, 
        imaging_npixel: int = None,
        imaging_cellsize: float = None,
        imaging_weighting: str = "uniform",
        imaging_robustness: float = 0.0,
        imaging_gaussian_taper: float = None,
        imaging_dopsf: Union[bool, str] = False,
        imaging_dft_kernel: str = None,  # DFT kernel: cpu_looped | cpu_numba | gpu_raw
    ):
        self.logfile: str = logfile
        self.performance_file: str = performance_file
        self.visibility: Visibility = visibility
        self.ingest_dd: List[int] = ingest_dd
        self.ingest_vis_nchan: int = ingest_vis_nchan
        self.ingest_chan_per_blockvis: int = ingest_chan_per_blockvis
        self.ingest_average_blockvis: Union[bool, str] = ingest_average_blockvis
        self.imaging_phasecentre: str = imaging_phasecentre
        self.imaging_pol: str = imaging_pol
        self.imaging_nchan: int = imaging_nchan
        self.imaging_context: str = imaging_context
        self.imaging_ng_threads: int = imaging_ng_threads
        self.imaging_w_stacking: Union[bool, str] = imaging_w_stacking
        self.imaging_flat_sky: Union[bool, str] = imaging_flat_sky
        self.imaging_npixel: int = imaging_npixel
        self.imaging_cellsize: float = imaging_cellsize
        self.imaging_weighting: str = imaging_weighting
        self.imaging_robustness: float = imaging_robustness
        self.imaging_gaussian_taper: float = imaging_gaussian_taper
        self.imaging_dopsf: Union[bool, str] = imaging_dopsf
        self.imaging_dft_kernel: str = imaging_dft_kernel

    def __getattribute__(self, name) -> object:
        """
        Ensures that the variable access of bool are casted to str since RASCIL defined their bool to be str
        """
        value = object.__getattribute__(self, name)
        if isinstance(value, bool):
            return str(value)
        else:
            return value

    def get_dirty_image(self) -> Image:
        """
        Get Dirty Image of visibilities passed to the Imager.
        :return: dirty image of visibilities.
        """
        block_visibilities = create_blockvisibility_from_ms(self.visibility.file.path)
        if len(block_visibilities) != 1:
            raise EnvironmentError("Visibilities are too large")
        visibility = block_visibilities[0]
        image = Image()
        model = create_image_from_visibility(
            visibility, cellsize=self.imaging_cellsize, npixel=self.imaging_npixel
        )
        dirty, sumwt = invert_blockvisibility(visibility, model, context="2d")
        export_image_to_fits(dirty, f"{image.file.path}")
        return image

    def imaging_rascil(
        self,
        client: Client = None,
        use_dask: bool = False,
        n_threads: int = 1,
        use_cuda: bool = False, # If True, use CUDA for Nifty Gridder
        img_context: str = "ng", # Imaging context: Which nifty gridder to use. See: https://ska-telescope.gitlab.io/external/rascil/RASCIL_wagg.html
        num_bright_sources: int = None,
        # Number of brightest sources to select for initial SkyModel (if None, use all sources from input file)
        clean_algorithm: str = "hogbom",
        # Type of deconvolution algorithm (hogbom or msclean or mmclean)
        clean_beam: Dict[str, float] = None,
        # Clean beam: major axis, minor axis, position angle (deg) DataFormat. 3 args. NEEDS TESTING!!
        clean_scales: List[int] = [
            0
        ],  # Scales for multiscale clean (pixels) e.g. [0, 6, 10]
        clean_nmoment: int = 4,
        # Number of frequency moments in mmclean (1 is a constant, 2 is linear, etc.)
        clean_nmajor: int = 5,  # Number of major cycles in cip or ical
        clean_niter: int = 1000,  # Number of minor cycles in CLEAN (i.e. clean iterations)
        clean_psf_support: int = 256,  # Half-width of psf used in cleaning (pixels)
        clean_gain: float = 0.1,  # Clean loop gain
        clean_threshold: float = 1e-4,  # Clean stopping threshold (Jy/beam)
        clean_component_threshold: float = None,
        # Sources with absolute flux > this level (Jy) are fit or extracted using skycomponents
        clean_component_method: str = "fit",
        # Method to convert sources in image to skycomponents: 'fit' in frequency or 'extract' actual values
        clean_fractional_threshold: float = 0.3,  # Fractional stopping threshold for major cycle
        clean_facets: int = 1,  # Number of overlapping facets in faceted clean (along each axis)
        clean_overlap: int = 32,  # Overlap of facets in clean (pixels)
        clean_taper: str = "tukey",
        # Type of interpolation between facets in deconvolution (none or linear or tukey)
        clean_restore_facets: int = 1,  # Number of overlapping facets in restore step (along each axis)
        clean_restore_overlap: int = 32,  # Overlap of facets in restore step (pixels)
        clean_restore_taper: str = "tukey",
        # Type of interpolation between facets in restore step (none or linear or tukey)
        clean_restored_output: str = "list",
        # Type of restored image output: taylor, list, or integrated
    ) -> Tuple[Image, Image, Image]:
        """
        Starts imaging process using RASCIL, will run a CLEAN algorithm on the passed visibilities to the
        Imager.

        :returns (Deconvolved Image, Restored Image, Residual Image)
        """
        if (use_cuda and use_dask) or (use_cuda and client is not None):
            raise EnvironmentError("Cannot use CUDA and Dask at the same time")
        if client and not use_dask:
            raise EnvironmentError("Client passed but use_dask is False")
        if use_dask and not client:
            client = get_global_client()
        if client:
            print(client.cluster)
        # Set CUDA parameters
        if use_cuda:
            img_context='wg'
        rsexecute.set_client(use_dask=use_dask, client=client, use_dlg=False)

        blockviss = create_blockvisibility_from_ms_rsexecute(
            msname=self.visibility.file.path,
            nchan_per_blockvis=self.ingest_chan_per_blockvis,
            nout=self.ingest_vis_nchan // self.ingest_chan_per_blockvis,
            dds=self.ingest_dd,
            average_channels=True,
        )

        blockviss = [
            rsexecute.execute(convert_blockvisibility_to_stokesI)(bv)
            for bv in blockviss
        ]

        cellsize = self.imaging_cellsize
        models = [
            rsexecute.execute(create_image_from_visibility)(
                bvis,
                npixel=self.imaging_npixel,
                nchan=self.imaging_nchan,
                cellsize=cellsize,
                polarisation_frame=PolarisationFrame("stokesI"),
            )
            for bvis in blockviss
        ]
        result = continuum_imaging_skymodel_list_rsexecute_workflow(
            blockviss,  # List of BlockVisibilitys
            models,  # List of model images
            context=img_context,
            threads=n_threads,
            wstacking=self.imaging_w_stacking
            == "True",  # Correct for w term in gridding
            niter=clean_niter,  # iterations in minor cycle
            nmajor=clean_nmajor,  # Number of major cycles
            algorithm=clean_algorithm,
            gain=clean_gain,  # CLEAN loop gain
            scales=clean_scales,  # Scales for multi-scale cleaning
            fractional_threshold=clean_fractional_threshold,
            # Threshold per major cycle
            threshold=clean_threshold,  # Final stopping threshold
            nmoment=clean_nmoment,
            # Number of frequency moments (1 = no dependence)
            psf_support=clean_psf_support,
            # Support of PSF used in minor cycles (halfwidth in pixels)
            restored_output=clean_restored_output,  # Type of restored image
            deconvolve_facets=clean_facets,
            deconvolve_overlap=clean_overlap,
            deconvolve_taper=clean_taper,
            restore_facets=clean_restore_facets,
            restore_overlap=clean_restore_overlap,
            restore_taper=clean_restore_taper,
            dft_compute_kernel=self.imaging_dft_kernel,
            component_threshold=clean_component_threshold,
            component_method=clean_component_method,
            flat_sky=self.imaging_flat_sky,
            clean_beam=clean_beam,
            clean_algorithm=clean_algorithm,
        )

        result = rsexecute.compute(result, sync=True)

        residual, restored, skymodel = result

        deconvolved = [sm.image for sm in skymodel]
        deconvolved_image = image_gather_channels(deconvolved)
        deconvoled_image = Image()
        export_image_to_fits(deconvolved_image, deconvoled_image.file.path)

        restored_image = Image()
        if isinstance(restored, list):
            restored = image_gather_channels(restored)
        export_image_to_fits(restored, restored_image.file.path)

        residual = remove_sumwt(residual)
        if isinstance(residual, list):
            residual = image_gather_channels(residual)
        residual_image = Image()
        export_image_to_fits(residual, residual_image.file.path)

        return deconvoled_image, restored_image, residual_image
