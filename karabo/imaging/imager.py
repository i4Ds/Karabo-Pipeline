import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from astropy.wcs import WCS
from distributed import Client
from numpy.typing import NDArray
from rascil import processing_components as rpc
from rascil.processing_components import create_visibility_from_ms
from rascil.workflows import (
    continuum_imaging_skymodel_list_rsexecute_workflow,
    create_visibility_from_ms_rsexecute,
)
from rascil.workflows.rsexecute.execution_support import rsexecute
from ska_sdp_datamodels.image.image_model import Image as SkaSdpImage
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_func_python.image import image_gather_channels
from ska_sdp_func_python.imaging import (
    create_image_from_visibility,
    invert_visibility,
    remove_sumwt,
)
from ska_sdp_func_python.visibility import convert_visibility_to_stokesI

from karabo.data.external_data import MGCLSContainerDownloadObject
from karabo.imaging.image import Image
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.visibility import Visibility
from karabo.util._types import FilePathType
from karabo.util.dask import fetch_dask_handler
from karabo.util.file_handler import FileHandler, check_ending

ImageContextType = Literal["awprojection", "2d", "ng", "wg"]
CleanAlgorithmType = Literal["hogbom", "msclean", "mmclean"]
CleanBeamInDegType = Literal["bmaj", "bmin", "bpa"]
CleanComponentMethodType = Literal["fit", "extract"]
CleanTaperType = Literal["none", "linear", "tukey"]
CleanRestoredOutputType = Literal["taylor", "list", "integrated"]


def get_MGCLS_images(regex_pattern: str, verbose: bool = False) -> List[SkaSdpImage]:
    """
    MeerKAT Galaxy Cluster Legacy Survey Data Release 1 (MGCLS DR1)
    https://doi.org/10.48479/7epd-w356
    The first data release of the MeerKAT Galaxy Cluster Legacy Survey (MGCLS)
    consists of the uncalibrated visibilities, a set of continuum imaging products,
    and several source catalogues. All clusters have Stokes-I products,
    and approximately 40% have Stokes-Q and U products as well. For full details,
    including caveats for usage,
    see the survey overview and DR1 paper (Knowles et al., 2021).

    When using any of the below products, please cite Knowles et al. (2021)
    and include the following Observatory acknowledgement:
    "MGCLS data products were provided by the South African Radio
    Astronomy Observatory and the MGCLS team and were derived from observations
    with the MeerKAT radio telescope. The MeerKAT telescope is operated by the
    South African Radio Astronomy Observatory, which is a facility of the National
    Research Foundation, an agency of the Department of Science and Innovation."

    The final enhanced image data products are five-plane cubes
    (referred to as the 5pln cubes in the following) in which the first
    plane is the brightness at the reference frequency, and the second
    is the spectral index, a**1656/908 , both determined by a least-squares fit
    to log(I) vs. log(v) at each pixel. The third plane is the brightness
    uncertainty estimate, fourth is the spectral index uncertainty, and
    fifth is the χ2 of the least-squares fit. Uncertainty estimates are
    only the statistical noise component and do not include calibration
    or other systematic effects. The five planes are accessible in the
    Xarray.Image in the frequency dimension (first dimension).

    Data will be accessed from the karabo_public folder. The data was downloaded
    from https://archive-gw-1.kat.ac.za/public/repository/10.48479/7epd-w356/
    data/enhanced_products/bucket_contents.html

    Parameters:
    ----------
    regex_pattern : str
        Regex pattern to match the files to download. Best is to check in the bucket
        and paper which data is available and then use the regex pattern to match
        the files you want to download.
    verbose : bool, optional
        If True, prints out the files being downloaded. Defaults to False.

    Returns:
    -------
    List[SkaSdpImage]
        List of images from the MGCLS Enhanced Products bucket.
    """
    mgcls_cdo = MGCLSContainerDownloadObject(regexr_pattern=regex_pattern)
    local_file_paths = mgcls_cdo.get_all(verbose=verbose)
    if len(local_file_paths) == 0:
        raise ValueError(
            f"No files in {mgcls_cdo._remote_container_url} for {regex_pattern=}"
        )
    mgcls_images: List[SkaSdpImage] = list()
    for local_file_path in local_file_paths:
        mgcls_images.append(
            rpc.image.operations.import_image_from_fits(local_file_path)
        )
    return mgcls_images


class Imager:
    """Imager class provides imaging functionality using the visibilities
    of an observation with the help of RASCIL.

    In addition, it provides the calculation of the pixel coordinates of point sources.

    Parameters
    ---------------------------------------------
    visibility : Visibility, required
        Visibility object containing the visibilities of an observation.
    logfile : str, default=None,
        Name of logfile (default is to construct one from msname)
    performance_file : str, default=None
        Name of json file to contain performance information
    ingest_dd : List[int], default=[0],
        Data descriptors in MS to read (all must have the same number of channels)
    ingest_vis_nchan : int, default=3,
        Number of channels in a single data descriptor in the MS
    ingest_chan_per_vis : int, defualt=1,
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
    override_cellsize : bool, default=False
        Override the cellsize if it is above the critical cellsize
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
    """

    def __init__(
        self,
        visibility: Visibility,
        logfile: Optional[str] = None,
        performance_file: Optional[str] = None,
        ingest_dd: List[int] = [0],
        ingest_vis_nchan: Optional[int] = None,
        ingest_chan_per_vis: int = 1,
        ingest_average_blockvis: Union[bool, str] = False,
        imaging_phasecentre: Optional[str] = None,
        imaging_pol: str = "stokesI",
        imaging_nchan: int = 1,
        imaging_context: str = "ng",
        imaging_ng_threads: int = 4,
        imaging_w_stacking: Union[bool, str] = True,
        imaging_flat_sky: Union[bool, str] = False,
        imaging_npixel: Optional[int] = None,
        imaging_cellsize: Optional[float] = None,
        override_cellsize: bool = False,
        imaging_weighting: str = "uniform",
        imaging_robustness: float = 0.0,
        imaging_gaussian_taper: Optional[float] = None,
        imaging_dopsf: Union[bool, str] = False,
        imaging_uvmax: Optional[float] = None,
        imaging_uvmin: float = 0,
        imaging_dft_kernel: Optional[
            str
        ] = None,  # DFT kernel: cpu_looped | cpu_numba | gpu_raw
    ) -> None:
        self.visibility = visibility
        self.logfile = logfile
        self.performance_file = performance_file
        self.ingest_dd = ingest_dd
        self.ingest_vis_nchan = ingest_vis_nchan
        self.ingest_chan_per_vis = ingest_chan_per_vis
        self.ingest_average_blockvis = ingest_average_blockvis
        self.imaging_phasecentre = imaging_phasecentre
        self.imaging_pol = imaging_pol
        self.imaging_nchan = imaging_nchan
        self.imaging_context = imaging_context
        self.imaging_ng_threads = imaging_ng_threads
        self.imaging_w_stacking = imaging_w_stacking
        self.imaging_flat_sky = imaging_flat_sky
        self.imaging_npixel = imaging_npixel
        self.imaging_cellsize = imaging_cellsize
        self.override_cellsize = override_cellsize
        self.imaging_weighting = imaging_weighting
        self.imaging_robustness = imaging_robustness
        self.imaging_gaussian_taper = imaging_gaussian_taper
        self.imaging_dopsf = imaging_dopsf
        self.imaging_dft_kernel = imaging_dft_kernel
        self.imaging_uvmax = imaging_uvmax
        self.imaging_uvmin = imaging_uvmin

    def get_dirty_image(
        self,
        fits_path: Optional[FilePathType] = None,
    ) -> Image:
        """Get Dirty Image of visibilities passed to the Imager.

        Note: If `fits_path` is provided and already exists, then this function will
        overwrite `fits_path`.

        Args:
            fits_path: Path to where the .fits file will get saved.

        Returns:
            Dirty Image
        """
        if fits_path is None:
            tmp_dir = FileHandler().get_tmp_dir(
                prefix="Imager-Dirty-",
                purpose="disk-cache for dirty.fits",
            )
            fits_path = os.path.join(tmp_dir, "dirty.fits")
        else:
            check_ending(path=fits_path, ending=".fits")

        block_visibilities = create_visibility_from_ms(
            str(self.visibility.ms_file_path)
        )

        if len(block_visibilities) != 1:
            raise EnvironmentError("Visibilities are too large")
        visibility = block_visibilities[0]
        model = create_image_from_visibility(
            visibility,
            npixel=self.imaging_npixel,
            cellsize=self.imaging_cellsize,
            override_cellsize=self.override_cellsize,
        )
        dirty, _ = invert_visibility(visibility, model, context="2d")
        if os.path.exists(fits_path):
            os.remove(fits_path)
        dirty.image_acc.export_to_fits(fits_file=fits_path)

        image = Image(path=fits_path)
        return image

    def imaging_rascil(
        self,
        deconvolved_fits_path: Optional[FilePathType] = None,
        restored_fits_path: Optional[FilePathType] = None,
        residual_fits_path: Optional[FilePathType] = None,
        client: Optional[Client] = None,
        use_dask: bool = False,
        n_threads: int = 1,
        use_cuda: bool = False,
        img_context: ImageContextType = "ng",
        clean_algorithm: CleanAlgorithmType = "hogbom",
        clean_beam: Optional[Dict[CleanBeamInDegType, float]] = None,
        clean_scales: List[int] = [0],
        clean_nmoment: int = 4,
        clean_nmajor: int = 5,
        clean_niter: int = 1000,
        clean_psf_support: int = 256,
        clean_gain: float = 0.1,
        clean_threshold: float = 1e-4,
        clean_component_threshold: Optional[float] = None,
        clean_component_method: CleanComponentMethodType = "fit",
        clean_fractional_threshold: float = 0.3,
        clean_facets: int = 1,
        clean_overlap: int = 32,
        clean_taper: CleanTaperType = "tukey",
        clean_restore_facets: int = 1,
        clean_restore_overlap: int = 32,
        clean_restore_taper: CleanTaperType = "tukey",
        clean_restored_output: CleanRestoredOutputType = "list",
    ) -> Tuple[Image, Image, Image]:
        """Starts imaging process using RASCIL using CLEAN.

        Note: For `deconvolved_fits_path`, `restored_fits_path` & `residual_fits_path`,
        if one or more of them are provided and already exist on the disk, then
        they will get overwritten if the imaging succeeds.

        Clean args see https://developer.skao.int/_/downloads/rascil/en/latest/pdf/

        Args:
            deconvolved_fits_path: Fits file path to save deconvolved image.
            restored_fits_path: Fits file path to save restored image.
            residual_fits_path: Fits file path to save residual image.
            client: Dask client.
            use_dask: Use dask?
            n_threads: n_threads per worker.
            use_cuda: use CUDA for Nifty Gridder?
            img_context: Which nifty gridder to use.
            clean_algorithm: Deconvolution algorithm (hogbom or msclean or mmclean).
            clean_beam: major axis, minor axis, position angle (deg).
            clean_scales: Scales for multiscale clean (pixels) e.g. [0, 6, 10].
            clean_nmoment: Number of frequency moments in mmclean (1=constant, 2=linear)
            clean_nmajor: Number of major cycles in cip or ical.
            clean_niter: Number of minor cycles in CLEAN.
            clean_psf_support: Half-width of psf used in cleaning (pixels).
            clean_gain: Clean loop gain.
            clean_threshold: Clean stopping threshold (Jy/beam).
            clean_component_threshold:  Sources with absolute flux > this level (Jy)
                are fit or extracted using skycomponents.
            clean_component_method: Method to convert sources in image to skycomponents:
                "fit" in frequency or "extract" actual values.
            clean_fractional_threshold: Fractional stopping threshold for major cycle
            clean_facets: Number of overlapping facets in faceted clean along each axis.
            clean_overlap: Overlap of facets in clean (pixels)
            clean_taper: Type of interpolation between facets in deconvolution:
                none or linear or tukey.
            clean_restore_facets: Number of overlapping facets in restore step
                along each axis.
            clean_restore_overlap: Overlap of facets in restore step (pixels)
            clean_restore_taper: Type of interpolation between facets in
                restore step (none, linear or tukey).
            clean_restored_output: Type of restored image output:
                taylor, list, or integrated.

        Returns:
            deconvolved, restored, residual
        """
        if deconvolved_fits_path is not None:
            check_ending(path=deconvolved_fits_path, ending=".fits")
        if restored_fits_path is not None:
            check_ending(path=restored_fits_path, ending=".fits")
        if residual_fits_path is not None:
            check_ending(path=residual_fits_path, ending=".fits")
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

        if client and not use_dask:
            raise RuntimeError("Client passed but use_dask is False")
        if use_dask:
            if not client:
                dask_handler = fetch_dask_handler()
                client = dask_handler.get_dask_client()
            print(client.cluster)
            rsexecute.set_client(use_dask=use_dask, client=client, use_dlg=False)
        # Set CUDA parameters
        if use_cuda:
            if img_context != "wg":
                print(
                    f"Changing imaging_rascil.img_context` from '{img_context}' "
                    + f"to 'wg' because {use_cuda=}"
                )
            img_context = "wg"

        if self.ingest_vis_nchan is None:
            raise ValueError(
                "`self.ingest_vis_nchan` is None but must set, but is None"
            )

        blockviss = create_visibility_from_ms_rsexecute(
            msname=str(self.visibility.ms_file_path),
            nchan_per_vis=self.ingest_chan_per_vis,
            nout=self.ingest_vis_nchan // self.ingest_chan_per_vis,
            dds=self.ingest_dd,
            average_channels=True,
        )

        blockviss = [
            rsexecute.execute(convert_visibility_to_stokesI)(bv) for bv in blockviss
        ]

        models = [
            rsexecute.execute(create_image_from_visibility)(
                bvis,
                npixel=self.imaging_npixel,
                nchan=self.imaging_nchan,
                cellsize=self.imaging_cellsize,
                override_cellsize=self.override_cellsize,
                polarisation_frame=PolarisationFrame("stokesI"),
            )
            for bvis in blockviss
        ]

        result = continuum_imaging_skymodel_list_rsexecute_workflow(
            vis_list=blockviss,
            model_imagelist=models,
            context=img_context,
            threads=n_threads,
            wstacking=self.imaging_w_stacking == "True",
            niter=clean_niter,
            nmajor=clean_nmajor,
            algorithm=clean_algorithm,
            gain=clean_gain,
            scales=clean_scales,
            fractional_threshold=clean_fractional_threshold,
            threshold=clean_threshold,
            nmoment=clean_nmoment,
            psf_support=clean_psf_support,
            restored_output=clean_restored_output,
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
            imaging_uvmax=self.imaging_uvmax,
            imaging_uvmin=self.imaging_uvmin,
        )

        result = rsexecute.compute(result, sync=True)

        residual, restored, skymodel = result

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

    @staticmethod
    def project_sky_to_image(
        sky: SkyModel,
        phase_center: Union[List[int], List[float]],
        imaging_cellsize: float,
        imaging_npixel: int,
        filter_outlier: bool = True,
        invert_ra: bool = True,
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Calculates the pixel coordinates `sky` sources as floats.
        If you want to have integer indices, just round them.

        :param sky: `SkyModel` with the sources
        :param phase_center: [RA,DEC]
        :param imaging_cellsize: Image cellsize in radian (pixel coverage)
        :param imaging_npixel: Number of pixels of the image
        :param filter_outlier: Exclude source outside of image?
        :param invert_ra: Invert RA axis?

        :return: image-coordinates as np.ndarray[px,py] and
        `SkyModel` sources indices as np.ndarray[idxs]
        """

        # calc WCS args
        def radian_degree(rad: float) -> float:
            return rad * (180 / np.pi)

        cdelt = radian_degree(imaging_cellsize)
        crpix = np.floor((imaging_npixel / 2)) + 1

        # setup WCS
        w = WCS(naxis=2)
        w.wcs.crpix = np.array([crpix, crpix])  # coordinate reference pixel per axis
        ra_sign = -1 if invert_ra else 1
        w.wcs.cdelt = np.array(
            [ra_sign * cdelt, cdelt]
        )  # coordinate increments on sphere per axis
        w.wcs.crval = phase_center
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]  # coordinate axis type

        # convert coordinates
        px, py = w.wcs_world2pix(sky[:, 0], sky[:, 1], 1)

        # check length to cover single source pre-filtering
        if len(px.shape) == 0 and len(py.shape) == 0:
            px, py = [px], [py]
            idxs = np.arange(sky.num_sources)
        # post processing, pre filtering before calling wcs.wcs_world2pix would be
        # more efficient, however this has to be done in the ra-dec space.
        # Maybe for future work!?
        elif filter_outlier:
            px_idxs = np.where(np.logical_and(px <= imaging_npixel, px >= 0))[0]
            py_idxs = np.where(np.logical_and(py <= imaging_npixel, py >= 0))[0]
            idxs = np.intersect1d(px_idxs, py_idxs)
            px, py = px[idxs], py[idxs]
        else:
            idxs = np.arange(sky.num_sources)
        img_coords = np.array([px, py])

        return img_coords, idxs
