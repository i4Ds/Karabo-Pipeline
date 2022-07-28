from distributed import Client, LocalCluster

from karabo.imaging.image import Image
from karabo.simulation.Visibility import Visibility
from karabo.util.dask import get_local_dask_client

from typing import List, Union, Dict, Tuple

import numpy as np
from rascil.processing_components import create_blockvisibility_from_ms, create_image_from_visibility, \
    invert_blockvisibility, export_image_to_fits, image_gather_channels, remove_sumwt, \
    convert_blockvisibility_to_stokesI
from rascil.data_models import PolarisationFrame
from rascil.workflows import create_blockvisibility_from_ms_rsexecute, \
    continuum_imaging_skymodel_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support import rsexecute

from karabo.simulation.sky_model import SkyModel


class Imager:
    """
    The Imager class provides imaging functionality using the visibilities of an observation.
    In addition, it provides the calculation of the pixel coordinates of point sources.
    """

    def __init__(self, visibility: Visibility,
                 logfile: str = None,  # Name of logfile (default is to construct one from msname)
                 performance_file: str = None,  # Name of json file to contain performance information
                 ingest_dd: List[int] = [0],
                 # Data descriptors in MS to read (all must have the same number of channels)
                 ingest_vis_nchan: int = None,  # Number of channels in a single data descriptor in the MS
                 ingest_chan_per_blockvis: int = 1,  # Number of channels per blockvis (before any average)
                 ingest_average_blockvis: Union[bool, str] = False,  # Average all channels in blockvis?
                 imaging_phasecentre: str = None,  # Phase centre (in SkyCoord string format)
                 imaging_pol: str = 'stokesI',  # RASCIL polarisation frame for image
                 imaging_nchan: int = 1,  # Number of channels per image
                 imaging_context: str = 'ng',  # imaging context i.e. the gridder used 2d | ng
                 imaging_ng_threads: int = 4,  # Number of Nifty Gridder threads to use (4 is a good choice)
                 imaging_w_stacking: Union[bool, str] = True,  # Use the improved w stacking method in Nifty Gridder?
                 imaging_flat_sky: Union[bool, str] = False,  # If using a primary beam, normalise to flat sky?
                 imaging_npixel: int = None,  # Number of pixels in ra, dec: Should be a composite of 2, 3, 5
                 imaging_cellsize: float = None,  # Cellsize (radians). Default is to calculate
                 imaging_weighting: str = 'uniform',  # Type of weighting uniform or robust or natural
                 imaging_robustness: float = .0,  # Robustness for robust weighting
                 imaging_gaussian_taper: float = None,
                 # Size of Gaussian smoothing, implemented as taper in weights (rad)
                 imaging_dopsf: Union[bool, str] = False,  # Make the PSF instead of the dirty image?
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
        block_visibilities = create_blockvisibility_from_ms(self.visibility.path)
        if len(block_visibilities) != 1:
            raise EnvironmentError("Visibilities are too large")
        visibility = block_visibilities[0]
        image = Image()
        model = create_image_from_visibility(visibility, cellsize=self.imaging_cellsize, npixel=self.imaging_npixel)
        dirty, sumwt = invert_blockvisibility(visibility, model, context="2d")
        export_image_to_fits(dirty, f"{image.file.path}")
        return image

    def imaging_rascil(self,
                       num_bright_sources: int = None,
                       # Number of brightest sources to select for initial SkyModel (if None, use all sources from input file)
                       clean_algorithm: str = 'mmclean',
                       # Type of deconvolution algorithm (hogbom or msclean or mmclean)
                       clean_beam: Dict[str, float] = None,
                       # Clean beam: major axis, minor axis, position angle (deg) DataFormat. 3 args. NEEDS TESTING!!
                       clean_scales: List[int] = [0],  # Scales for multiscale clean (pixels) e.g. [0, 6, 10]
                       clean_nmoment: int = 4,
                       # Number of frequency moments in mmclean (1 is a constant, 2 is linear, etc.)
                       clean_nmajor: int = 5,  # Number of major cycles in cip or ical
                       clean_niter: int = 1000,  # Number of minor cycles in CLEAN (i.e. clean iterations)
                       clean_psf_support: int = 256,  # Half-width of psf used in cleaning (pixels)
                       clean_gain: float = .1,  # Clean loop gain
                       clean_threshold: float = 1e-4,  # Clean stopping threshold (Jy/beam)
                       clean_component_threshold: float = None,
                       # Sources with absolute flux > this level (Jy) are fit or extracted using skycomponents
                       clean_component_method: str = 'fit',
                       # Method to convert sources in image to skycomponents: 'fit' in frequency or 'extract' actual values
                       clean_fractional_threshold: float = .3,  # Fractional stopping threshold for major cycle
                       clean_facets: int = 1,  # Number of overlapping facets in faceted clean (along each axis)
                       clean_overlap: int = 32,  # Overlap of facets in clean (pixels)
                       clean_taper: str = 'tukey',
                       # Type of interpolation between facets in deconvolution (none or linear or tukey)
                       clean_restore_facets: int = 1,  # Number of overlapping facets in restore step (along each axis)
                       clean_restore_overlap: int = 32,  # Overlap of facets in restore step (pixels)
                       clean_restore_taper: str = 'tukey',
                       # Type of interpolation between facets in restore step (none or linear or tukey)
                       clean_restored_output: str = 'list',
                       # Type of restored image output: taylor, list, or integrated
                       ) -> (Image, Image, Image):
        """
        Starts imaging process using RASCIL, will run a CLEAN algorithm on the passed visibilities to the
        Imager.

        :returns (Deconvolved Image, Restored Image, Residual Image)
        """
        client = get_local_dask_client(5)
        print(client.cluster)
        rsexecute.set_client(client)

        blockviss = create_blockvisibility_from_ms_rsexecute(
            msname=self.visibility.path,
            nchan_per_blockvis=self.ingest_chan_per_blockvis,
            nout=self.ingest_vis_nchan // self.ingest_chan_per_blockvis,
            dds=self.ingest_dd,
            average_channels=True
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
                polarisation_frame=PolarisationFrame('stokesI'),
            )
            for bvis in blockviss
        ]
        result = continuum_imaging_skymodel_list_rsexecute_workflow(
            blockviss,  # List of BlockVisibilitys
            models,  # List of model images
            context=self.imaging_context,  # Use nifty-gridder
            threads=self.imaging_ng_threads,
            wstacking=self.imaging_w_stacking == "True",  # Correct for w term in gridding
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

    @staticmethod
    def sky_sources_to_pixel_coordinates(image_cell_size: float, image_pixel_per_side: float, sky: SkyModel,
                                         filter_outlier: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the pixel coordinates of the given sky sources, based on the dimensions passed for a certain image

        :param image_pixel_per_side: Image cell-size in radian (pixel coverage)
        :param image_cell_size: Number of pixels of the image
        :param sky: SkyModel with the sources at catalog
        :param filter_outlier: Exclude source

        :return: pixel-coordinates x-axis, pixel-coordinates y-axis, sky sources indices
        """

        if sky.wcs is None:
            raise BaseException("Sky does not have a WCS (world coordinate system). "
                  "Please add one with sky.setup_default_wcs(phase_center) or with sky.add_wcs(wcs)")

        radian_degree = lambda rad: rad * (180 / np.pi)
        cdelt = radian_degree(image_cell_size)
        crpix = np.floor((image_pixel_per_side / 2)) + 1
        wcs = sky.wcs.copy()
        wcs.wcs.crpix = np.array([crpix, crpix])
        wcs.wcs.cdelt = np.array([-cdelt, cdelt])
        px, py = wcs.wcs_world2pix(sky[:, 0], sky[:, 1], 1)

        # pre-filtering before calling wcs.wcs_world2pix would be more efficient,
        # however this has to be done in the ra-dec space. maybe for future work
        if filter_outlier:
            px_idxs = np.where(np.logical_and(px <= image_pixel_per_side, px >= 0))[0]
            py_idxs = np.where(np.logical_and(py <= image_pixel_per_side, py >= 0))[0]
            idxs = np.intersect1d(px_idxs, py_idxs)
            px, py = px[idxs], py[idxs]
        else:
            idxs = np.arange(sky.num_sources)
        return np.vstack((px, py, idxs))
