from astropy.io import fits
from astropy import wcs
import numpy as np
from karabo.simulation.sky_model import SkyModel
from rascil.apps import rascil_imager
from rascil.processing_components.util.performance import (
    performance_store_dict,
    performance_environment,
)
from typing import List, Union, Dict, Tuple

class Imager:
    """
    The Imager class provides imaging functionality using the visibilities of an observation.
    """
    def __init__(self, mode: str = 'cip', # Processing  cip | ical | invert | load
                 logfile: str = None, # Name of logfile (default is to construct one from msname)
                 performance_file: str = None, # Name of json file to contain performance information
                 ingest_msname: str = None, # MeasurementSet to be read
                 ingest_dd: List[int] = [0], # Data descriptors in MS to read (all must have the same number of channels)
                 ingest_vis_nchan: int = None, # Number of channels in a single data descriptor in the MS
                 ingest_chan_per_blockvis: int = 1, # Number of channels per blockvis (before any average)
                 ingest_average_blockvis: Union[bool,str] = False, # Average all channels in blockvis?
                 imaging_phasecentre: str = None, # Phase centre (in SkyCoord string format)
                 imaging_pol: str = 'stokesI', # RASCIL polarisation frame for image
                 imaging_nchan: int = 1, # Number of channels per image
                 imaging_context: str = 'ng', # Imaging context i.e. the gridder used 2d | ng
                 imaging_ng_threads: int = 4, # Number of Nifty Gridder threads to use (4 is a good choice)
                 imaging_w_stacking: Union[bool,str] = True, # Use the improved w stacking method in Nifty Gridder?
                 imaging_flat_sky: Union[bool,str] = False, # If using a primary beam, normalise to flat sky?
                 imaging_npixel: int = None, # Number of pixels in ra, dec: Should be a composite of 2, 3, 5
                 imaging_cellsize: float = None, # Cellsize (radians). Default is to calculate
                 imaging_weighting: str = 'uniform', # Type of weighting uniform or robust or natural
                 imaging_robustness: float = .0, # Robustness for robust weighting
                 imaging_gaussian_taper: float = None, # Size of Gaussian smoothing, implemented as taper in weights (rad)
                 imaging_dopsf: Union[bool,str] = False, # Make the PSF instead of the dirty image?
                 imaging_dft_kernel: str = None, # DFT kernel: cpu_looped | cpu_numba | gpu_raw
                 imaging_uvmax: float = None, # NotImplementedByRASCIL despite it's documented
                 imaging_uvmin: float = None, # NotImplementedByRASCIL despite it's documented
                 imaging_rmax: float = None, # NotImplementedByRASCIL despite it's documented
                 imaging_rmin: float = None, # NotImplementedByRASCIL despite it's documented
                 calibration_reset_skymodel: Union[bool,str] = True, # Reset the initial skymodel after initial calibration?
                 calibration_T_first_selfcal: int = 1, # First selfcal for T (complex gain). T is common to both receptors
                 calibration_T_phase_only: Union[bool,str] = True, # Phase only solution
                 calibration_T_timeslice: float = None, # Solution length (s) 0 means minimum
                 calibration_G_first_selfcal: int = 3, # First selfcal for G (complex gain). G is different for the two receptors
                 calibration_G_phase_only: Union[bool,str] = False, # Phase only solution?
                 calibration_G_timeslice: float = None, # Solution length (s) 0 means minimum
                 calibration_B_first_selfcal: int = 4, # First selfcal for B (bandpass complex gain). B is complex gain per frequency.
                 calibration_B_phase_only: Union[bool,str] = False, # Phase only solution
                 calibration_B_timeslice: float = None, # Solution length (s)
                 calibration_global_solution: Union[bool,str] = True, # Solve across frequency
                 calibration_context: str = 'T', # Terms to solve (in order e.g. TGB)
                 use_initial_skymodel: Union[bool,str] = False, # Whether to use an initial SkyModel in ICAL or not
                 input_skycomponent_file: str = None, # Input name of skycomponents file (in hdf or txt format) for initial SkyModel in ICAL
                 num_bright_sources: int = None, # Number of brightest sources to select for initial SkyModel (if None, use all sources from input file)
                 clean_algorithm: str = 'mmclean', # Type of deconvolution algorithm (hogbom or msclean or mmclean)
                 clean_beam: Dict[str, float] = None, # Clean beam: major axis, minor axis, position angle (deg) DataFormat. 3 args. NEEDS TESTING!!
                 clean_scales: List[int] = [0], # Scales for multiscale clean (pixels) e.g. [0, 6, 10]
                 clean_nmoment: int = 4, # Number of frequency moments in mmclean (1 is a constant, 2 is linear, etc.)
                 clean_nmajor: int = 5, # Number of major cycles in cip or ical
                 clean_niter: int = 1000, # Number of minor cycles in CLEAN (i.e. clean iterations)
                 clean_psf_support: int = 256, # Half-width of psf used in cleaning (pixels)
                 clean_gain: float = .1, # Clean loop gain
                 clean_threshold: float = 1e-4, # Clean stopping threshold (Jy/beam)
                 clean_component_threshold: float = None, # Sources with absolute flux > this level (Jy) are fit or extracted using skycomponents
                 clean_component_method: str = 'fit', # Method to convert sources in image to skycomponents: 'fit' in frequency or 'extract' actual values
                 clean_fractional_threshold: float = .3, # Fractional stopping threshold for major cycle
                 clean_facets: int = 1, # Number of overlapping facets in faceted clean (along each axis)
                 clean_overlap: int = 32, # Overlap of facets in clean (pixels)
                 clean_taper: str = 'tukey', # Type of interpolation between facets in deconvolution (none or linear or tukey)
                 clean_restore_facets: int = 1, # Number of overlapping facets in restore step (along each axis)
                 clean_restore_overlap: int = 32, # Overlap of facets in restore step (pixels)
                 clean_restore_taper: str = 'tukey', # Type of interpolation between facets in restore step (none or linear or tukey)
                 clean_restored_output: str = 'list', # Type of restored image output: taylor, list, or integrated
                 use_dask: Union[bool,str] = True, # Use Dask processing? False means that graphs are executed as they are constructed
                 dask_nthreads: int = None, # Number of threads in each Dask worker (None means Dask will choose)
                 dask_memory: str = None, # Memory per Dask worker (GB), e.g. 5GB (None means Dask will choose)
                 dask_memory_usage_file: str = None, # File in which to track Dask memory use (using dask-memusage)
                 dask_nodes: str = None, # Node names for SSHCluster
                 dask_nworkers: int = None, # Number of workers (None means Dask will choose)
                 dask_scheduler: str = None, # Externally defined Dask scheduler e.g. 127.0.0.1:8786 or ssh for SSHCluster or existing for current scheduler
                 dask_scheduler_file: str = None, # Externally defined Dask scheduler file to setup dask cluster
                 dask_tcp_timeout: str = None, # Dask TCP timeout
                 dask_connect_timeout: str = None, # Dask connect timeout
                 dask_malloc_trim_threshold: int = 0 # Threshold for trimming memory on release (0 is aggressive)
                 ):
        self.mode: str = mode
        self.logfile: str = logfile
        self.performance_file: str = performance_file
        self.ingest_msname: str = ingest_msname
        self.ingest_dd: List[int] = ingest_dd
        self.ingest_vis_nchan: int = ingest_vis_nchan
        self.ingest_chan_per_blockvis: int = ingest_chan_per_blockvis
        self.ingest_average_blockvis: Union[bool,str] = ingest_average_blockvis
        self.imaging_phasecentre: str = imaging_phasecentre
        self.imaging_pol: str = imaging_pol
        self.imaging_nchan: int = imaging_nchan
        self.imaging_context: str = imaging_context
        self.imaging_ng_threads: int = imaging_ng_threads
        self.imaging_w_stacking: Union[bool,str] = imaging_w_stacking
        self.imaging_flat_sky: Union[bool,str] = imaging_flat_sky
        self.imaging_npixel: int = imaging_npixel
        self.imaging_cellsize: float = imaging_cellsize
        self.imaging_weighting: str = imaging_weighting
        self.imaging_robustness: float = imaging_robustness
        self.imaging_gaussian_taper: float = imaging_gaussian_taper
        self.imaging_dopsf: Union[bool,str] = imaging_dopsf
        self.imaging_dft_kernel: str = imaging_dft_kernel
        self.imaging_uvmax: float = imaging_uvmax
        self.imaging_uvmin: float = imaging_uvmin
        self.imaging_rmax: float = imaging_rmax
        self.imaging_rmin: float = imaging_rmin
        self.calibration_reset_skymodel: Union[bool,str] = calibration_reset_skymodel
        self.calibration_T_first_selfcal: int = calibration_T_first_selfcal
        self.calibration_T_phase_only: Union[bool,str] = calibration_T_phase_only
        self.calibration_T_timeslice: float = calibration_T_timeslice
        self.calibration_G_first_selfcal: int = calibration_G_first_selfcal
        self.calibration_G_phase_only: Union[bool,str] = calibration_G_phase_only
        self.calibration_G_timeslice: float = calibration_G_timeslice
        self.calibration_B_first_selfcal: int = calibration_B_first_selfcal
        self.calibration_B_phase_only: Union[bool,str] = calibration_B_phase_only
        self.calibration_B_timeslice: float = calibration_B_timeslice
        self.calibration_global_solution: Union[bool,str] = calibration_global_solution
        self.calibration_context: str = calibration_context
        self.use_initial_skymodel: Union[bool,str] = use_initial_skymodel
        self.input_skycomponent_file: str = input_skycomponent_file
        self.num_bright_sources: int = num_bright_sources
        self.clean_algorithm: str = clean_algorithm
        self.clean_beam: Dict[str, float] = clean_beam
        self.clean_scales: List[int] = clean_scales
        self.clean_nmoment: int = clean_nmoment
        self.clean_nmajor: int = clean_nmajor
        self.clean_niter: int = clean_niter
        self.clean_psf_support: int = clean_psf_support
        self.clean_gain: float = clean_gain
        self.clean_threshold: float = clean_threshold
        self.clean_component_threshold: float = clean_component_threshold
        self.clean_component_method: str = clean_component_method
        self.clean_fractional_threshold: float = clean_fractional_threshold
        self.clean_facets: int = clean_facets
        self.clean_overlap: int = clean_overlap
        self.clean_taper: str = clean_taper
        self.clean_restore_facets: int = clean_restore_facets
        self.clean_restore_overlap: int = clean_restore_overlap
        self.clean_restore_taper: str = clean_restore_taper
        self.clean_restored_output: str = clean_restored_output
        self.use_dask: Union[bool,str] = use_dask
        self.dask_nthreads: int = dask_nthreads
        self.dask_memory: str = dask_memory
        self.dask_memory_usage_file: str = dask_memory_usage_file
        self.dask_nodes: str = dask_nodes
        self.dask_nworkers: int = dask_nworkers
        self.dask_scheduler: str = dask_scheduler
        self.dask_scheduler_file: str = dask_scheduler_file
        self.dask_tcp_timeout: str = dask_tcp_timeout
        self.dask_connect_timeout: str = dask_connect_timeout
        self.dask_malloc_trim_threshold: int = dask_malloc_trim_threshold

    def __getattribute__(self, name) -> object:
        """
        Ensures that the variable access of bool are casted to str since RASCIL defined their bool to be str
        """
        value = object.__getattribute__(self, name)
        if isinstance(value, bool):
            return str(value)
        else:
            return value
    
    def imaging_rascil(self):
        """
        Starts imagimg process using RASCIL
        """
        performance_environment(self.performance_file, mode='w')
        performance_store_dict(self.performance_file, 'imgaging_args', vars(self), mode='a')
        _ = rascil_imager.imager(self) # _ is image_name

    @staticmethod
    def get_pixel_coord(fits_path: str, sky: SkyModel) -> Tuple[np.ndarray,np.ndarray]:
        """
        Calculates the pixel coordinates of the produced .fits file
        
        :param fits_path: path to the .fits image
        :param sky: SkyModel which was used to produce the .fits image

        :return: pixel-coordinates x-axis, pixel-coordinates y-axis
        """
        hdulist = fits.open(fits_path)
        wcs_fits = wcs.WCS(hdulist[0].header)
        wcs = sky.wcs.copy()
        wcs.wcs.crpix = wcs_fits.wcs.crpix[0:2]
        wcs.wcs.cdelt = wcs_fits.wcs.cdelt[0:2]
        px, py = wcs.wcs_world2pix(sky[:,0], sky[:,1], 1)
        return px, py
