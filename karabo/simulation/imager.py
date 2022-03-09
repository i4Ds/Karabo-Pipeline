from rascil.apps import rascil_imager
from rascil.processing_components.util.performance import (
    performance_store_dict,
    performance_environment,
)
from typing import List, Union, Dict

class Imager:
    """
    The Imager class provides imaging functionality using the visibilities of an observation.
    """
    def __init__(self, mode: str = 'cip',
                 logfile: str = None,
                 performance_file: str = None,
                 ingest_msname: str = None,
                 ingest_dd: int = 0,
                 ingest_vis_nchan: int = None,
                 ingest_chan_per_blockvis: int = 1,
                 ingest_average_blockvis: bool = False,
                 imaging_phasecentre: str = None,
                 imaging_pol: str = 'stokesI',
                 imaging_nchan: int = 1,
                 imaging_context: str = 'ng',
                 imaging_ng_threads: int = 4,
                 imaging_w_stacking: bool = True,
                 imaging_flat_sky: bool = False,
                 imaging_npixel: int = None,
                 imaging_cellsize: float = None,
                 imaging_weighting: str = 'uniform',
                 imaging_robustness: float = .0,
                 imaging_gaussian_taper: float = None,
                 imaging_dopsf: bool = False,
                 imaging_dft_kernel: str = None,
                 imaging_uvmax: float = None,
                 imaging_uvmin: float = None,
                 imaging_rmax: float = None,
                 imaging_rmin: float = None,
                 calibration_reset_skymodel: bool = True,
                 calibration_T_first_selfcal: int = 1,
                 calibration_T_phase_only: bool = True,
                 calibration_T_timeslice: float = None,
                 calibration_G_first_selfcal: int = 3,
                 calibration_G_phase_only: bool = False,
                 calibration_G_timeslice: float = None,
                 calibration_B_first_selfcal: int = 4,
                 calibration_B_phase_only: bool = False,
                 calibration_B_timeslice: float = None,
                 calibration_global_solution: bool = True,
                 calibration_context: str = 'T',
                 use_initial_skymodel: bool = False,
                 input_skycomponent_file: str = None,
                 num_bright_sources: int = None,
                 clean_algorithm: str = 'mmclean',
                 clean_beam: Dict[str, float] = None,
                 clean_scales: List[int] = [0],
                 clean_nmoment: int = 4,
                 clean_nmajor: int = 5,
                 clean_niter: int = 1000,
                 clean_psf_support: int = 256,
                 clean_gain: float = .1,
                 clean_threshold: float = 1e-4,
                 clean_component_threshold: float = None,
                 clean_component_method: str = 'fit',
                 clean_fractional_threshold: float = .3,
                 clean_facets: int = 1,
                 clean_overlap: int = 32,
                 clean_taper: str = 'tukey',
                 clean_restore_facets: int = 1,
                 clean_restore_overlap: int = 32,
                 clean_restore_taper: str = 'tukey',
                 clean_restored_output: str = 'list',
                 use_dask: bool = True,
                 dask_nthreads: int = None,
                 dask_memory: float = None,
                 dask_memory_usage_file: str = None,
                 dask_nodes: str = None,
                 dask_nworkers: int = None,
                 dask_scheduler: str = None,
                 dask_scheduler_file: str = None,
                 dask_tcp_timeout: float = None, #float, int, str? prob in seconds
                 dask_connect_timeout: float = None, #float, int, str? prob in seconds
                 dask_malloc_trim_threshold: int = 0):
        pass
    
def start_imager(rawargs):
    parser = rascil_imager.cli_parser()
    args = parser.parse_args(rawargs)
    performance_environment(args.performance_file, mode="w")
    performance_store_dict(args.performance_file, "cli_args", vars(args), mode="a")
    image_name = rascil_imager.imager(args)

start_imager(
    [
        '--ingest_msname','visibilities_gleam.ms',
        '--ingest_dd', '0', 
        '--ingest_vis_nchan', '16',
        '--ingest_chan_per_blockvis', '1' ,
        '--ingest_average_blockvis', 'True',
        '--imaging_npixel', '2048', 
        '--imaging_cellsize', '3.878509448876288e-05',
        '--imaging_weighting', 'robust',
        '--imaging_robustness', '-0.5',
        '--clean_nmajor', '2' ,
        '--clean_algorithm', 'mmclean',
        '--clean_scales', '0', '6', '10', '30', '60',
        '--clean_fractional_threshold', '0.3',
        '--clean_threshold', '0.12e-3',
        '--clean_nmoment' ,'5',
        '--clean_psf_support', '640',
        '--clean_restored_output', 'integrated'
    ])