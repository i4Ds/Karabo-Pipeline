# Intensity Mapping Sample Script

from karabo.simulation import telescope
from karabo.simulation import observation
from karabo.simulation import sky_model
from karabo.simulation.interferometer import InterferometerSimulation
import numpy as np
import oskar
import concurrent




def get_params(ra,dec,filename):
    '''
    Input: RA, DEC, output file
    Output: Param dictionary
    '''
    params = {
        "simulator": {
            "use_gpus": "False"
        },
        "observation": {
            "num_channels": "64",
            "start_frequency_hz": "100e6",
            "frequency_inc_hz": "20e6",
            "phase_centre_ra_deg": str(ra),
            "phase_centre_dec_deg": str(dec),
            "num_time_steps": "24",
            "start_time_utc": "01-01-2000 12:00:00.000",
            "length": "12:00:00.000"
        },
        "telescope": {
            #"input_directory": tel,
            "input_directory": '/home/rohit/simulations/meerKat/meerkat_core.tm',
            "gaussian_beam/fwhm_deg": "10.0",
            "gaussian_beam/ref_freq_hz":"8.e8"
        },
        "interferometer": {
            #"ms_filename": "visibilities.ms",
            "oskar_vis_filename":str(filename)+".vis",
            "channel_bandwidth_hz": "1e6",
            "time_average_sec": "10"
        }
    }
    return params

ra_deg=np.array([20,21,22]);dec_array=([-30,-30,-30]);I=1
sky_data = np.array([[20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0]])
precision='double'
tel=telescope.get_MEERKAT_Telescope()
#ss=sky_model.SkyModel.to_array(sky_data,precision)
ss=oskar.Sky.from_array(sky_data, precision)
sky_model.SkyModel.to_array(sky_data)
settings = oskar.SettingsTree("oskar_sim_interferometer")
vis=[0]*len(ra_deg);auto_vis=[0]*len(ra_deg)
for i in range(len(ra_deg)):
    params=get_params(ra_deg[i],dec_array[i],"output_"+str(i))
    settings.from_dict(params)
    if precision == "single":
        settings["simulator/double_precision"] = "False"
    # Set the sky model and run the simulation.
    sim = oskar.Interferometer(settings=settings)
    sim.set_sky_model(ss)
    sim.run()
    (header, handle) = oskar.VisHeader.read('output_'+str(i)+'.vis')
    block = oskar.VisBlock.create_from_header(header);tasks_read = []
    executor = concurrent.futures.ThreadPoolExecutor(2)
    for i_block in range(header.num_blocks):
        tasks_read.append(executor.submit(block.read, header, handle,i_block))
    vis[i] = block.cross_correlations()
    #auto_vis[i]=block.auto_correlations()
vis_comb=np.array(vis).reshape(vis[0].shape[0]*len(ra_deg),vis[0].shape[1],vis[0].shape[2],vis[0].shape[3])
#sim.write_block(block,1)


def start_imager(rawargs):
    parser = rascil_imager.cli_parser()
    args = parser.parse_args(rawargs)
    rascil_imager.performance_environment(args.performance_file, mode="w")
    rascil_imager.performance_store_dict(args.performance_file, "cli_args", vars(args), mode="a")
    image_name = rascil_imager.imager(args)

image=0
if image:
    start_imager([
        '--ingest_msname', 'visibilities.ms',
        '--ingest_dd', '0',
        '--ingest_vis_nchan', '64',
        '--ingest_chan_per_blockvis', '4',
        '--ingest_average_blockvis', 'True',
        '--imaging_npixel', '2048',
        '--imaging_cellsize', '3.878509448876288e-05',
        '--imaging_weighting', 'robust',
        '--imaging_robustness', '-0.5',
        '--clean_nmajor', '5',
        '--clean_algorithm', 'mmclean',
        '--clean_scales', '0', '6', '10', '30', '60',
        '--clean_fractional_threshold', '0.3',
        '--clean_threshold', '0.12e-3',
        '--clean_nmoment', '5',
        '--clean_psf_support', '640',
        '--clean_restored_output', 'integrated'])


