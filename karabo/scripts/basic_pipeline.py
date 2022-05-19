from karabo.util.jupyter import setup_jupyter_env

setup_jupyter_env()
import numpy as np
from karabo.Imaging import imager
from karabo.simulation import sky_model, telescope, observation, interferometer

sky_data = np.array([
    [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
    [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
    [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10]])
sky = sky_model.SkyModel(sky_data)

ska_tel = telescope.get_SKA1_MID_Telescope()
observation_settings = observation.Observation(100e6, phase_centre_ra_deg=20, phase_centre_dec_deg=-30,
                                               number_of_channels=64, number_of_time_steps=24)

interferometer_sim = interferometer.InterferometerSimulation(channel_bandwidth_hz=1e6)
vis = interferometer_sim.run_simulation(ska_tel, sky, observation_settings)

imager_ska = imager.Imager(vis, imaging_npixel=2048,
                           imaging_cellsize=3.878509448876288e-05, ingest_vis_nchan=16, ingest_chan_per_blockvis=1)
deconvolved, restored, residual = imager_ska.imaging_rascil(clean_nmajor=0,
                                                            clean_algorithm='mmclean',
                                                            clean_scales=[0, 6, 10, 30, 60],
                                                            clean_fractional_threshold=.3, clean_threshold=.12e-3,
                                                            clean_nmoment=5, clean_psf_support=640,
                                                            clean_restored_output='integrated')
deconvolved.plot()
