import os
import unittest
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import oskar

from karabo.imaging.imager import Imager
from karabo.simulation.beam import BeamPattern
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.test import data_path
from astropy.io import fits


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists("result/"):
            os.makedirs("result/")


    def test_solar(self):
        #tel_str='vla'
        #telescope = Telescope.get_VLA_Telescope(version='c')
        #dtime=datetime(2000, 1, 1, 21, 0, 00, 0) # VLA
        #tel_str='ska_mid'
        #telescope=Telescope.get_SKA1_MID_Telescope()
        #dtime=datetime(2000, 1, 1, 7, 0, 00, 0) # SKA-mid
        #tel_str='meerkat'
        #telescope=Telescope.get_MEERKAT_Telescope()
        #dtime=datetime(2000, 1, 1, 7, 0, 00, 0) # Meerkat
        # tel_str='mwa'
        # telescope=Telescope.get_MWA_phase1_Telescope()
        # dtime=datetime(2000, 1, 1, 2, 0, 00, 0) # MWA
        # tel_str='mwa'
        # telescope=Telescope.get_MWA_phase1_Telescope()
        # dtime=datetime(2000, 1, 1, 2, 0, 00, 0) # MWA
        # tel_str='lofar'
        # telescope=Telescope.get_LOFAR_Telescope()
        # dtime=datetime(2000, 1, 1, 7, 0, 00, 0) # lofar
        tel_str='ska-low'
        telescope=Telescope.get_SKA1_LOW_Telescope()
        dtime=datetime(2000, 1, 1, 7, 0, 00, 0) # lofar
        hour_=1
        start_frequency_hz_=1.e8
        #-------------
        aa=fits.open('/home/rohit/karabo/karabo-pipeline/karabo/test/data/solar/20151203_240MHz_psimas.fits')
        solar_map=aa[0].data;solar_map_jy=solar_map/np.nanmax(solar_map)*20*1.e2
        ra_sun_center=249.141666667;dec_sun_center=21.986 #16 34 34.52 -21 59 09.7
        ra_grid,dec_grid=np.meshgrid((np.arange(256)-128)*22.5/3600.,(np.arange(256)-128)*22.5/3600.)
        ra_grid=ra_grid+ra_sun_center;dec_grid=dec_grid+dec_sun_center
        idx=np.where(solar_map>0.001*np.nanmax(solar_map))
        sky_model_ra=ra_grid[idx];sky_model_dec=dec_grid[idx];flux=solar_map_jy[idx]
        print('Number of Sources in the Skymodel: ',sky_model_ra.shape[0])
        sky = SkyModel()
        sky_data = np.array([sky_model_ra, sky_model_dec, flux,np.zeros(len(flux)), \
         np.zeros(len(flux)),np.zeros(len(flux)), np.zeros(len(flux)),np.zeros(len(flux)), \
        np.zeros(len(flux)),np.zeros(len(flux)), np.zeros(len(flux)),np.zeros(len(flux))]).T
        sky.add_point_sources(sky_data)
        enable_array_beam=False
        #------------- Simulation Begins
        simulation = InterferometerSimulation(vis_path='./karabo/test/data/solar/solar_'+tel_str+'0.vis',
        channel_bandwidth_hz=2e7,
        time_average_sec=1, noise_enable=False,
        noise_seed="time", noise_freq="Range", noise_rms="Range",
        noise_start_freq=1.e9,
        noise_inc_freq=1.e8,
        noise_number_freq=24,
        noise_rms_start=0,
        noise_rms_end=0,
        enable_numerical_beam=enable_array_beam,enable_array_beam=enable_array_beam)
        observation = Observation(mode='Tracking',phase_centre_ra_deg=ra_sun_center,
        start_date_and_time=dtime,
        length=timedelta(hours=hour_, minutes=1, seconds=0, milliseconds=0),
        phase_centre_dec_deg=dec_sun_center,
        number_of_time_steps=100,
        start_frequency_hz=start_frequency_hz_,
        frequency_increment_hz=1e6,
        number_of_channels=1, )
        visibility = simulation.run_simulation(telescope, sky, observation)
        visibility.write_to_file("./karabo/test/data/solar_"+tel_str+"0.ms")
        imager = Imager(visibility, imaging_npixel=2048,imaging_cellsize=1.2e-4) # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        dirty.write_to_file("./karabo/test/data/solar/solar_"+tel_str+"0.fits",overwrite=True)
        dirty.plot(title='Flux Density (Jy)')



if __name__ == "__main__":
    unittest.main()
