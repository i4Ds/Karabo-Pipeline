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
        #dtime=datetime(2000, 1, 1, 8, 0, 00, 0) # SKA-mid
        tel_str='meerkat_32'
        telescope=Telescope.get_MEERKAT_Telescope()
        dtime=datetime(2000, 1, 1, 8, 0, 00, 0) # Meerkat
        #################################################################
        #tel_str='mwa'
        #telescope=Telescope.get_MWA_phase1_Telescope()
        #dtime=datetime(2000, 1, 1, 2, 0, 00, 0) # MWA
        #tel_str='lofar'
        #telescope=Telescope.get_LOFAR_Telescope()
        #dtime=datetime(2000, 1, 1, 9, 0, 00, 0) # lofar
        #tel_str='ska-low'
        #telescope=Telescope.get_SKA1_LOW_Telescope()
        #dtime=datetime(2000, 1, 1, 9, 0, 00, 0) # ska-low
        #tel_str='ngvlad'
        #telescope=Telescope.get_NG_VLAD_Telescope()
        #dtime=datetime(2000, 1, 1, 21, 0, 00, 0) # ngvlad
        #hour_=0;start_frequency_hz_=1.e8;flmax=2
        hour_=2;start_frequency_hz_=1.e9;flmax=20;npix=8
        obs='long'
        #-------------
        telescope_layout=np.loadtxt(telescope.path+'/layout.txt')
        aa=fits.open('/home/rohit/karabo/karabo-pipeline/karabo/test/data/solar/20151203_240MHz_psimas.fits')
        solar_map=aa[0].data;nx_,ny_=solar_map.shape
        solar_map=np.nanmean(solar_map.reshape(int(nx_/npix),npix,int(ny_/npix),npix),axis=(1,3))
        nx,ny=solar_map.shape
        solar_map_jy=solar_map/np.nanmax(solar_map)*flmax*1.e2
        ra_sun_center=249.141666667;dec_sun_center=21.986 #16 34 34.52 -21 59 09.7
        ra_grid,dec_grid=np.meshgrid((np.arange(nx)-int(nx/2))*22.5*npix/3600.,(np.arange(ny)-int(ny/2))*22.5*npix/3600.)
        ra_grid=ra_grid+ra_sun_center;dec_grid=dec_grid+dec_sun_center
        idx=np.where(solar_map>0.1*np.nanmax(solar_map))
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
        enable_numerical_beam=enable_array_beam,enable_array_beam=enable_array_beam)
        observation = Observation(mode='Tracking',phase_centre_ra_deg=ra_sun_center,
        start_date_and_time=dtime,
        length=timedelta(hours=hour_, minutes=1, seconds=0, milliseconds=0),
        phase_centre_dec_deg=dec_sun_center,
        number_of_time_steps=5,
        start_frequency_hz=start_frequency_hz_,
        frequency_increment_hz=1e6,
        number_of_channels=1, )
        visibility = simulation.run_simulation(telescope, sky, observation)
        visibility.write_to_file("./karabo/test/data/solar_"+tel_str+"0.ms")
        imager = Imager(visibility, imaging_npixel=2048,imaging_cellsize=1.2e-4) # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        dirty.write_to_file("./karabo/test/data/solar/solar_"+tel_str+"0.fits",overwrite=True)
        dirty.plot(title='Flux Density (Jy)',filename="./karabo/test/data/solar/solar_"+tel_str+"_"+obs+".png")

    def analysis(self):
        f256=fits.open('/home/rohit/karabo/karabo-pipeline/karabo/test/data/solar/solar_meerkat_2560.fits')
        f128=fits.open('/home/rohit/karabo/karabo-pipeline/karabo/test/data/solar/solar_meerkat_1280.fits')
        f64=fits.open('/home/rohit/karabo/karabo-pipeline/karabo/test/data/solar/solar_meerkat_640.fits')
        f256_data=f256[0].data;f128_data=f128[0].data;f64_data=f64[0].data
        plt.imshow(f256_data[0][0]-f128_data[0][0])
        plt.show()



if __name__ == "__main__":
    unittest.main()
