import os
import unittest
from datetime import timedelta, datetime
import numpy as np
from karabo.imaging.imager import Imager
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.interferometer import InterferometerSimulation
from astropy.io import fits
import time
import matplotlib.pyplot as plt
import rascil.processing_components.simulation.rfi as rf


class TestSystemNoise(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists('result/system_noise'):
            os.makedirs('result/system_noise')

    def test_mock_mightee(self):
        sky = SkyModel()
        start_time = time.time()
        mightee0=fits.open('/home/rohit/simulations/MIGHTEE/MIGHTEE_Continuum_Early_Science_COSMOS_Level1.fits');mightee_continuum=mightee0[1].data
        ra=mightee_continuum['RA'];dec=mightee_continuum['DEC'];name=mightee_continuum['NAME'];s_peak = mightee_continuum['S_PEAK'];f_eff=mightee_continuum['NU_EFF']
        im_maj=mightee_continuum['IM_MAJ'];im_min=mightee_continuum['IM_MIN'];im_pa=mightee_continuum['IM_PA']
        sky_data=np.zeros((len(ra),12));sky_data[:,0]=ra;sky_data[:,1]=dec;sky_data[:,2]=s_peak;sky_data[:,6]=f_eff;sky_data[:,9]=im_maj;sky_data[:,10]=im_min;sky_data[:,11]=im_pa
        sky.add_point_sources(sky_data)
        phase_ra=150.0;phase_dec=2.2;phasecenter=(phase_ra,phase_dec)
        f_obs=1.e9;chan=1.e7
        sky_filter=sky.filter_by_radius(ra0_deg=phase_ra,dec0_deg=phase_dec,inner_radius_deg=0,outer_radius_deg=2.0)
        telescope = Telescope.get_MEERKAT_Telescope()
        # telescope.centre_longitude = 3
        simulation = InterferometerSimulation(channel_bandwidth_hz=1e6,
                                               time_average_sec=1, noise_enable=True,
                                                noise_seed="time", noise_freq="Range", noise_rms="Range",
                                                noise_start_freq=f_obs,
                                                noise_inc_freq=chan,
                                                noise_number_freq=1,
                                                noise_rms_start=5,
                                                noise_rms_end=10)
        observation = Observation(phase_centre_ra_deg=phase_ra,
                                   start_date_and_time=datetime(2022, 9, 1, 9, 00, 00, 521489),
                                   length=timedelta(hours=10, minutes=0, seconds=1, milliseconds=0),
                                   phase_centre_dec_deg=phase_dec,
                                   number_of_time_steps=1,
                                   start_frequency_hz=1.e9+chan,
                                   frequency_increment_hz=chan,
                                   number_of_channels=1,)

        visibility = simulation.run_simulation(telescope, sky_filter, observation)
        time_vis = (time.time() - start_time)
        visibility.write_to_file("./result/mock_mightee/mock_mightee.ms")
        time_vis_write=(time.time() - start_time)
        imager = Imager(visibility,
                         imaging_npixel=4096,
                         imaging_cellsize=50) # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        dirty.write_to_file("result/mock_mightee/noise_dirty.fits")
        time_end=(time.time() - start_time)
        print(time_vis,time_vis_write,time_end)
        dirty.plot(title='Flux Density (Jy)',vmin=0,vmax=0.5)
        plt.plot([1,10,30,60,80,100],[20.5,22.4,30.2,40.3,42.2,44.4],'o-',label='Vis Run')
        plt.plot([1, 10, 30, 60, 80, 100], [24.3,46.5,148.2,247.2,373.6,537.7], 'o-', label='Vis + Image Run')
        plt.xlabel('Number of Channels');plt.ylabel('Execution Time (sec)')
        plt.legend();plt.show()

