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
import glob
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
import matplotlib.pyplot as plt
import rascil.processing_components.simulation.rfi as rf
from karabo.data.external_data import (
    GLEAMSurveyDownloadObject,
    MIGHTEESurveyDownloadObject,
)


class TestSystemNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists("result/system_noise"):
            os.makedirs("result/system_noise")

    def test_mightee_download(self):
        mightee1 = SkyModel.get_MIGHTEE_Sky()
        survey = MIGHTEESurveyDownloadObject()
        path = survey.get()
        mightee = SkyModel.get_fits_catalog(path)

    def test_mock_mightee(self):
        sky = SkyModel()
        start_time = time.time()
        mightee1 = SkyModel.get_MIGHTEE_Sky()
        # mightee0=fits.open('https://object.cscs.ch:443/v1/AUTH_1e1ed97536cf4e8f9e214c7ca2700d62/karabo_public/MIGHTEE_Continuum_Early_Science_COSMOS_Level1.fits');mightee_continuum=mightee0[1].data
        mightee_continuum = mightee1.to_array()
        sky_data = np.zeros((len(mightee_continuum), 12))
        sky_data[:, 0] = mightee_continuum[:, 0]
        sky_data[:, 1] = mightee_continuum[:, 1]
        sky_data[:, 2] = mightee_continuum[:, 2]
        sky_data[:, 6] = mightee_continuum[:, 6]
        sky_data[:, 9] = mightee_continuum[:, 9]
        sky_data[:, 10] = mightee_continuum[:, 10]
        sky_data[:, 11] = mightee_continuum[:, 11]
        sky.add_point_sources(sky_data)
        ra_list = [150.0, 150.5, 160.0]
        dec_list = [2.0, 2.5, 3.0]
        f_obs = 1.0e9
        chan = 1.0e7
        chan_bandwidth = 1.0e6
        for phase_ra in ra_list:
            for phase_dec in dec_list:
                sky_filter = sky.filter_by_radius(
                    ra0_deg=phase_ra,
                    dec0_deg=phase_dec,
                    inner_radius_deg=0,
                    outer_radius_deg=2.0,
                )
                telescope = Telescope.get_MEERKAT_Telescope()
                # telescope.centre_longitude = 3
                simulation = InterferometerSimulation(
                    channel_bandwidth_hz=chan_bandwidth,
                    time_average_sec=1,
                    noise_enable=False,
                    noise_seed="time",
                    noise_freq="Range",
                    noise_rms="Range",
                    noise_start_freq=f_obs,
                    noise_inc_freq=chan,
                    noise_number_freq=1,
                    noise_rms_start=5,
                    noise_rms_end=10,
                )
                observation = Observation(
                    phase_centre_ra_deg=phase_ra,
                    start_date_and_time=datetime(2022, 9, 1, 9, 00, 00, 521489),
                    length=timedelta(hours=10, minutes=0, seconds=1, milliseconds=0),
                    phase_centre_dec_deg=phase_dec,
                    number_of_time_steps=1,
                    start_frequency_hz=1.0e9 + chan,
                    frequency_increment_hz=chan,
                    number_of_channels=1,
                )
                visibility = simulation.run_simulation(
                    telescope, sky_filter, observation
                )
                visibility.write_to_file(
                    "./result/mock_mightee/mock_mightee_dec"
                    + str(phase_ra)
                    + "ra_"
                    + str(phase_dec)
                    + ".ms"
                )
                imager = Imager(
                    visibility, imaging_npixel=4096, imaging_cellsize=50
                )  # imaging cellsize is over-written in the Imager based on max uv dist.
                dirty = imager.get_dirty_image()
                dirty.write_to_file(
                    "result/mock_mightee/noise_dirty"
                    + str(phase_ra)
                    + "ra_"
                    + str(phase_dec)
                    + ".fits",
                    overwrite=True,
                )

        imglist = sorted(glob.glob("result/mock_mightee/noise_dirty1*.fits"))
        data = [0] * len(imglist)
        hdu = [0] * len(imglist)
        i = 0
        ff = [0] * len(imglist)
        for img in imglist:
            ff[i] = fits.open(img, memmap=True)
            data[i] = ff[i][0].data
            hdu[i] = ff[i][0].header
            i = i + 1
        mc_hdu = hdu[0]
        mc_array, footprint = reproject_interp(ff[1], mc_hdu)
        # mc_array, footprint = reproject_and_coadd((data[0], hdu[0]), mc_hdu, reproject_function=reproject_interp)
        mosaic_hdu = fits.PrimaryHDU(data=mc_array, header=mc_hdu)
        mosaic_hdu.writeto("result/mock_mightee/mosaic.fits", overwrite=True)
        f, ax = plt.subplots(2, 1)
        ax0 = ax[0]
        ax1 = ax[1]
        ax0.imshow(data[0][0][0])
        ax1.imshow(mc_array[0][0])
        plt.show()

        # cc=reproject_and_coadd(imglist[1:2],output_projection=hdu[0],reproject_function=reproject_interp)

        """      
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
        plt.legend();plt.show()"""
