import glob
import os
import tempfile
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from reproject import reproject_interp

from karabo.data.external_data import MIGHTEESurveyDownloadObject
from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


def test_mightee_download():
    _ = SkyModel.get_MIGHTEE_Sky()
    survey = MIGHTEESurveyDownloadObject()
    path = survey.get()
    _ = SkyModel.get_fits_catalog(path)


def test_mock_mightee():
    sky = SkyModel()
    mightee1 = SkyModel.get_MIGHTEE_Sky()
    mightee_continuum = mightee1.to_np_array()
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
    with tempfile.TemporaryDirectory() as tmpdir:
        for phase_ra in ra_list:
            for phase_dec in dec_list:
                sky_filter = sky.filter_by_radius(
                    ra0_deg=phase_ra,
                    dec0_deg=phase_dec,
                    inner_radius_deg=0,
                    outer_radius_deg=2.0,
                )
                telescope = Telescope.constructor("MeerKAT")
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
                # imaging cellsize is over-written in the Imager based on max uv dist.
                imager = Imager(visibility, imaging_npixel=4096, imaging_cellsize=50)
                dirty = imager.get_dirty_image()
                dirty.write_to_file(
                    os.path.join(tmpdir, "noise_dirty")
                    + str(phase_ra)
                    + "ra_"
                    + str(phase_dec)
                    + ".fits",
                    overwrite=True,
                )

        imglist = sorted(glob.glob(os.path.join(tmpdir, "noise_dirty1*.fits")))
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
        mc_array, _ = reproject_interp(ff[1], mc_hdu)
        mosaic_hdu = fits.PrimaryHDU(data=mc_array, header=mc_hdu)
        mosaic_hdu.writeto(os.path.join(tmpdir, "mosaic.fits"), overwrite=True)
        _, ax = plt.subplots(2, 1)
        ax0 = ax[0]
        ax1 = ax[1]
        ax0.imshow(data[0][0][0])
        ax1.imshow(mc_array[0][0])
        plt.show()
