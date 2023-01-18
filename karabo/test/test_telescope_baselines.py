import os
import unittest
from karabo.simulation.telescope import Telescope
from karabo.simulation.telescope import create_baseline_cut_telelescope
import numpy as np
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from datetime import timedelta, datetime
from karabo.imaging.imager import Imager
from math import comb
from karabo.simulation.telescope_versions import ALMAVersions, ACAVersions, CARMAVersions, NGVLAVersions, \
    PDBIVersions, SMAVersions, VLAVersions



class MyTestCase(unittest.TestCase):

    def test_baselines_based_cutoff(self):
        lcut=5000;hcut=10000 #Lower cut off and higher cut-off in meters
        parant_tel=Telescope.get_MEERKAT_Telescope()
        telescope_path=create_baseline_cut_telelescope(lcut,hcut,parant_tel)
        telescope=Telescope.read_OSKAR_tm_file(telescope_path)
        sky = SkyModel()
        sky_data = np.array([
              [20.0, -30.0, 100, 0, 0, 0, 1.0e9, -0.7, 0.0, 0, 0, 0],
              [20.0, -30.5, 100, 2, 2, 0, 1.0e9, -0.7, 0.0, 0, 50, 45],
              [20.5, -30.5, 100, 0, 0, 2, 1.0e9, -0.7, 0.0, 0, 10, -10]])
        sky.add_point_sources(sky_data)
        simulation = InterferometerSimulation(channel_bandwidth_hz=1e6,
                                               time_average_sec=1, noise_enable=False,
                                                noise_seed="time", noise_freq="Range", noise_rms="Range",
                                                noise_start_freq=1.e9,
                                                noise_inc_freq=1.e8,
                                                noise_number_freq=24,
                                                noise_rms_start=5000,
                                                noise_rms_end=10000)
        observation = Observation(phase_centre_ra_deg=20.0,
                                   start_date_and_time=datetime(2022, 1, 1, 11, 00, 00, 521489),
                                   length=timedelta(hours=0, minutes=0, seconds=1, milliseconds=0),
                                   phase_centre_dec_deg=-30.5,
                                   number_of_time_steps=1,
                                   start_frequency_hz=1.e9,
                                   frequency_increment_hz=1e6,
                                   number_of_channels=1,)

        visibility = simulation.run_simulation(telescope, sky, observation)
        #visibility = simulation.run_simulation(parant_tel, sky, observation)
        visibility.copy_image_file_to("./result/baseline_cut.ms")

        imager = Imager(visibility,
                         imaging_npixel=4096*1,
                         imaging_cellsize=50) # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        dirty.copy_image_file_to("result/baseline_cut.fits")
        dirty.plot(title='Flux Density (Jy)')











