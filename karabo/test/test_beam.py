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


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists("result/"):
            os.makedirs("result/")

    def test_fit_element(self):
        tel = Telescope.get_MEERKAT_Telescope()
        beam = BeamPattern(f"{data_path}/run5.cst")
        beam.fit_elements(tel, freq_hz=1.0e08, avg_frac_error=0.5)

    def test_katbeam(self):
        beampixels = BeamPattern.get_meerkat_uhfbeam(
            f=800, pol="I", beamextentx=40, beamextenty=40
        )
        BeamPattern.show_kat_beam(
            beampixels[2], 40, 800, "I", path="./result/katbeam_beam.png"
        )

    def test_eidosbeam(self):
        npix = 500
        dia = 10
        thres = 0
        ch = 0
        B_ah = BeamPattern.get_eidos_holographic_beam(
            npix=npix, ch=ch, dia=dia, thres=thres, mode="AH"
        )
        BeamPattern.show_eidos_beam(B_ah, path="./result/eidos_AH_beam.png")
        B_em = BeamPattern.get_eidos_holographic_beam(
            npix=npix, ch=ch, dia=dia, thres=thres, mode="EM"
        )
        BeamPattern.show_eidos_beam(B_em, path="./result/eidos_EM_beam.png")
        BeamPattern.eidos_lineplot(
            B_ah, B_em, npix, path="./result/eidos_residual_beam.png"
        )

    def test_compare_karabo_oskar(self):
        """
        We test the that oskar and karabo give the same output when
        using the same imager -> resulting plot should be zero everywhere.
        """
        # KARABO ----------------------------
        freq = 8.0e8
        precision = "single"
        beam_type = "Isotropic beam"
        vis_path = "./karabo/test/data/beam_vis.vis"
        ms_file_path = "./karabo/test/data/beam_vis.ms"
        sky_txt = "./karabo/test/data/sky_model.txt"
        telescope_tm = "./karabo/data/meerkat.tm"

        sky = SkyModel()
        sky_data = np.zeros((81, 12))
        a = np.arange(-32, -27.5, 0.5)
        b = np.arange(18, 22.5, 0.5)
        dec_arr, ra_arr = np.meshgrid(a, b)
        sky_data[:, 0] = ra_arr.flatten()
        sky_data[:, 1] = dec_arr.flatten()
        sky_data[:, 2] = 1

        sky.add_point_sources(sky_data)

        telescope = Telescope.get_MEERKAT_Telescope()
        # Remove beam if already present
        test = os.listdir(telescope.path)
        for item in test:
            if item.endswith(".bin"):
                os.remove(os.path.join(telescope.path, item))
        # ------------- Simulation Begins
        simulation = InterferometerSimulation(
            vis_path=vis_path,
            ms_file_path=ms_file_path,
            channel_bandwidth_hz=2e7,
            time_average_sec=8,
            noise_enable=False,
            ignore_w_components=True,
            precision=precision,
            use_gpus=False,
            station_type=beam_type,
            gauss_beam_fwhm_deg=1.0,
            gauss_ref_freq_hz=1.5e9,
        )
        observation = Observation(
            mode="Tracking",
            phase_centre_ra_deg=20.0,
            start_date_and_time=datetime(2000, 3, 20, 12, 6, 39, 0),
            length=timedelta(hours=3, minutes=5, seconds=0, milliseconds=0),
            phase_centre_dec_deg=-30.0,
            number_of_time_steps=10,
            start_frequency_hz=freq,
            frequency_increment_hz=2e7,
            number_of_channels=1,
        )
        visibility = simulation.run_simulation(telescope, sky, observation)
        print(visibility.ms_file.path)

        # RASCIL IMAGING
        uvmax = 3000 / (3.0e8 / freq)  # in wavelength units
        imager = Imager(
            visibility,
            imaging_npixel=4096,
            imaging_cellsize=2.13e-5,
            imaging_dopsf=True,
            imaging_weighting="uniform",
            imaging_uvmax=uvmax,
            imaging_uvmin=1,
        )  # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        image_karabo = dirty.data[0][0]

        # OSKAR -------------------------------------

        # Setting tree
        params = {
            "simulator": {"use_gpus": True},
            "observation": {
                "num_channels": 1,
                "start_frequency_hz": freq,
                "frequency_inc_hz": 2e7,
                "phase_centre_ra_deg": 20,
                "phase_centre_dec_deg": -30,
                "num_time_steps": 10,
                "start_time_utc": "2000-03-20 12:06:39",
                "length": "03:05:00.000",
            },
            "telescope": {
                "input_directory": telescope_tm,
                "normalise_beams_at_phase_centre": True,
                "pol_mode": "Full",
                "allow_station_beam_duplication": True,
                "station_type": beam_type,
                "gaussian_beam/fwhm_deg": 1,
                "gaussian_beam/ref_freq_hz": 1.5e9,  # Mid-frequency in
                # the redshift range
            },
            "interferometer": {
                "oskar_vis_filename": vis_path + ".vis",
                "channel_bandwidth_hz": 2e7,
                "time_average_sec": 8,
                "ignore_w_components": True,
            },
        }

        settings = oskar.SettingsTree("oskar_sim_interferometer")
        settings.from_dict(params)

        # Choose the numerical precision
        if precision == "single":
            settings["simulator/double_precision"] = False

        # The following line depends on the mode with which we're loading the sky
        # (explained in documentation)
        np.savetxt(sky_txt, sky.sources[:, :3])
        sky_sim = oskar.Sky.load(sky_txt, precision)

        sim = oskar.Interferometer(settings=settings)
        sim.set_sky_model(sky_sim)
        sim.run()

        # RASCIL IMAGING
        uvmax = 3000 / (3.0e8 / freq)  # in wavelength units
        imager = Imager(
            visibility,
            imaging_npixel=4096,
            imaging_cellsize=2.13e-5,
            imaging_dopsf=True,
            imaging_weighting="uniform",
            imaging_uvmax=uvmax,
            imaging_uvmin=1,
        )  # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        image_oskar = dirty.data[0][0]

        # Plotting the difference between karabo and oskar using oskar imager
        # -> should be zero everywhere
        plt.imshow(
            image_karabo - image_oskar, aspect="auto", origin="lower", cmap="jet"
        )
        plt.colorbar()
        plt.show()

    def create_random_sky(self):
        freq = 8.0e8
        nsource = 16
        nsource_side = int(np.sqrt(nsource))
        random = 0
        if random:
            ra_array_side = 19 + np.random.random(nsource_side) * (21 - 19)
            dec_array_side = -29 + np.random.random(nsource_side) * (-31 + 29)
        else:
            ra_array_side = 19 + np.arange(nsource_side) * (21 - 19)
            dec_array_side = -29 + np.arange(nsource_side) * (-31 + 29)
        radec_grid = np.meshgrid(ra_array_side, dec_array_side)
        ra_array = radec_grid[0].flatten()
        dec_array = radec_grid[1].flatten()
        s_array = np.ones(nsource) * 10
        freq_array = freq * np.ones(nsource)
        sky_data = np.zeros((nsource, 8))
        sky_data[:, 0] = ra_array
        sky_data[:, 1] = dec_array
        sky_data[:, 2] = s_array
        sky_data[:, 6] = freq_array
        # sky.add_point_sources(sky_data)
        return sky_data

    def test_gaussian_beam(self):
        """
        We test that image reconstruction works also with a Gaussian beam and
        test both Imagers: Oskar and Rascil.
        """
        # --------------------------
        freq = 8.0e8
        precision = "double"
        beam_type = "Gaussian beam"
        vis_path = "./karabo/test/data/beam_vis.vis"
        ms_path = "./karabo/test/data/beam_vis.ms"

        sky = SkyModel.sky_test()

        telescope = Telescope.get_MEERKAT_Telescope()
        # Remove beam if already present
        test = os.listdir(telescope.path)
        for item in test:
            if item.endswith(".bin"):
                os.remove(os.path.join(telescope.path, item))
        # ------------- Simulation Begins
        simulation = InterferometerSimulation(
            vis_path=vis_path,
            ms_file_path=ms_path,
            channel_bandwidth_hz=2e7,
            time_average_sec=8,
            noise_enable=False,
            ignore_w_components=True,
            precision=precision,
            use_gpus=False,
            station_type=beam_type,
            gauss_beam_fwhm_deg=1.0,
            gauss_ref_freq_hz=1.5e9,
        )
        observation = Observation(
            mode="Tracking",
            phase_centre_ra_deg=20.0,
            start_date_and_time=datetime(2000, 3, 20, 12, 6, 39, 0),
            length=timedelta(hours=3, minutes=5, seconds=0, milliseconds=0),
            phase_centre_dec_deg=-30.0,
            number_of_time_steps=10,
            start_frequency_hz=freq,
            frequency_increment_hz=2e7,
            number_of_channels=1,
        )
        visibility = simulation.run_simulation(telescope, sky, observation)

        # RASCIL IMAGING
        uvmax = 3000 / (3.0e8 / freq)  # in wavelength units
        imager = Imager(
            visibility,
            imaging_npixel=4096,
            imaging_cellsize=2.13e-5,
            imaging_dopsf=True,
            imaging_weighting="uniform",
            imaging_uvmax=uvmax,
            imaging_uvmin=1,
        )  # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        dirty.plot(title="Flux Density (Jy)")


if __name__ == "__main__":
    unittest.main()
