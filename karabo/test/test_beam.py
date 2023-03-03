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
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
import oskar


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



    def compare_karabo_oskar(self):
        # KARABO ----------------------------
        freq=8.0e8
        syarr=np.loadtxt('/home/rohit/simulations/primary_beam/jennifer_sky_model.txt')
        nsource=syarr.shape[0];sky_data=np.zeros((nsource,8))
        sky_data[:,0] = syarr[:,0];sky_data[:,1] = syarr[:,1];sky_data[:,2] = syarr[:,2]
        sky=SkyModel()
        sky.add_point_sources(sky_data)
        telescope = Telescope.get_MEERKAT_Telescope()
        # Remove beam if already present
        test = os.listdir(telescope.path)
        for item in test:
            if item.endswith(".bin"):
                os.remove(os.path.join(telescope.path, item))
        # ------------- Simulation Begins
        simulation = InterferometerSimulation(
            vis_path="./karabo/test/data/beam_vis.vis",
            channel_bandwidth_hz=2e7,
            time_average_sec=8,
            noise_enable=False,
            ignore_w_components=True,
            precision="single",
            use_gpus=False,
            station_type="Isotropic Beam"
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
        simulation.run_simulation(telescope, sky, observation)
        #visibility.write_to_file("./karabo/test/result/beam/beam_vis.ms")
        #-------------------------------------
        # OSKAR
        syarr=np.loadtxt('/home/rohit/simulations/primary_beam/jennifer_sky_model.txt')
        nsource=syarr.shape[0];sky_data=np.zeros((nsource,8))
        sky_data[:,0] = syarr[:,0];sky_data[:,1] = syarr[:,1];sky_data[:,2] = syarr[:,2]
        params = {
            "simulator": {
                "use_gpus": False
            },
            "observation": {
                "num_channels": 1,
                "start_frequency_hz": freq,
                "frequency_inc_hz": 2.e7,
                "phase_centre_ra_deg": 20,
                "phase_centre_dec_deg": -30,
                "num_time_steps": 10,
                "start_time_utc": "2000-03-20 12:06:39",
                "length": timedelta(hours=3, minutes=5, seconds=0, milliseconds=0)
            },
            "telescope": {
                "input_directory": '/home/rohit/karabo/karabo-pipeline/karabo/data/meerkat.tm',
                "station_type": "Isotropic beam",
                "pol_mode": "Full",
                "allow_station_beam_duplication": True,
                "normalise_beams_at_phase_centre": True,
            },
            "interferometer": {
                "oskar_vis_filename": "./karabo/test/data/beam_vis_oskar.vis",
                "ms_filename": "./karabo/test/result/beam/beam_vis_oskar.ms",
                "channel_bandwidth_hz": 2e7,
                "time_average_sec": 8,
                "ignore_w_components":True,
            }
        }
        settings = oskar.SettingsTree("oskar_sim_interferometer")
        settings.from_dict(params)
        # Set the numerical precision to use.
        precision = "single"
        if precision == "single":
            settings["simulator/double_precision"] = False

        # Create a sky model containing three sources from a numpy array.
        sky= oskar.Sky.load('/home/rohit/simulations/primary_beam/jennifer_sky_model.txt',precision)
        telescope = oskar.Telescope(settings=settings)
        # Set the sky model and run the simulation.
        sim = oskar.Interferometer(settings=settings)
        sim.set_sky_model(sky)
        sim.run()
        # Imaging
        imager_kb = oskar.Imager(precision)
        imager_kb.set(image_size=4096,fov_deg=5,weighting='Uniform', uv_filter_max=3000)
        imager_kb.set(input_file="./karabo/test/data/beam_vis.vis", output_root="./karabo/test/result/beam/beam_vis")
        output_kb = imager_kb.run(return_images=1)
        image_karabo = output_kb["images"][0]
        #aa=fits.open('./karabo/test/result/beam/beam_vis_I.fits');image_karabo=aa[0].data[0][0]
        imager_os = oskar.Imager(precision)
        imager_os.set(image_size=4096,fov_deg=5,weighting='Uniform', uv_filter_max=3000)
        imager_os.set(input_file="./karabo/test/data/beam_vis_oskar.vis", output_root="./karabo/test/result/beam/beam_vis_oskar")
        output_os = imager_os.run(return_images=1)
        image_oskar = output_os["images"][0]
        plt.imshow(image_oskar-image_karabo,aspect='auto',origin='lower',cmap='jet')
        plt.colorbar()
        plt.show()

    def create_random_sky(self):
        nsource=16;nsource_side=int(np.sqrt(nsource))
        random=0
        if(random):
            ra_array_side=19+np.random.random(nsource_side)*(21-19)
            dec_array_side=-29+np.random.random(nsource_side)*(-31+29)
        else:
            ra_array_side=19+np.arange(nsource_side)*(21-19)
            dec_array_side=-29+np.arange(nsource_side)*(-31+29)
        radec_grid=np.meshgrid(ra_array_side,dec_array_side)
        ra_array=radec_grid[0].flatten();dec_array=radec_grid[1].flatten()
        s_array=np.ones(nsource)*10;freq_array=freq*np.ones(nsource)
        sky_data=np.zeros((nsource,8));sky_data[:,0] = ra_array;sky_data[:,1] = dec_array;sky_data[:,2] = s_array;sky_data[:,6] = freq_array
        #sky.add_point_sources(sky_data)
        return  sky_data


    def test_beam(self):
        #--------------------------
        freq=8.0e8
        syarr=np.loadtxt('/home/rohit/simulations/primary_beam/jennifer_sky_model.txt')
        nsource=syarr.shape[0];sky_data=np.zeros((nsource,8))
        sky_data[:,0] = syarr[:,0];sky_data[:,1] = syarr[:,1];sky_data[:,2] = syarr[:,2]
        sky=SkyModel()
        sky.add_point_sources(sky_data)
        telescope = Telescope.get_MEERKAT_Telescope()
        # Remove beam if already present
        test = os.listdir(telescope.path)
        for item in test:
            if item.endswith(".bin"):
                os.remove(os.path.join(telescope.path, item))
        # ------------- Simulation Begins
        simulation = InterferometerSimulation(
            vis_path="./karabo/test/data/beam_vis.vis",
            channel_bandwidth_hz=2e7,
            time_average_sec=8,
            noise_enable=False,
            ignore_w_components=False,
            precision="single",
            use_gpus=False,
            station_type="Gaussian beam", # 'Isotropic beam' / 'Gaussian beam' / 'Aperture array'
            gauss_beam_fwhm_deg=2.0,
            gauss_ref_freq_hz=freq,

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
        visibility=simulation.run_simulation(telescope, sky, observation)
        visibility.write_to_file(path="./karabo/test/data/beam_vis.ms")
        #-------------------------------------
        # OSKAR IMAGING
        precision='double'
        imager_kb = oskar.Imager(precision)
        imager_kb.set(image_size=4096,fov_deg=5,weighting='Uniform', uv_filter_max=3000)
        imager_kb.set(input_file="./karabo/test/data/beam_vis.vis", output_root="./karabo/test/result/beam/beam_vis")
        output_kb = imager_kb.run(return_images=1)
        image_karabo = output_kb["images"][0]
        plt.imshow(image_karabo,aspect='auto',origin='lower')
        plt.show()
        #-------------------------------------
        # RASCIL IMAGING
        uvmax=3000/(3.e8/freq) # in wavelength units
        imager = Imager(
            visibility,
            imaging_npixel=4096,
            imaging_cellsize=2.13e-5,
            imaging_dopsf=True,
            imaging_weighting='uniform',
            imaging_uvmax=uvmax,
            imaging_uvmin=1,
        )  # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        dirty.write_to_file("./karabo/test/result/beam/beam_vis_rascil.fits", overwrite=True)
        dirty.plot(title="Flux Density (Jy)")
        #-------------------------------------
        plot_diff=0
        if(plot_diff):
            ab = fits.open("./karabo/test/result/beam/beam_vis.fits")
            a = fits.open("./karabo/test/result/beam/beam_vis_no_beam_4096.fits")
            adiff = ab[0].data[0][0] - a[0].data[0][0]
            wcs = WCS(a[0].header)
            f,ax=plt.subplots(subplot_kw=dict(projection=wcs,slices=['x','y',0,0]))
            im=ax.imshow(adiff,aspect='auto',origin='lower',vmin=-2e0,vmax=2.e0)
            ellipse = Ellipse(xy=(400 , 400), width=405, height=405,
                              edgecolor='r', fc='None', lw=2,alpha=0.5)
            ax.add_patch(ellipse)
            f.colorbar(im)
            f.show()



if __name__ == "__main__":
    unittest.main()
