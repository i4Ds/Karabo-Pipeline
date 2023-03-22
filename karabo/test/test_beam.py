import os
import unittest
from datetime import datetime, timedelta
import numpy as np
import oskar

from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.test import data_path
from karabo.simulation.beam import BeamPattern
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

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
        vis_path = "./karabo/test/data/beam_vis"
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
            vis_path=vis_path + ".vis",
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
        visibility.write_to_file(path=vis_path + ".ms")

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
        vis_path = "./karabo/test/data/beam_vis"

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
            vis_path=vis_path + ".vis",
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
        visibility.write_to_file(path=vis_path + ".ms")

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


    def test_gauss(self):
        freq = 8.0e8
        precision = "double"
        beam_type = "Gaussian beam"
        vis_path = "./karabo/test/data/beam_vis_gauss"
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
            vis_path=vis_path + ".vis",
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
        visibility.write_to_file(path=vis_path + ".ms")

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
        dirty.write_to_file(path=vis_path+'.fits',overwrite=True)


    def convert_thetaphi_2_HV(self,theta,phi,amp_theta,amp_phi):
        H=amp_theta*np.cos(theta)-amp_phi*np.sin(theta)
        V=amp_theta*np.sin(theta)+amp_phi*np.cos(theta)
        return H,V
    def convert_HV_2_thetaphi(self,theta,phi,H,V):
        amp_theta=H*np.cos(theta) + V*np.sin(theta)
        amp_phi= V*np.sin(theta) - H*np.cos(theta)
        return amp_theta,amp_phi

    def make_cst_file(self):
        grid_th_phi=np.meshgrid(np.linspace(0,np.pi,180),np.linspace(0,2*np.pi,360))
        theta = np.ravel(grid_th_phi[0])
        phi = np.ravel(grid_th_phi[1])
        sigma = np.deg2rad(50)
        power_beam = np.exp(-(theta**2) / 2 / sigma**2)
        dtheta = np.max(np.diff(theta))
        dphi = phi[180] - phi[0]
        dsa=(dtheta * dphi * np.sin(theta))
        int_power_beam = np.sum(dsa*power_beam)
        norm_power_beam=power_beam/int_power_beam
        cross_power = theta**2 * norm_power_beam * np.cos(2 * phi + np.pi / 2)
        int_power_beam2=np.sum(dsa*power_beam**2)
        power_norm=np.sum(dsa*cross_power**2)
        rel_power_dB = -10
        cross_power *= (10 ** (rel_power_dB / 10) * int_power_beam2 / power_norm) ** 0.5
        cst_arr=np.zeros((len(power_beam),8))
        theta_deg=np.rad2deg(theta);phi_deg=np.rad2deg(phi)
        cst_arr[:,0] = theta_deg;cst_arr[:,1]=phi_deg
        cst_arr[:,3] = norm_power_beam; cst_arr[:,5] = cross_power
        cst_arr[:,4] = 205; cst_arr[:,6] = 45
        str1='Theta [deg.]  Phi   [deg.]  Abs(E   )[      ]   Abs(Theta)[      ]  Phase(Theta)[deg.]  Abs(Phi  )[      ]  Phase(Phi  )[deg.]  Ax.Ratio[      ] \n'
        str2='----------------------------------------------------------'
        filename='/home/rohit/karabo/karabo-pipeline/karabo/test/data/cst_X.txt'
        np.savetxt(filename,cst_arr,header=str1+str2,
                   comments='',fmt=['%12.4f', '%12.4f', '%20.6e', '%20.6e','%12.4f','%20.6e','%12.4f','%12.4f'])
        filename='/home/rohit/karabo/karabo-pipeline/karabo/test/data/cst_Y.txt'
        np.savetxt(filename,cst_arr,header=str1+str2,
                   comments='',fmt=['%12.4f', '%12.4f', '%20.6e', '%20.6e','%12.4f','%20.6e','%12.4f','%12.4f'])

    def plot_diff(self):
        aa=fits.open('/home/rohit/simulations/primary_beam/run9/test_cst_0.fits_I_no_beam.fits')
        bb=fits.open('/home/rohit/simulations/primary_beam/run9/test_cst_0.fits_I.fits')
        aa_data=aa[0].data[0];bb_data=bb[0].data[0]
        plt.imshow(aa_data-bb_data,aspect='auto',origin='lower',vmin=-1.e-2,vmax=1.e-2)
        plt.show()

    def cst_gauss(self):
        freq = 8.0e8
        precision = "double"
        #beam_type = "CST Gaussian"
        beam_type = "CST Dipole"
        #beam_type = "Gaussian beam"
        #beam_type = "Aperture array"
        vis_path = "./karabo/test/data/beam_vis"
        cst_path = "./karabo/test/data/"
        telescope = Telescope.get_MEERKAT_Telescope()
        sky = SkyModel()
        sky_data = np.zeros((81, 12))
        a = np.arange(-32, -27.5, 0.5)
        b = np.arange(18, 22.5, 0.5)
        dec_arr, ra_arr = np.meshgrid(a, b)
        sky_data[:, 0] = ra_arr.flatten()
        sky_data[:, 1] = dec_arr.flatten()
        sky_data[:, 2] = 1
        sky.add_point_sources(sky_data)
        # Remove beam if already present
        test = os.listdir(telescope.path)
        for item in test:
            if item.endswith(".bin"):
                os.remove(os.path.join(telescope.path, item))
        if(beam_type=='CST Gaussian'):
            gbx=BeamPattern(beam_method='Gaussian CST Beam',pol='X',cst_file_path=cst_path+'cst_X.txt',savecstx=True)
            gby=BeamPattern(beam_method='Gaussian CST Beam',pol='Y',cst_file_path=cst_path+'cst_Y.txt',savecsty=True)
            gcstx_beam=gbx.sim_beam();gcsty_beam=gby.sim_beam()
            #--------------------
            plot_beam=1
            if(plot_beam):
                f,axs=plt.subplots(2,2, subplot_kw={"projection": "polar"}, figsize=(8, 8))
                XX_ax, XY_ax, YX_ax, YY_ax = axs.flat
                imxx = XX_ax.pcolormesh(gcstx_beam[0][1].to("rad").value,
                    gcstx_beam[0][0].value,
                    gcstx_beam[3][:,3].reshape(180,360),
                    vmin=0.01,
                    vmax=1,
                )
                imxy = XY_ax.pcolormesh(gcsty_beam[0][1].to("rad").value,
                    gcsty_beam[0][0].value,
                    gcsty_beam[3][:,5].reshape(180,360),
                    vmin=0.01,
                    vmax=1,
                )
                imyx = YX_ax.pcolormesh(gcsty_beam[0][1].to("rad").value,
                    gcsty_beam[0][0].value,
                    gcsty_beam[4][:,3].reshape(180,360),
                    vmin=0.01,
                    vmax=1,
                )
                imyy = YY_ax.pcolormesh(gcsty_beam[0][1].to("rad").value,
                    gcsty_beam[0][0].value,
                    gcsty_beam[4][:,5].reshape(180,360),
                    vmin=0.01,
                    vmax=1,
                )
                plt.show()
            #gb.save_cst_file(gcst_beam)
            gbx.fit_elements(telescope, freq_hz=freq, avg_frac_error=0.5)
            gby.fit_elements(telescope, freq_hz=freq, avg_frac_error=0.5)
        if(beam_type=='CST Dipole'):
            gbx=BeamPattern(beam_method='Gaussian CST Beam',pol='X',cst_file_path=cst_path+'cst_X.txt',savecstx=True)
            gby=BeamPattern(beam_method='Gaussian CST Beam',pol='Y',cst_file_path=cst_path+'cst_Y.txt',savecsty=True)
            gcstx_beam=gbx.sim_beam();gcsty_beam=gby.sim_beam()
            #--------------------
            plot_beam=1
            if(plot_beam):
                f,axs=plt.subplots(2,2, subplot_kw={"projection": "polar"}, figsize=(8, 8))
                XX_ax, XY_ax, YX_ax, YY_ax = axs.flat
                imxx = XX_ax.pcolormesh(gcstx_beam[0][1].to("rad").value,
                    gcstx_beam[0][0].value,
                    gcstx_beam[3][:,3].reshape(180,360),
                    vmin=0.01,
                    vmax=1,
                )
                imxy = XY_ax.pcolormesh(gcsty_beam[0][1].to("rad").value,
                    gcsty_beam[0][0].value,
                    gcsty_beam[3][:,5].reshape(180,360),
                    vmin=0.01,
                    vmax=1,
                )
                imyx = YX_ax.pcolormesh(gcsty_beam[0][1].to("rad").value,
                    gcsty_beam[0][0].value,
                    gcsty_beam[4][:,3].reshape(180,360),
                    vmin=0.01,
                    vmax=1,
                )
                imyy = YY_ax.pcolormesh(gcsty_beam[0][1].to("rad").value,
                    gcsty_beam[0][0].value,
                    gcsty_beam[4][:,5].reshape(180,360),
                    vmin=0.01,
                    vmax=1,
                )
                plt.show()
            #gb.save_cst_file(gcst_beam)
            gbx.fit_elements(telescope, freq_hz=freq, avg_frac_error=0.5)
            gby.fit_elements(telescope, freq_hz=freq, avg_frac_error=0.5)
        # ------------- Simulation Begins
        simulation = InterferometerSimulation(
            vis_path=vis_path + ".vis",
            channel_bandwidth_hz=2e7,
            time_average_sec=8,
            noise_enable=False,
            ignore_w_components=True,
            precision=precision,
            use_gpus=False,
            station_type=beam_type,
            gauss_beam_fwhm_deg=1.0,
            gauss_ref_freq_hz=1.5e9,
            enable_numerical_beam=True
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
        visibility.write_to_file(path=vis_path + ".ms")

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
        #dirty.plot(title="Flux Density (Jy)")
        dirty.write_to_file(path=vis_path+'.fits',overwrite=True)

        plot_cst=1
        if(plot_cst):
            cstfile='/home/rohit/karabo/karabo-pipeline/karabo/test/data/cst_X.txt'
            cstbeam=np.loadtxt(cstfile,skiprows=2)
            x=cstbeam[:,0][0:45];y=cstbeam[:,3][0:45]
            f,ax=plt.subplots(1,1)
            ax.plot(x,y,'o-');ax.set_xlabel('$\\theta$ (deg)')
            ax.set_ylabel('Amp');plt.show()

        aa=fits.open(vis_path+'.fits');data_cst=aa[0].data[0][0]
        bb=fits.open(vis_path+'_gauss.fits');data_gauss=bb[0].data[0][0]
        cc=fits.open(vis_path+'_aperture.fits');data_aper=cc[0].data[0][0]
        f,((ax00,ax01),(ax10,ax11))=plt.subplots(2,2,sharex=True,sharey=True)
        ax00.imshow(data_cst,origin='lower',vmin=0.01,vmax=0.1)
        ax01.imshow(data_gauss,origin='lower',vmin=0.01,vmax=0.1)
        ax10.imshow(data_aper,origin='lower',vmin=0.01,vmax=0.1)
        ax11.imshow(data_aper-data_cst,origin='lower',vmin=-1.e-17,vmax=1.e-17)
        plt.show()




if __name__ == "__main__":
    unittest.main()
