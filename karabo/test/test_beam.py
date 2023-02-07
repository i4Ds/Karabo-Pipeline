import os
import unittest
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.beam import BeamPattern
from karabo.simulation.telescope import Telescope
from karabo.test import data_path
from karabo.simulation.sky_model import SkyModel
import numpy as np
from karabo.simulation.observation import Observation
from datetime import timedelta, datetime
from karabo.imaging.imager import Imager
from astropy.io import fits


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
            beampixels[0], 40, 800, "I", path="./result/katbeam_beam.png"
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

    def test_beam(self):
        sky = SkyModel()
        sky_data = np.array(
            [
                [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
                [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
                [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10],
            ]
        )
        sky.add_point_sources(sky_data)
        telescope = Telescope.get_MEERKAT_Telescope()
        # telescope.centre_longitude = 3
        xcstfile_path = "./karabo/test/data/cst_like_beam_port_1.txt"
        ycstfile_path = "./karabo/test/data/cst_like_beam_port_2.txt"
        #xcstfile_path = '/home/jennifer/Karabo-Pipeline/karabo/test/data/cst_like_beam_port_1.txt'
        #ycstfile_path = '/home/jennifer/Karabo-Pipeline/karabo/test/data/cst_like_beam_port_2.txt'
        enable_array_beam = True
        # Remove beam if already present
        test = os.listdir(telescope.path)
        for item in test:
            if item.endswith(".bin"):
                os.remove(os.path.join(telescope.path, item))
        if enable_array_beam:
            # ------------ X-coordinate
            pb = BeamPattern(xcstfile_path)  # Instance of the Beam class
            beam = pb.sim_beam(beam_method="KatBeam", f=1000,fov=30)  # Computing beam
            pb.save_meerkat_cst_file(beam[3])  # Saving the beam cst file
            pb.fit_elements(
                telescope, freq_hz=1.0e9, avg_frac_error=0.005, pol="X"
            )  # Fitting the beam using cst file
            # ------------ Y-coordinate
            pb = BeamPattern(ycstfile_path)
            pb.save_meerkat_cst_file(beam[4])
            pb.fit_elements(telescope, freq_hz=1.0e9, avg_frac_error=0.005, pol="Y")
        # ------------- Simulation Begins
        simulation = InterferometerSimulation(
            vis_path="./karabo/test/data/beam_vis.vis",
            channel_bandwidth_hz=2e7,
            time_average_sec=1,
            noise_enable=False,
            noise_seed="time",
            noise_freq="Range",
            noise_rms="Range",
            noise_start_freq=1.0e9,
            noise_inc_freq=1.0e8,
            noise_number_freq=24,
            noise_rms_start=0,
            noise_rms_end=0,
            enable_numerical_beam=enable_array_beam,
            enable_array_beam=enable_array_beam,
        )
        observation = Observation(
            mode="Tracking",
            phase_centre_ra_deg=20.0,
            start_date_and_time=datetime(2000, 1, 1, 13, 00, 00, 521489),
            length=timedelta(hours=0, minutes=0, seconds=1, milliseconds=0),
            phase_centre_dec_deg=-30.0,
            number_of_time_steps=1,
            start_frequency_hz=1.0e9,
            frequency_increment_hz=1e6,
            number_of_channels=1,
        )
        #
        visibility = simulation.run_simulation(telescope, sky, observation)
        # visibility.write_to_file("./test/result/beam/beam_vis.ms")

        # imager = Imager(visibility, imaging_npixel=4096,imaging_cellsize=50) # imaging cellsize is over-written in the Imager based on max uv dist.
        # dirty = imager.get_dirty_image()
        # dirty.write_to_file("./test/result/beam/beam_vis.fits")
        # dirty.plot(title='Flux Density (Jy)')
        # aa=fits.open('./result/beam/beam_vis.fits');bb=fits.open('/home/rohit/karabo/karabo-pipeline/karabo/test/result/beam/beam_vis_aperture.fits')
        # print(np.nanmax(aa[0].data-bb[0].data),np.nanmax(aa[0].data),np.nanmax(bb[0].data))


if __name__ == "__main__":
    unittest.main()
