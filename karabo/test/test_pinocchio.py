import os
import unittest

from karabo.simulation.pinocchio import Pinocchio
# from karabo.Imaging.imager import Imager
# from karabo.simulation.observation import Observation
# from karabo.simulation.telescope import get_MEERKAT_Telescope, get_OSKAR_Example_Telescope
# from karabo.simulation.interferometer import InterferometerSimulation
# from datetime import datetime
# from astropy import units as u
# from astropy.coordinates import SkyCoord

class TestPinocchio(unittest.TestCase):

    RESULT_FOLDER = "result/"

    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists(TestPinocchio.RESULT_FOLDER):
            os.makedirs(TestPinocchio.RESULT_FOLDER)

    def testSimpleInstance(self) -> None:
        p = Pinocchio()
        p.setRunName("unittest")
        p.printConfig()
        p.printRedShiftRequest()
        p.runPlanner(16, 1)
        p.run(mpiThreads=2)

        p.save(f"./{TestPinocchio.RESULT_FOLDER}")

        # sky = p.getSkyModel()
        # sky.plot_sky()
        # sky = sky.filter_by_radius(0, 1, 32, 45)
        # sky.plot_sky()

        # telescope = get_OSKAR_Example_Telescope()

        # simulation = InterferometerSimulation(channel_bandwidth_hz=1e6,
        #                                           time_average_sec=10)
        # observation = Observation(100e6,
        #                               phase_centre_ra_deg=32,
        #                               phase_centre_dec_deg=45,
        #                               number_of_time_steps=24,
        #                               frequency_increment_hz=20e6,
        #                               number_of_channels=64,
        #                               start_date_and_time=datetime.fromisoformat("2022-07-21T16:00:00+01:00"))

        # visibility = simulation.run_simulation(telescope, sky, observation)

        # visibility.save_to_ms("./result/pinocchiotest/vis.ms")

        # imager = Imager(visibility, imaging_npixel=4096,
        #                     imaging_cellsize=0.03)
        #                     # imaging_phasecentre = SkyCoord(ra=32*u.degree, dec=45*u.degree, frame='icrs').to_string())

        # dirty = imager.get_dirty_image()
        # # dirty.save_as_fits("result/dirty.fits")
        # dirty.plot()
