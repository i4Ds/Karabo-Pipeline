import os
import unittest
import datetime
from karabo.simulation.pinocchio import Pinocchio
from karabo.simulation.observation import Observation
from karabo.simulation.telescope import Telescope
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.imaging.imager import Imager
from astropy import units as u


class TestPinocchio(unittest.TestCase):

    RESULT_FOLDER = "./result"

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

        p.save(TestPinocchio.RESULT_FOLDER)
        sky = p.getSkyModel()
        sky = sky.filter_by_radius(0, 1, 32, 45)

        # telescope = Telescope.get_SKA1_MID_Telescope()

        # simulation = InterferometerSimulation(channel_bandwidth_hz=1e6,
        #                                            time_average_sec=10)
        # observation = Observation(1e9,
        #                                phase_centre_ra_deg=31.9875,
        #                                phase_centre_dec_deg=45.1333,
        #                                length=datetime.timedelta(hours=4),
        #                                number_of_time_steps=1,
        #                                frequency_increment_hz=20e6,
        #                                number_of_channels=1,
        #                                start_date_and_time=datetime.datetime.fromisoformat("2022-03-01T11:00:00"))

        # visibility = simulation.run_simulation(telescope, sky, observation)

        # visibility.write_to_file(f"{TestPinocchio.RESULT_FOLDER}/pinocchiotest/vis.ms")
        # cellsize=0.003;boxsize=4096*4
        # imager = Imager(visibility, imaging_npixel=boxsize, imaging_cellsize=cellsize)

        # dirty = imager.get_dirty_image()
        # dirty.write_to_file(f"{TestPinocchio.RESULT_FOLDER}/dirty.fits")
        # dirty.plot()
