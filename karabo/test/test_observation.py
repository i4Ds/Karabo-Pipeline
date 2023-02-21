import unittest
from datetime import datetime

from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation, ObservationLong
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.tools.observation_planner import ObservationPlotter


class TestObservation(unittest.TestCase):
    def testConstructorWithString(self):
        sDateTime: str = "1992-05-28T23:00:00"

        o: Observation = Observation(start_date_and_time=sDateTime)
        self.assertTrue(
            o.start_date_and_time == datetime.fromisoformat(sDateTime),
            "Observation constructor with string input broken",
        )

        ol: ObservationLong = ObservationLong(
            start_date_and_time=sDateTime, number_of_days=2
        )
        self.assertTrue(
            ol.start_date_and_time == datetime.fromisoformat(sDateTime),
            "ObservationLong constructor with string input broken",
        )

    def testConstructorWithDateTime(self):
        dt: datetime = datetime.fromisoformat("1988-06-30T16:32:14")

        o: Observation = Observation(start_date_and_time=dt)
        self.assertTrue(
            o.start_date_and_time == dt,
            "Observation constructor with datetime object broken",
        )

        ol: ObservationLong = ObservationLong(start_date_and_time=dt, number_of_days=2)
        self.assertTrue(
            ol.start_date_and_time == dt,
            "ObservationLong constructor with datetime object broken",
        )

    def test_observation_planer(self):
        sky = SkyModel.get_GLEAM_Sky()
        tel = Telescope.get_MEERKAT_Telescope()
        _ = InterferometerSimulation(channel_bandwidth_hz=1e6, time_average_sec=10)
        observation = Observation(
            100e6,
            phase_centre_ra_deg=240,
            phase_centre_dec_deg=-70,
            number_of_time_steps=24,
            frequency_increment_hz=20e6,
            number_of_channels=64,
        )

        imager = Imager(None, imaging_cellsize=0.03, imaging_npixel=512)

        ObservationPlotter(sky, tel, observation, imager).plot()
