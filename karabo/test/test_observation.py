import unittest
from datetime import datetime

from karabo.simulation.observation import Observation, ObservationLong


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
