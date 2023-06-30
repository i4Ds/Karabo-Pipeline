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

    def test_create_observations_oskar_settings_tree(self):
        central_frequencies_hz = [200, 250, 370, 300, 301]
        CHANNEL_BANDWIDTHS = [1, 2, 3, 4, 5]
        N_CHANNELS = [1, 2, 3, 4, 5]

        obs = Observation()
        settings_tree = obs.get_OSKAR_settings_tree()
        observations = Observation.create_observations_oskar_from_lists(
            settings_tree, central_frequencies_hz, CHANNEL_BANDWIDTHS, N_CHANNELS
        )
        for i, observation in enumerate(observations):
            start_frequency_hz = float(observation["observation"]["start_frequency_hz"])
            n_channels = int(observation["observation"]["num_channels"])
            freq_inc_hz = float(observation["observation"]["frequency_inc_hz"])

            assert start_frequency_hz == central_frequencies_hz[i]
            assert n_channels == N_CHANNELS[i]
            assert freq_inc_hz == CHANNEL_BANDWIDTHS[i]

        central_frequencies_hz = [200, 300, 400]
        CHANNEL_BANDWIDTHS = 5
        N_CHANNELS = 1

        obs = Observation()
        settings_tree = obs.get_OSKAR_settings_tree()
        observations = Observation.create_observations_oskar_from_lists(
            settings_tree, central_frequencies_hz, CHANNEL_BANDWIDTHS, N_CHANNELS
        )

        for i, observation in enumerate(observations):
            start_frequency_hz = float(observation["observation"]["start_frequency_hz"])
            n_channels = int(observation["observation"]["num_channels"])
            freq_inc_hz = float(observation["observation"]["frequency_inc_hz"])

            assert start_frequency_hz == central_frequencies_hz[i]
            assert n_channels == N_CHANNELS
            assert freq_inc_hz == CHANNEL_BANDWIDTHS

        CENTRAL_FREQ = 200
        CHANNEL_BANDWIDTHS = [1, 2, 3, 4, 5]
        N_CHANNELS = [1, 2, 3, 4, 5]

        obs = Observation()
        settings_tree = obs.get_OSKAR_settings_tree()
        observations = Observation.create_observations_oskar_from_lists(
            settings_tree, CENTRAL_FREQ, CHANNEL_BANDWIDTHS, N_CHANNELS
        )

        for i, observation in enumerate(observations):
            start_frequency_hz = float(observation["observation"]["start_frequency_hz"])
            n_channels = int(observation["observation"]["num_channels"])
            freq_inc_hz = float(observation["observation"]["frequency_inc_hz"])

            assert start_frequency_hz == CENTRAL_FREQ
            assert n_channels == N_CHANNELS[i]
            assert freq_inc_hz == CHANNEL_BANDWIDTHS[i]

        CENTRAL_FREQ = [200, 300, 400, 500]
        CHANNEL_BANDWIDTHS = [1, 2]
        N_CHANNELS = [1, 2]

        obs = Observation()
        settings_tree = obs.get_OSKAR_settings_tree()
        observations = Observation.create_observations_oskar_from_lists(
            settings_tree, CENTRAL_FREQ, CHANNEL_BANDWIDTHS, N_CHANNELS
        )

        for i, observation in enumerate(observations):
            start_frequency_hz = float(observation["observation"]["start_frequency_hz"])
            n_channels = int(observation["observation"]["num_channels"])
            freq_inc_hz = float(observation["observation"]["frequency_inc_hz"])

            assert start_frequency_hz == CENTRAL_FREQ[i]
            assert n_channels == [1, 2, 1, 2][i]
            assert freq_inc_hz == [1, 2, 1, 2][i]
