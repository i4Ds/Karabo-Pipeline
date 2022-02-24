import datetime
from datetime import timedelta, datetime
from typing import Union


class Observation:
    """
    The Observation class acts as an object to hold all important information about an Observation.

    Required parameters for running a simulation with these observational settings.

    :ivar start_frequency_hz: The frequency at the midpoint of the first channel, in Hz.
    :ivar start_date_and_time: The start time and date for the observation. Default is datetime.utcnow().
    :ivar length: Length of observation. Default is 12 hours.

    Optional parameters for running a simulation with these observational settings.

    :ivar number_of_channels: Number of channels / bands to use
    :ivar frequency_increment_hz: The frequency increment between successive channels, in Hz.
    :ivar phase_centre_ra_deg: Right Ascension of the observation pointing (phase centre), in degrees.
    :ivar phase_centre_dec_deg: Declination of the observation pointing (phase centre), in degrees.
    :ivar number_of_time_steps: Number of time steps in the output data during the observation length.
                                This corresponds to the number of correlator dumps for interferometer simulations,
                                and the number of beam pattern snapshots for beam pattern simulations.
    """

    def __init__(self, start_frequency_hz: float,
                 start_date_and_time: datetime = datetime.utcnow(),
                 length: timedelta = timedelta(12, 0, 0, 0)):
        # required
        self.start_frequency_hz: float = start_frequency_hz
        self.start_date_and_time: datetime = start_date_and_time
        self.length: timedelta = length

        # optional
        self.number_of_channels: float
        self.frequency_increment_hz: float
        self.phase_centre_ra_deg: float
        self.phase_centre_dec_deg: float
        self.number_of_time_steps: float

    def set_length_of_observation(self, hours: float, minutes: float, seconds: float, milliseconds: float):
        """
        Set a new length for the observation. Overriding the observation length set in the constructor.

        :param hours: hours
        :param minutes: minutes
        :param seconds: seconds
        :param milliseconds: milliseconds
        """
        self.length = timedelta(hours, minutes, seconds, milliseconds)

    def get_OSKAR_settings_tree(self):
        """
        Get the settings of this observation as an oskar setting tree.
        This function returns a python dictionary formatted according to the OSKAR documentation.
        `OSKAR Documentation <https://fdulwich.github.io/oskarpy-doc/settings_tree.html>`_.

        :return: Dictionary containing the full configuration in the OSKAR Settings Tree format.
        """
        settings = {
            "observation": {
                "start_frequency_hz": self.start_frequency_hz,
                # remove last three digits from milliseconds
                "start_time_utc": self.start_date_and_time.strftime("%d-%m-%Y %H:%M:%S.%f")[:-3],
                "length": str(self.length)
            }
        }
        return settings
