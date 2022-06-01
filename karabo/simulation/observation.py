import datetime, enum
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
                 length: timedelta = timedelta(hours=12),
                 number_of_channels: float = 1,
                 frequency_increment_hz: float = 0,
                 phase_centre_ra_deg: float = 0,
                 phase_centre_dec_deg: float = 0,
                 number_of_time_steps: float = 1):
        # required
        self.start_frequency_hz: float = start_frequency_hz
        self.start_date_and_time: datetime = start_date_and_time
        self.length: timedelta = length

        # optional
        self.number_of_channels: float = number_of_channels
        self.frequency_increment_hz: float = frequency_increment_hz
        self.phase_centre_ra_deg: float = phase_centre_ra_deg
        self.phase_centre_dec_deg: float = phase_centre_dec_deg
        self.number_of_time_steps: float = number_of_time_steps

    def set_length_of_observation(self, hours: float, minutes: float, seconds: float, milliseconds: float):
        """
        Set a new length for the observation. Overriding the observation length set in the constructor.

        :param hours: hours
        :param minutes: minutes
        :param seconds: seconds
        :param milliseconds: milliseconds
        """
        self.length = timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

    def get_OSKAR_settings_tree(self):
        """
        Get the settings of this observation as an oskar setting tree.
        This function returns a python dictionary formatted according to the OSKAR documentation.
        `OSKAR Documentation <https://fdulwich.github.io/oskarpy-doc/settings_tree.html>`_.

        :return: Dictionary containing the full configuration in the OSKAR Settings Tree format.
        """
        settings = {
            "observation": {
                "start_frequency_hz": str(self.start_frequency_hz),
                # remove last three digits from milliseconds
                "start_time_utc": self.start_date_and_time.strftime("%d-%m-%Y %H:%M:%S.%f")[:-3],
                "length": self.__strfdelta(self.length),
                "num_channels": str(self.number_of_channels),
                "frequency_inc_hz": str(self.frequency_increment_hz),
                "phase_centre_ra_deg": str(self.phase_centre_ra_deg),
                "phase_centre_dec_deg": str(self.phase_centre_dec_deg),
                "num_time_steps": str(self.number_of_time_steps)
            }
        }
        return settings

    def __strfdelta(self, tdelta):
        hours = tdelta.seconds // 3600 + tdelta.days * 24
        rm = tdelta.seconds % 3600
        minutes = rm // 60
        seconds = rm % 60
        milliseconds = tdelta.microseconds // 1000
        return "{}:{}:{}:{}".format(hours, minutes, seconds, milliseconds)


    def observe(self, telescope, sky):
        # start once
        pass