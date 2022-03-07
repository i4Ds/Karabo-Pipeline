import datetime, enum
from datetime import timedelta, datetime
from typing import Union

class ObservationMode(enum.Enum):
    """
    Enum for selecting between observation mode, either Tracking or Drift_Scan
    """

    Tracking = "Tracking"
    Drift_Scan = "Drift scan"

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
    :ivar mode: ObservationMode, either Tracking (default) or Drift_Scan
    """

    def __init__(self, start_frequency_hz: float,
                 start_date_and_time: datetime = datetime.utcnow(),
                 length: timedelta = timedelta(hours=12),
                 number_of_channels: float = None,
                 frequency_increment_hz: float = None,
                 phase_centre_ra_deg: float = None,
                 phase_centre_dec_deg: float = None,
                 number_of_time_steps: float = None,
                 mode: ObservationMode = None):
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
        self.mode: ObservationMode = mode

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
                "length": self.__strfdelta(self.length)
            }
        }
        if self.number_of_channels:
            settings["observation"]["number_of_channels"] = str(self.number_of_channels)
        if self.frequency_increment_hz:
            settings["observation"]["frequency_increment_hz"] = str(self.frequency_increment_hz)
        if self.phase_centre_ra_deg:
            settings["observation"]["phase_centre_ra_deg"] = str(self.phase_centre_ra_deg)
        if self.phase_centre_dec_deg:
            settings["observation"]["phase_centre_dec_deg"] = str(self.phase_centre_dec_deg)
        if self.number_of_time_steps:
            settings["observation"]["number_of_time_steps"] = str(self.number_of_time_steps)
        if self.mode:
            settings["observation"]["mode"] = self.mode.value

        return settings

    def __strfdelta(self, tdelta):
        hours = tdelta.seconds // 3600 + tdelta.days * 24
        rm = tdelta.seconds % 3600
        minutes = rm // 60
        seconds = rm % 60
        milliseconds = tdelta.microseconds // 1000
        return "{}:{}:{}:{}".format(hours, minutes, seconds, milliseconds)
