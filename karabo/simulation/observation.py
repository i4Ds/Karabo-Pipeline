from datetime import datetime, timedelta
from typing import Dict, List, Union

from karabo.error import KaraboError
from karabo.util._types import IntFloat


class Observation:
    """
    The Observation class acts as an object to hold all
    important information about an Observation.

    Required parameters for running a simulation with these observational settings.

    :ivar start_frequency_hz: The frequency at the midpoint of the first channel in Hz.
    :ivar start_date_and_time: The start time and date for the observation.
    Default is datetime.utcnow().
    :ivar length: Length of observation. Default is 12 hours.

    Optional parameters for running a simulation with these observational settings.

    :ivar number_of_channels: Number of channels / bands to use
    :ivar frequency_increment_hz: Frequency increment between successive channels in Hz.
    :ivar phase_centre_ra_deg: Right Ascension of the observation pointing
    (phase centre) in degrees.
    :ivar phase_centre_dec_deg: Declination of the observation pointing
    (phase centre) in degrees.
    :ivar number_of_time_steps: Number of time steps in the output data during the
    observation length. This corresponds to the number of correlator dumps for
    interferometer simulations, and the number of beam pattern snapshots for
    beam pattern simulations.
    """

    def __init__(
        self,
        mode: str = "Tracking",
        start_frequency_hz: IntFloat = 0,
        start_date_and_time: Union[datetime, str] = datetime.utcnow(),
        length: timedelta = timedelta(hours=4),
        number_of_channels: int = 1,
        frequency_increment_hz: IntFloat = 0,
        phase_centre_ra_deg: IntFloat = 0,
        phase_centre_dec_deg: IntFloat = 0,
        number_of_time_steps: int = 1,
    ) -> None:
        self.start_frequency_hz = start_frequency_hz

        if isinstance(start_date_and_time, str):
            self.start_date_and_time = datetime.fromisoformat(start_date_and_time)
        else:
            self.start_date_and_time = start_date_and_time

        self.length = length
        self.mode = mode

        # optional
        self.number_of_channels = number_of_channels
        self.frequency_increment_hz = frequency_increment_hz
        self.phase_centre_ra_deg = phase_centre_ra_deg
        self.phase_centre_dec_deg = phase_centre_dec_deg
        self.number_of_time_steps = number_of_time_steps

    def set_length_of_observation(
        self,
        hours: IntFloat,
        minutes: IntFloat,
        seconds: IntFloat,
        milliseconds: IntFloat,
    ) -> None:
        """
        Set a new length for the observation.
        Overriding the observation length set in the constructor.

        :param hours: hours
        :param minutes: minutes
        :param seconds: seconds
        :param milliseconds: milliseconds
        """
        self.length = timedelta(
            hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
        )

    def get_OSKAR_settings_tree(self) -> Dict[str, Dict[str, str]]:
        """
        Get the settings of this observation as an oskar setting tree.
        This function returns a python dictionary formatted
        according to the OSKAR documentation.
        `<https://fdulwich.github.io/oskarpy-doc/settings_tree.html>`.

        :return: Dictionary containing the full configuration
        in the OSKAR Settings Tree format.
        """
        settings = {
            "observation": {
                "start_frequency_hz": str(self.start_frequency_hz),
                "mode": self.mode,
                # remove last three digits from milliseconds
                "start_time_utc": self.start_date_and_time.strftime(
                    "%d-%m-%Y %H:%M:%S.%f"
                )[:-3],
                "length": self.__strfdelta(self.length),
                "num_channels": str(self.number_of_channels),
                "frequency_inc_hz": str(self.frequency_increment_hz),
                "phase_centre_ra_deg": str(self.phase_centre_ra_deg),
                "phase_centre_dec_deg": str(self.phase_centre_dec_deg),
                "num_time_steps": str(self.number_of_time_steps),
            },
        }
        return settings

    def __strfdelta(
        self,
        tdelta: timedelta,
    ) -> str:
        hours = tdelta.seconds // 3600 + tdelta.days * 24
        rm = tdelta.seconds % 3600
        minutes = rm // 60
        seconds = rm % 60
        milliseconds = tdelta.microseconds // 1000
        return "{}:{}:{}:{}".format(hours, minutes, seconds, milliseconds)

    def get_phase_centre(self) -> List[float]:
        return [self.phase_centre_ra_deg, self.phase_centre_dec_deg]


class ObservationLong(Observation):
    """
    This class allows the use of several observations on different
    days over a certain period of time within one day.
    If only ONE observation is desired, even if it takes a little longer,
    this is already possible using `Observation`.
    This class extends `Observation` so its parameters (except `length`)
    are not discussed here.
    `length` is little different, which describes the duration of ONE observation,
    whose maximum duration for `ObservationLong` is 24h.

    :ivar number_of_days: Number of successive days to observe
    """

    def __init__(
        self,
        mode: str = "Tracking",
        start_frequency_hz: IntFloat = 0,
        start_date_and_time: Union[datetime, str] = datetime.utcnow(),
        length: timedelta = timedelta(hours=4),
        number_of_channels: int = 1,
        frequency_increment_hz: IntFloat = 0,
        phase_centre_ra_deg: IntFloat = 0,
        phase_centre_dec_deg: IntFloat = 0,
        number_of_time_steps: int = 1,
        number_of_days: int = 2,
    ) -> None:
        self.enable_check = False
        super().__init__(
            mode=mode,
            start_frequency_hz=start_frequency_hz,
            start_date_and_time=start_date_and_time,
            length=length,
            number_of_channels=number_of_channels,
            frequency_increment_hz=frequency_increment_hz,
            phase_centre_ra_deg=phase_centre_ra_deg,
            phase_centre_dec_deg=phase_centre_dec_deg,
            number_of_time_steps=number_of_time_steps,
        )
        self.number_of_days: int = number_of_days
        self.__check_attrs()

    def __check_attrs(self) -> None:
        if not isinstance(self.number_of_days, int):
            raise KaraboError(
                "`number_of_days` must be of type int but "
                + f"is of type {type(self.number_of_days)}!"
            )
        if self.number_of_days <= 1:
            raise KaraboError(
                f"`number_of_days` must be >=2 but is {self.number_of_days}!"
            )
        if self.length > timedelta(hours=12):
            raise KaraboError(f"`length` should be max 4 hours but is {self.length}!")
