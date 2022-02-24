import datetime
from datetime import timedelta
from typing import Union


class Observation:
    def __init__(self, start_frequency_hz: float,
                 start_time: datetime = datetime.datetime.utcnow(),
                 length: timedelta = timedelta(1, 0, 0, 0)):
        # required
        self.start_frequency_hz: float = start_frequency_hz
        self.start_time: datetime = start_time
        self.length: timedelta = length

        # optional
        self.number_of_channels: float
        self.frequency_increment_hz: float
        self.phase_centre_ra_deg: float
        self.phase_centre_dec_deg: float
        self.number_of_time_steps: float

    def set_length_of_observation(self, hours: float, minutes: float, seconds: float, milliseconds: float):
        self.length = timedelta(hours, minutes, seconds, milliseconds)

    def get_OSKAR_settings_tree(self) -> dict[str, dict[str, Union[str, float]]]:
        settings = {
            "observation": {
                "start_frequency_hz": self.start_frequency_hz,
                # remove last three digits from milliseconds
                "start_time_utc": self.start_time.strftime("%d-%m-%Y %H:%M:%S.%f")[:-3],
                "length": str(self.length)
            }
        }
        return settings
