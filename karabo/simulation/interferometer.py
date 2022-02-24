import enum
from typing import Dict, Union, Any

import oskar

from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


class CorrelationType(enum.Enum):
    """

    """
    Cross_Correlations = "Cross-Correlations"
    Auto_Correlations = "Auto-Correlations"
    Both = "Both"


class FilterUnits(enum.Enum):
    WaveLengths = "Wavelengths"
    Metres = "Metres"


# TODO: Add noise for the interferometer simulation
# Investigate the Noise file specification by oskar
# class InterferometerNoise()

class InterferometerSimulation:

    def __init__(self, output_path: str = "."):
        self.output_path = output_path
        self.channel_bandwidth_hz: float = 0
        self.time_average_sec: float = 0
        self.max_time_per_samples: int = 8
        self.correlation_type: CorrelationType = CorrelationType.Cross_Correlations
        self.uv_filter_min: float = float('-inf')
        self.uv_filter_max: float = float('inf')
        self.uv_filter_units: FilterUnits = FilterUnits.WaveLengths

    def run_simulation(self, telescope: Telescope, sky: SkyModel, observation: Observation):
        os_sky = sky.get_OSKAR_sky()
        observation_settings = observation.get_OSKAR_settings_tree()
        interferometer_settings = self.__get_OSKAR_settings_tree()
        settings = {**interferometer_settings, **observation_settings}
        setting_tree = oskar.SettingsTree("oskar_sim_interferometer")
        setting_tree.from_dict(settings)
        simulation = oskar.Interferometer(None, setting_tree)
        simulation.set_telescope_model(telescope.get_OSKAR_telescope())
        simulation.set_sky_model(os_sky)
        simulation.set_output_measurement_set(self.output_path)
        simulation.run()

    def __get_OSKAR_settings_tree(self) -> Dict[str, Dict[str, Union[Union[int, float, str], Any]]]:
        settings = {
            "interferometer": {
                "channel_bandwidth_hz": self.channel_bandwidth_hz,
                "time_average_sec": self.time_average_sec,
                "max_time_samples_per_block": self.max_time_per_samples,
                "correlation_type": self.correlation_type.value,
                "uv_filter_min": self.__interpret_uv_filter(self.uv_filter_min),
                "uv_filter_max": self.__interpret_uv_filter(self.uv_filter_max),
                "uv_filter_units": self.uv_filter_units.value
            }
        }
        return settings

    @staticmethod
    def __interpret_uv_filter(uv_filter: float) -> str:
        if uv_filter == float('inf'):
            return "max"
        elif uv_filter == float('-inf'):
            return "min"
        else:
            return str(uv_filter)
