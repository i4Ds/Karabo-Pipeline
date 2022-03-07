import enum
from typing import Dict, Union, Any

import oskar

from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


class CorrelationType(enum.Enum):
    """
    Enum for selecting between the different Correlation Types for the Simulator.
    """

    Cross_Correlations = "Cross-Correlations"
    Auto_Correlations = "Auto-Correlations"
    Both = "Both"


class FilterUnits(enum.Enum):
    """
    Enum for selecting between the different Filter Units for the Simulator.
    """

    WaveLengths = "Wavelengths"
    Metres = "Metres"


# TODO: Add noise for the interferometer simulation
# Investigate the Noise file specification by oskar
# class InterferometerNoise()

class InterferometerSimulation:
    """
    Class containing all configuration for the Interferometer Simulation.

    :ivar output_path: Path where the resulting measurement set will be stored.
    :ivar channel_bandwidth_hz: The channel width, in Hz, used to simulate bandwidth smearing.
                                (Note that this can be different to the frequency increment if channels do not cover a contiguous frequency range.)
    :ivar time_average_sec: The correlator time-average duration, in seconds, used to simulate time averaging smearing.
    :ivar max_time_per_samples: The maximum number of time samples held in memory before being written to disk.
    :ivar correlation_type: The type of correlations to produce. Any value of Enum CorrelationType
    :ivar uv_filter_min: The minimum value of the baseline UV length allowed by the filter.
                         Values outside this range are not evaluated
    :ivar uv_filter_max: The maximum value of the baseline UV length allowed by the filter.
                         Values outside this range are not evaluated.
    :ivar uv_filter_units: The units of the baseline UV length filter values. Any value of Enum FilterUnits
    """

    def __init__(self, output_path: str = ".",
                 channel_bandwidth_hz: float = None,
                 time_average_sec: float = None,
                 max_time_per_samples: int = None,
                 max_channels_per_block: Union[str, int] = None,
                 correlation_type: CorrelationType = None,
                 uv_filter_min: float = None,
                 uv_filter_max: float = None,
                 uv_filter_units: FilterUnits = None,
                 force_polarised_ms: bool = None,
                 ignore_w_components: bool = None):

        self.output_path = output_path
        self.channel_bandwidth_hz: float = 0
        self.time_average_sec: float = 0
        self.max_time_per_samples: int = 8
        self.max_channels_per_block = 'auto'
        self.correlation_type: CorrelationType = CorrelationType.Cross_Correlations
        self.uv_filter_min: float = float(0)
        self.uv_filter_max: float = float('inf')
        self.uv_filter_units: FilterUnits = FilterUnits.WaveLengths
        self.force_polarised_ms: bool = False
        self.ignore_w_components: bool = False

    def run_simulation(self, telescope: Telescope, sky: SkyModel, observation: Observation):
        """
        Run a singel interferometer simulation with the given sky, telescope.png and observation settings.
        :param telescope: telescope.png model defining the telescope.png configuration
        :param sky: sky model defining the sky sources
        :param observation: observation settings
        """

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
        simulation.set_output_vis_file("./result.vis")
        simulation.run()

    def __get_OSKAR_settings_tree(self) -> Dict[str, Dict[str, Union[Union[int, float, str], Any]]]:
        settings = {
            "interferometer": {
                "channel_bandwidth_hz": str(self.channel_bandwidth_hz),
                "time_average_sec": str(self.time_average_sec),
                "max_time_samples_per_block": str(self.max_time_per_samples),
                "correlation_type": str(self.correlation_type.value),
                "uv_filter_min": str(self.__interpret_uv_filter(self.uv_filter_min)),
                "uv_filter_max": str(self.__interpret_uv_filter(self.uv_filter_max)),
                "uv_filter_units": str(self.uv_filter_units.value)
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
