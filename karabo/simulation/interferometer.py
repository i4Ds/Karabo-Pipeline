import enum, os
from typing import Dict, Union, Any

import oskar

import karabo.error
from karabo.simulation.visibility import Visibility
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.util.FileHandle import FileHandle


class CorrelationType(enum.Enum):
    """
    Enum for selecting between the different Correlation Types for the Simulator.
    """

    Cross_Correlations = "Cross-correlations"
    Auto_Correlations = "Auto-correlations"
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

    :ivar ms_path: Path where the resulting measurement set will be stored.
    :ivar vis_path: Path of the visibility output file containing results of the simulation.
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
    :ivar force_polarised_ms: If True, always write the Measurment Set in polarised format even if the simulation
                              was run in the single polarisation ‘Scalar’ (or Stokes-I) mode. If False, the size of
                              the polarisation dimension in the the Measurement Set will be determined by the simulation mode.
    :ivar ignore_w_components: If enabled, baseline W-coordinate component values will be set to 0. This will disable
                               W-smearing. Use only if you know what you’re doing!
    """

    def __init__(
        self,
        vis_path: str = None,
        channel_bandwidth_hz: float = 0,
        time_average_sec: float = 0,
        max_time_per_samples: int = 8,
        correlation_type: CorrelationType = CorrelationType.Cross_Correlations,
        uv_filter_min: float = .0,
        uv_filter_max: float = float('inf'),
        uv_filter_units: FilterUnits = FilterUnits.WaveLengths,
        force_polarised_ms: bool = False,
        ignore_w_components: bool = False,
        noise_enable: bool = False,
        noise_seed: Union[str, int] = 'time',
        noise_start_freq = 1.e9,
        noise_inc_freq = 1.e8,
        noise_number_freq = 24,
        noise_rms_start: float = 0,
        noise_rms_end: float = 0,
        noise_rms: str = 'Range',
        noise_freq: str = 'Range',
        enable_array_beam:bool = False,
        enable_numerical_beam:bool = False):

        self.ms_file: Visibility = Visibility()
        self.vis_path: str = vis_path
        self.channel_bandwidth_hz: float = channel_bandwidth_hz
        self.time_average_sec: float = time_average_sec
        self.max_time_per_samples: int = max_time_per_samples
        self.correlation_type: CorrelationType = correlation_type
        self.uv_filter_min: float = uv_filter_min
        self.uv_filter_max: float = uv_filter_max
        self.uv_filter_units: FilterUnits = uv_filter_units
        self.force_polarised_ms: bool = force_polarised_ms
        self.ignore_w_components: bool = ignore_w_components
        self.noise_enable: bool = noise_enable
        self.noise_start_freq = noise_start_freq
        self.noise_inc_freq = noise_inc_freq
        self.noise_number_freq = noise_number_freq
        self.noise_seed = noise_seed
        self.noise_rms_start = noise_rms_start
        self.noise_rms_end = noise_rms_end
        self.noise_rms=noise_rms
        self.noise_freq=noise_freq
        self.enable_array_beam=enable_array_beam
        self.enable_numerical_beam=enable_numerical_beam

    def run_simulation(self, telescope: Telescope, sky: SkyModel, observation: Observation) -> Visibility:
        """
        Run a singel interferometer simulation with the given sky, telescope.png and observation settings.
        :param telescope: telescope.png model defining the telescope.png configuration
        :param sky: sky model defining the sky sources
        :param observation: observation settings
        """

        os_sky = sky.get_OSKAR_sky()
        observation_settings = observation.get_OSKAR_settings_tree()
        input_telpath=telescope.path
        interferometer_settings = self.__get_OSKAR_settings_tree(input_telpath=input_telpath)
        telescope.get_OSKAR_telescope()
        settings1 = {**interferometer_settings, **observation_settings}
        #settings["telescope"] = {"input_directory": telescope.path, "station_type": 'Aperture array', "aperture_array/element_pattern/enable_numerical": True}
        setting_tree = oskar.SettingsTree("oskar_sim_interferometer")
        setting_tree.from_dict(settings1)
        #settings["telescope"] = {"input_directory":telescope.path} # hotfix #59
        simulation = oskar.Interferometer(settings=setting_tree)
        # simulation.set_telescope_model( # outcommented by hotfix #59
        simulation.set_sky_model(os_sky)
        simulation.run()
        return self.ms_file

    def __get_OSKAR_settings_tree(self,input_telpath) -> Dict[str, Dict[str, Union[Union[int, float, str], Any]]]:
        settings = {
            "interferometer": {
                "ms_filename": self.ms_file.file.path,
                "channel_bandwidth_hz": str(self.channel_bandwidth_hz),
                "time_average_sec": str(self.time_average_sec),
                "max_time_samples_per_block": str(self.max_time_per_samples),
                "correlation_type": str(self.correlation_type.value),
                "uv_filter_min": str(self.__interpret_uv_filter(self.uv_filter_min)),
                "uv_filter_max": str(self.__interpret_uv_filter(self.uv_filter_max)),
                "uv_filter_units": str(self.uv_filter_units.value),
                "force_polarised_ms": str(self.force_polarised_ms),
                "ignore_w_components": str(self.ignore_w_components),
                "noise/enable":str(self.noise_enable),
                "noise/seed":str(self.noise_seed),
                "noise/freq/start":str(self.noise_start_freq),
                "noise/freq/inc":str(self.noise_inc_freq),
                "noise/freq/number":str(self.noise_number_freq),
                "noise/rms":str(self.noise_rms),
                "noise/freq": str(self.noise_freq),
                "noise/rms/start":str(self.noise_rms_start),
                "noise/rms/end": str(self.noise_rms_end)

             },
            "telescope":{"input_directory":input_telpath,
                         "normalise_beams_at_phase_centre": True,
                         "allow_station_beam_duplication": True,
                         "pol_mode":'Full',
                         "station_type":'Aperture array',
                         "aperture_array/array_pattern/enable":self.enable_array_beam,
                         "aperture_array/array_pattern/normalise":True,
                         "aperture_array/element_pattern/enable_numerical":self.enable_numerical_beam,
                         "aperture_array/element_pattern/normalise":True,
                         "aperture_array/element_pattern/taper/type":'None',
            }
        }
        if self.vis_path:
            settings["interferometer"]["oskar_vis_filename"] = self.vis_path
        return settings

    @staticmethod
    def __interpret_uv_filter(uv_filter: float) -> str:
        if uv_filter == float('inf'):
            return "max"
        elif uv_filter <= 0:
            return "min"
        else:
            return str(uv_filter)
