import enum
import glob
import os
import sys
from copy import deepcopy
from datetime import timedelta
from typing import Any, Dict, List, Union

import numpy as np
import oskar
import pandas as pd
from distributed import Client

from karabo.simulation.beam import BeamPattern
from karabo.simulation.observation import Observation, ObservationLong
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.util.data_util import input_wrapper
from karabo.warning import KaraboWarning


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
    :ivar vis_path: Path of the visibility output file containing results of the
                    simulation.
    :ivar channel_bandwidth_hz: The channel width, in Hz, used to simulate bandwidth
                                smearing. (Note that this can be different to the
                                frequency increment if channels do not cover a
                                contiguous frequency range.)
    :ivar time_average_sec: The correlator time-average duration, in seconds, used
                            to simulate time averaging smearing.
    :ivar max_time_per_samples: The maximum number of time samples held in memory
                                before being written to disk.
    :ivar correlation_type: The type of correlations to produce. Any value of Enum
                            CorrelationType
    :ivar uv_filter_min: The minimum value of the baseline UV length allowed by the
                         filter. Values outside this range are not evaluated
    :ivar uv_filter_max: The maximum value of the baseline UV length allowed by the
                         filter. Values outside this range are not evaluated.
    :ivar uv_filter_units: The units of the baseline UV length filter values.
                           Any value of Enum FilterUnits
    :ivar force_polarised_ms: If True, always write the Measurement Set in polarised
                              format even if the simulation was run in the single
                              polarisation 'Scalar' (or Stokes-I) mode. If False,
                              the size of the polarisation dimension in the
                              Measurement Set will be determined by the simulation
                              mode.
    :ivar ignore_w_components: If enabled, baseline W-coordinate component values will
                               be set to 0. This will disable W-smearing. Use only if
                               you know what you're doing!
    :ivar noise_enable: If true, noise is added.
    :ivar noise_seed: Random number generator seed.
    :ivar noise_start_freq: The start frequency in Hz for which noise is included, if
                            noise is set to true.
    :ivar noise_inc_freq: The frequency increment in Hz, if noise is set to true.
    :ivar noise_number_freq: The number of frequency taken into account, if noise is set
                             to true.
    :ivar noise_rms_start: Station RMS (noise) flux density range start value, in Jy.
                           The range is expanded linearly over the number of frequencies
                           for which noise is defined.
    :ivar noise_rms_end: Station RMS (noise) flux density range end value, in Jy. The
                         range is expanded linearly over the number of frequencies for
                         which noise is defined.
    :ivar noise_rms: The specifications for the RMS noise value:
                        Telescope model: values are loaded from files in the telescope
                                         model directory.
                        Data file: values are loaded from the specified file.
                        Range: values are evaluated according to the specified range
                               parameters (Default).
                     The noise values are specified in Jy and represent the RMS noise of
                     an unpolarised source in terms of flux measured in a single
                     polarisation of the detector.
    :ivar noise_freq: The list of frequencies for which noise values are defined:
                        Telescope model: frequencies are loaded from a data file in
                                         the telescope model directory.
                        Observation settings: frequencies are defined by the observation
                                              settings.
                        Data file: frequencies are loaded from the specified data file.
                        Range: frequencies are specified by the range parameters
                               (Default).
    :ivar enable_array_beam: If true, then the contribution to the station beam from
                             the array pattern (given by beam-forming the antennas in
                             the station) is evaluated.
    :ivar enable_numerical_beam: If true, make use of any available numerical element
                                 pattern files. If numerical pattern data are missing,
                                 the functional type will be used instead.
    :ivar beam_polX: currently only considered for `ObservationLong`
    :ivar beam_polX: currently only considered for `ObservationLong`
    :ivar use_gpus: Set to true if you want to use gpus for the simulation
    :ivar client: The dask client to use for the simulation
    :ivar split_idxs_per_group: The a list of list of indices to split the groups by
    :ivar split_sky_for_dask_by: The attribute to split the data by for dask. Can be
                        "frequency"
    :ivar max_vram_usage_gpu: The maximum vram usage per gpu in GB. Used to split the
                                data into chunks for the simulation on the gpu.
    :ivar precision: For the arithmetic use you can choose between "single" or
                     "double" precision
    :ivar station_type: Here you can choose the type of each station in the
                        interferometer. You can either disable all station beam
                        effects by choosing "Isotropic beam". Or select one of the
                        following beam types:
                        "Gaussian beam", "Aperture array" or "VLA (PBCOR)"
    :ivar gauss_beam_fwhm_deg: If you choose "Gaussian beam" as station type you need
                               specify the full-width half maximum value at the
                               reference frequency of the Gaussian beam here.
                               Units = degrees.
    :ivar gauss_ref_freq_hz: The reference frequency of the specified FWHM, in Hz.
    :ivar ionosphere_fits_path: The path to a fits file containing an ionospheric screen
                                generated with ARatmospy. The file parameters
                                (times/frequencies) should coincide with the planned
                                observation.
    """

    def __init__(
        self,
        vis_path: str = None,
        channel_bandwidth_hz: float = 0,
        time_average_sec: float = 0,
        max_time_per_samples: int = 8,
        correlation_type: CorrelationType = CorrelationType.Cross_Correlations,
        uv_filter_min: float = 0.0,
        uv_filter_max: float = float("inf"),
        uv_filter_units: FilterUnits = FilterUnits.WaveLengths,
        force_polarised_ms: bool = False,
        ignore_w_components: bool = False,
        noise_enable: bool = False,
        noise_seed: Union[str, int] = "time",
        noise_start_freq=1.0e9,
        noise_inc_freq=1.0e8,
        noise_number_freq=24,
        noise_rms_start: float = 0,
        noise_rms_end: float = 0,
        noise_rms: str = "Range",
        noise_freq: str = "Range",
        enable_array_beam: bool = False,
        enable_numerical_beam: bool = False,
        beam_polX: BeamPattern = None,  # currently only considered
        # for `ObservationLong`
        beam_polY: BeamPattern = None,  # currently only considered
        # for `ObservationLong`
        use_gpus: bool = False,
        client: Union[Client, None] = None,
        split_idxs_per_group: Union[List[List[int]], None] = None, 
        split_sky_for_dask_by: str = "frequency",
        max_vram_usage_gpu: float = 0.8,
        precision: str = "single",
        station_type: str = "Isotropic beam",
        gauss_beam_fwhm_deg: float = 0.0,
        gauss_ref_freq_hz: float = 0.0,
        ionosphere_fits_path: str = None,
    ) -> None:
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
        self.noise_rms = noise_rms
        self.noise_freq = noise_freq
        self.enable_array_beam = enable_array_beam
        self.enable_numerical_beam = enable_numerical_beam
        self.beam_polX: BeamPattern = beam_polX
        self.beam_polY: BeamPattern = beam_polY
        self.use_gpus = use_gpus
        self.client = client
        self.split_idxs_per_group = split_idxs_per_group
        self.split_sky_for_dask_by = split_sky_for_dask_by
        self.max_vram_usage_gpu = max_vram_usage_gpu
        self.precision = precision
        self.station_type = station_type
        self.gauss_beam_fwhm_deg = gauss_beam_fwhm_deg
        self.gauss_ref_freq_hz = gauss_ref_freq_hz
        self.ionosphere_fits_path = ionosphere_fits_path

    def run_simulation(
        self, telescope: Telescope, sky: SkyModel, observation: Observation
    ) -> Union[Visibility, List[str]]:
        """
        Run a single interferometer simulation with the given sky, telescope.png and
        observation settings.
        :param telescope: telescope.png model defining the telescope.png configuration
        :param sky: sky model defining the sky sources
        :param observation: observation settings
        """
        if isinstance(observation, ObservationLong):
            return self.__run_simulation_long(
                telescope=telescope,
                sky=sky,
                observation=observation
            )
        else:
            return self.__setup_run_simulation_oskar(
                telescope=telescope,
                sky=sky,
                observation=observation
            )

    def set_ionosphere(self, file_path: str) -> None:
        """
        Set the path to an ionosphere screen file generated with ARatmospy. The file
        parameters (times/frequencies) should coincide with the planned observation.
        see https://github.com/timcornwell/ARatmospy

        :param file_path: file path to fits file.
        """
        self.ionosphere_fits_path = file_path

    def __run_simulation_oskar(sky: SkyModel, simulation: oskar.Interferometer):
        simulation.set_sky_model(sky)
        simulation.run()
        

    def __setup_run_simulation_oskar(
        self,
        telescope: Telescope,
        sky: SkyModel,
        observation: Observation,
    ) -> Visibility:
        """
        Run a single interferometer simulation with a given sky,
        telescope and observation settings.
        :param telescope: telescope model defining it's configuration
        :param sky: sky model defining the sources
        :param observation: observation settings
        """
        observation_params = observation.get_OSKAR_settings_tree()
        input_telpath = telescope.path
        interferometer_params = self.__get_OSKAR_settings_tree(
            input_telpath=input_telpath
        )
        # print(interferometer_settings)
        params_total = {**interferometer_params, **observation_params}
        setting_tree = oskar.SettingsTree("oskar_sim_interferometer")
        setting_tree.from_dict(params_total)

        # The following line depends on the mode with which we're loading
        # the sky (explained in documentation)
        os_sky = sky.get_OSKAR_sky(precision=self.precision)

        # To convert to a numpy array
        array_sky = os_sky.to_array()

        if self.client is not None:
            print(array_sky.shape)
            if self.split_idxs_per_group:
                split_array_sky = np.take(array_sky, self.split_idxs_per_group, axis=0)
            elif self.split_sky_for_dask_by == "frequency":
                # Sort the array by frequency
                array_sky = array_sky[array_sky[:, 6].argsort()]

                # Extract the frequencies from the sky model
                frequencies = array_sky[:, 6]

                # Create dataframe for groupby operations
                frequencies = pd.DataFrame(frequencies, columns=['freq'])

                # Create a column with the rank of the frequency
                frequencies['rank'] = frequencies['freq'].rank(method='dense')

                # Extract N by the number of workers
                N = len(self.client.scheduler_info()["workers"])

                # Create list with the ranks to split on
                spacing = np.ceil(frequencies['freq'].iloc[-1] / N).astype(int)
                split_ranks = [0 + spacing * i for i in range(N)]

                # Split idxs
                split_idxs = []
                for i in range(len(split_ranks) - 1):
                    split_idxs.append(
                        frequencies[frequencies['rank'] == split_ranks[i + 1]].index[0]
                    )

                # Split the array
                split_array_sky = np.split(array_sky, split_idxs)
            else:
                raise ValueError("Unknown split_sky_for_dask_by value. Please use 'frequency' or 'group'.")
        


        simulation = oskar.Interferometer(settings=setting_tree)
        visibility = self.__run_simulation_oskar(sky=split_array_sky[0], Simulation=oskar.Interferometer(settings=setting_tree))

        return self.ms_file

    def __run_simulation_long(
        self,
        telescope: Telescope,
        sky: SkyModel,
        observation: ObservationLong,
    ) -> List[str]:
        try:
            visiblity_files = [0] * observation.number_of_days
            ms_files = [0] * observation.number_of_days  # ms_files is out of range!!!!
            current_date = observation.start_date_and_time
            beam_vis_prefix = "beam_vis_"
            files_existing = []
            if os.path.exists(self.vis_path):
                vis_files_existing = glob.glob(
                    os.path.join(self.vis_path, beam_vis_prefix + "*.vis")
                )
                ms_files_existing = glob.glob(
                    os.path.join(self.vis_path, beam_vis_prefix + "*.ms")
                )
                files_existing = [*vis_files_existing, *ms_files_existing]
                if len(files_existing) > 0:
                    print("Some example files to remove/replace:")
                    print(f"{[*vis_files_existing[:3],*ms_files_existing[:3]]}")
                    msg = (
                        f'Found already existing "beam_vis_*.vis" and '
                        f'"beam_vis_*.ms" files inside {self.vis_path}, \
                        Do you want to replace remove/replace them? [y/N]'
                    )
                    msg = (
                        'Found already existing "beam_vis_*.vis" and '
                        + f'beam_vis_*.ms" files inside {self.vis_path}, \
                        + Do you want to replace remove/replace them? [y/N]'
                    )
                    ans = input_wrapper(msg=msg, ret="y")
                    if ans != "y":
                        sys.exit(0)
                    else:
                        [
                            os.system("rm -rf " + file_name)
                            for file_name in files_existing
                        ]
                        print(
                            f"Removed {len(files_existing)} file(s) matching the "
                            + 'glob pattern "beam_vis_*.vis" and "beam_vis_*.ms"!'
                        )
            else:
                os.makedirs(self.vis_path, exist_ok=True)
                print(f"Created dirs {self.vis_path}")
            vis_path_long = self.vis_path
            for i in range(observation.number_of_days):
                sky_run = SkyModel(
                    sources=deepcopy(sky.sources)
                )  # is deepcopy or copy needed?
                telescope_run = Telescope.read_OSKAR_tm_file(telescope.path)
                # telescope.centre_longitude = 3
                # Remove beam if already present
                test = os.listdir(telescope.path)
                for item in test:
                    if item.endswith(".bin"):
                        os.remove(os.path.join(telescope.path, item))
                if self.enable_array_beam:
                    # ------------ X-coordinate
                    pb = deepcopy(self.beam_polX)
                    beam = pb.sim_beam()
                    pb.save_cst_file(
                        beam[3], telescope=telescope_run
                    )  # Saving the beam cst file
                    pb.fit_elements(telescope_run)
                    # ------------ Y-coordinate
                    pb = deepcopy(self.beam_polY)
                    pb.save_cst_file(beam[4], telescope=telescope_run)
                    pb.fit_elements(telescope_run)
                print("Observing Day: " + str(i) + " the " + str(current_date))
                # ------------- Simulation Begins
                visiblity_files[i] = os.path.join(
                    vis_path_long, beam_vis_prefix + str(i) + ".vis"
                )
                print(visiblity_files[i])
                ms_files[i] = visiblity_files[i].split(".vis")[0] + ".ms"
                self.vis_path = visiblity_files[i]
                # ------------- Design Observation
                observation_run = deepcopy(observation)
                observation_run.start_date_and_time = current_date
                visibility = self.__setup_run_simulation_oskar(
                    telescope_run, sky_run, observation_run
                )
                visibility.write_to_file(ms_files[i])
                current_date + timedelta(days=1)
            self.vis_path = vis_path_long
            return visiblity_files

        except BaseException as exp:
            # self.vis_path = vis_path_long
            raise exp

    def yes_double_precision(self):
        if self.precision == "single":
            pres = False
        else:
            pres = True
        return pres

    def __get_OSKAR_settings_tree(
        self, input_telpath
    ) -> Dict[str, Dict[str, Union[Union[int, float, str], Any]]]:
        settings = {
            "simulator": {
                "use_gpus": self.use_gpus,
                "double_precision": self.yes_double_precision(),
            },
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
                "noise/enable": str(self.noise_enable),
                "noise/seed": str(self.noise_seed),
                "noise/freq/start": str(self.noise_start_freq),
                "noise/freq/inc": str(self.noise_inc_freq),
                "noise/freq/number": str(self.noise_number_freq),
                "noise/rms": str(self.noise_rms),
                "noise/freq": str(self.noise_freq),
                "noise/rms/start": str(self.noise_rms_start),
                "noise/rms/end": str(self.noise_rms_end),
            },
            "telescope": {
                "input_directory": input_telpath,
                "normalise_beams_at_phase_centre": True,
                "allow_station_beam_duplication": True,
                "pol_mode": "Full",
                "station_type": self.station_type,
                "aperture_array/array_pattern/enable": self.enable_array_beam,
                "aperture_array/array_pattern/normalise": True,
                "aperture_array/element_pattern/enable_numerical": self.enable_numerical_beam,  # noqa
                "aperture_array/element_pattern/normalise": True,
                "aperture_array/element_pattern/taper/type": "None",
                "gaussian_beam/fwhm_deg": self.gauss_beam_fwhm_deg,
                "gaussian_beam/ref_freq_hz": self.gauss_ref_freq_hz,
            },
        }

        if self.ionosphere_fits_path:
            settings["telescope"].update(
                {
                    "ionosphere_screen_type": "External",
                    "external_tec_screen/input_fits_file": str(
                        self.ionosphere_fits_path
                    ),
                }
            )

        if self.vis_path:
            settings["interferometer"]["oskar_vis_filename"] = self.vis_path
        return settings

    @staticmethod
    def __interpret_uv_filter(uv_filter: float) -> str:
        if uv_filter == float("inf"):
            return "max"
        elif uv_filter <= 0:
            return "min"
        else:
            return str(uv_filter)
