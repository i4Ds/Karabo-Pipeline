import enum
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast
from typing import get_args as typing_get_args
from typing import overload
from warnings import warn

import numpy as np
import oskar
import pandas as pd
import xarray as xr
from astropy.coordinates import SkyCoord
from dask import compute, delayed  # type: ignore[attr-defined]
from dask.delayed import Delayed
from dask.distributed import Client
from numpy.typing import NDArray
from ska_sdp_datamodels.image.image_model import Image as RASCILImage
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.visibility import Visibility as RASCILVisibility
from ska_sdp_datamodels.visibility import create_visibility, export_visibility_to_hdf5
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility
from ska_sdp_func_python.sky_component import apply_beam_to_skycomponent
from typing_extensions import assert_never

from karabo.error import KaraboInterferometerSimulationError
from karabo.simulation.beam import BeamPattern
from karabo.simulation.observation import (
    Observation,
    ObservationAbstract,
    ObservationLong,
    ObservationParallelized,
)
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.simulator_backend import SimulatorBackend
from karabo.util._types import (
    DirPathType,
    IntFloat,
    OskarSettingsTreeType,
    PrecisionType,
)
from karabo.util.dask import DaskHandler
from karabo.util.file_handler import FileHandler
from karabo.util.gpu_util import is_cuda_available


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


NoiseRmsType = Literal["Range", "Data file", "Telescope model"]
NoiseFreqType = Literal["Range", "Data file", "Observation settings", "Telescope model"]
StationTypeType = Literal[
    "Aperture array", "Isotropic beam", "Gaussian beam", "VLA (PBCOR)"
]


# TODO: Add noise for the interferometer simulation
# Investigate the Noise file specification by oskar
# class InterferometerNoise()


class InterferometerSimulation:
    """
    Class containing all configuration for the Interferometer Simulation.
    :ivar ms_file_path: Optional. File path of the MS (mass spectrometry) file.
    :ivar vis_path: Optional. File path for visibilities output.
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
    :ivar split_idxs_per_group: The indices of the sky model to split for each group
                                of workers. If None, the sky model will not be split.
                                Useful if the sky model is too large to fit into the
                                memory of a single worker. Group index should be
                                strictly monotonic increasing.
    :ivar precision: For the arithmetic use you can choose between "single" or
                     "double" precision
    :ivar station_type: Here you can choose the type of each station in the
                        interferometer. You can either disable all station beam
                        effects by choosing "Isotropic beam". Or select one of the
                        following beam types:
                        "Gaussian beam", "Aperture array" or "VLA (PBCOR)"
    :ivar enable_power_pattern: If true, gauss_beam_fwhm_deg will be taken in as
                                power pattern.
    :ivar gauss_beam_fwhm_deg: If you choose "Gaussian beam" as station type you need
                               specify the full-width half maximum value at the
                               reference frequency of the Gaussian beam here.
                               Units = degrees. If enable_power_pattern is True,
                               gauss_beam_fwhm_deg is in power pattern, otherwise
                               it is in field pattern.
    :ivar gauss_ref_freq_hz: The reference frequency of the specified FWHM, in Hz.
    :ivar ionosphere_fits_path: The path to a fits file containing an ionospheric screen
                                generated with ARatmospy. The file parameters
                                (times/frequencies) should coincide with the planned
                                observation.
    """

    def __init__(
        self,
        ms_file_path: Optional[str] = None,
        vis_path: Optional[Union[Path, str]] = None,
        channel_bandwidth_hz: IntFloat = 0,
        time_average_sec: IntFloat = 0,
        max_time_per_samples: int = 8,
        correlation_type: CorrelationType = CorrelationType.Cross_Correlations,
        uv_filter_min: IntFloat = 0,
        uv_filter_max: IntFloat = float("inf"),
        uv_filter_units: FilterUnits = FilterUnits.WaveLengths,
        force_polarised_ms: bool = False,
        ignore_w_components: bool = False,
        noise_enable: bool = False,
        noise_seed: Union[str, int] = "time",
        noise_start_freq: IntFloat = 1.0e9,
        noise_inc_freq: IntFloat = 1.0e8,
        noise_number_freq: int = 24,
        noise_rms_start: IntFloat = 0,
        noise_rms_end: IntFloat = 0,
        noise_rms: NoiseRmsType = "Range",
        noise_freq: NoiseFreqType = "Range",
        enable_array_beam: bool = False,
        enable_numerical_beam: bool = False,
        beam_polX: Optional[BeamPattern] = None,  # currently only considered
        # for `ObservationLong`
        beam_polY: Optional[BeamPattern] = None,  # currently only considered
        # for `ObservationLong`
        use_gpus: Optional[bool] = None,
        use_dask: Optional[bool] = None,
        split_observation_by_channels: bool = False,
        n_split_channels: Union[int, str] = "each",
        client: Optional[Client] = None,
        precision: PrecisionType = "single",
        station_type: StationTypeType = "Isotropic beam",
        enable_power_pattern: bool = False,
        gauss_beam_fwhm_deg: IntFloat = 0.0,
        gauss_ref_freq_hz: IntFloat = 0.0,
        ionosphere_fits_path: Optional[str] = None,
        ionosphere_screen_type: Optional[str] = None,
        ionosphere_screen_height_km: Optional[float] = 300,
        ionosphere_screen_pixel_size_m: Optional[float] = 0,
        ionosphere_isoplanatic_screen: Optional[bool] = False,
    ) -> None:
        self._ms_file_path = ms_file_path
        self._vis_path = vis_path

        self.channel_bandwidth_hz: IntFloat = channel_bandwidth_hz
        self.time_average_sec: IntFloat = time_average_sec
        self.max_time_per_samples: int = max_time_per_samples
        self.correlation_type: CorrelationType = correlation_type
        self.uv_filter_min: IntFloat = uv_filter_min
        self.uv_filter_max: IntFloat = uv_filter_max
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
        self.beam_polX = beam_polX
        self.beam_polY = beam_polY
        # set use_gpu
        if use_gpus is None:
            use_gpus = is_cuda_available()
            print(
                "Parameter 'use_gpus' is None! Using function "
                + "'karabo.util.gpu_util.is_cuda_available()'. To overwrite, set "
                + "'use_gpus' True or False."
            )
        self.use_gpus = use_gpus

        if (use_dask is True) or (client is not None):
            if (client is not None) and (use_dask is None):
                use_dask = True
            elif client and use_dask is False:
                raise RuntimeError(
                    "Providing `client` and `use_dask`=False is not allowed."
                )
            elif (client is None) and (use_dask is True):
                client = DaskHandler.get_dask_client()
            else:
                pass
        elif (use_dask is None) and (client is None):
            use_dask = DaskHandler.should_dask_be_used()
            if use_dask:
                client = DaskHandler.get_dask_client()
        self.use_dask = use_dask
        self.client = client

        self.split_observation_by_channels = split_observation_by_channels
        self.n_split_channels = n_split_channels

        self.precision = precision
        self.station_type = station_type
        if self.station_type not in typing_get_args(StationTypeType):
            raise TypeError(
                f"Station type {self.station_type} is not a valid station type. "
                f"Valid station types are: {typing_get_args(StationTypeType)}"
                "This limitation comes from OSKAR itself."
            )
        self.enable_power_pattern = enable_power_pattern
        if self.enable_power_pattern:
            # Convert power pattern to field pattern
            self.gauss_beam_fwhm_deg = gauss_beam_fwhm_deg * np.sqrt(2)
        else:
            self.gauss_beam_fwhm_deg = gauss_beam_fwhm_deg
        self.gauss_ref_freq_hz = gauss_ref_freq_hz
        self.ionosphere_fits_path = ionosphere_fits_path
        self.ionosphere_screen_type = ionosphere_screen_type
        self.ionosphere_screen_height_km = ionosphere_screen_height_km
        self.ionosphere_screen_pixel_size_m = ionosphere_screen_pixel_size_m
        self.ionosphere_isoplanatic_screen = ionosphere_isoplanatic_screen

    @property
    def ms_file_path(self) -> str:
        ms_file_path = self._ms_file_path
        if ms_file_path is None:
            tmp_dir = FileHandler().get_tmp_dir(
                prefix="interferometer-",
                purpose="interferometer disk-cache.",
                unique=self,
            )
            ms_file_path = os.path.join(tmp_dir, "measurements.MS")
            self._ms_file_path = ms_file_path
        return ms_file_path

    @ms_file_path.setter
    def ms_file_path(self, value: str) -> None:
        self._ms_file_path = value

    @property
    def vis_path(self) -> str:
        vis_path = self._vis_path
        if vis_path is None:
            tmp_dir = FileHandler().get_tmp_dir(
                prefix="interferometer-",
                purpose="interferometer disk-cache.",
                unique=self,
            )
            vis_path = os.path.join(tmp_dir, "visibility.vis")
            self._vis_path = vis_path
        return vis_path

    @vis_path.setter
    def vis_path(self, value: str) -> None:
        self._vis_path = value

    @overload
    def run_simulation(
        self,
        telescope: Telescope,
        sky: SkyModel,
        observation: Observation,
        backend: Literal[SimulatorBackend.OSKAR] = ...,
        primary_beam: None = ...,
    ) -> Visibility:
        ...

    @overload
    def run_simulation(
        self,
        telescope: Telescope,
        sky: SkyModel,
        observation: ObservationLong,
        backend: Literal[SimulatorBackend.OSKAR] = ...,
        primary_beam: None = ...,
    ) -> Visibility:
        ...

    @overload
    def run_simulation(
        self,
        telescope: Telescope,
        sky: SkyModel,
        observation: ObservationParallelized,
        backend: Literal[SimulatorBackend.OSKAR] = ...,
        primary_beam: None = ...,
    ) -> List[Visibility]:
        ...

    @overload
    def run_simulation(
        self,
        telescope: Telescope,
        sky: SkyModel,
        observation: Observation,
        backend: Literal[SimulatorBackend.RASCIL],
        primary_beam: Optional[RASCILImage],
    ) -> RASCILVisibility:
        ...

    @overload
    def run_simulation(
        self,
        telescope: Telescope,
        sky: SkyModel,
        observation: ObservationAbstract,
        backend: SimulatorBackend,
        primary_beam: None = ...,
    ) -> Union[Visibility, List[Visibility], RASCILVisibility]:
        ...

    def run_simulation(
        self,
        telescope: Telescope,
        sky: SkyModel,
        observation: ObservationAbstract,
        backend: SimulatorBackend = SimulatorBackend.OSKAR,
        primary_beam: Optional[RASCILImage] = None,
    ) -> Union[Visibility, List[Visibility], RASCILVisibility]:
        """
        Run a single interferometer simulation with the given sky, telescope and
        observation settings.
        :param telescope: telescope model defining the configuration
        :param sky: sky model defining the sky sources
        :param observation: observation settings
        :param backend: Backend used to perform calculations (e.g. OSKAR, RASCIL)
        :param primary_beam: Primary beam to be included into visibilities.
            Currently only relevant for RASCIL.
            For OSKAR, use the InterferometerSimulation constructor parameters instead.
        """
        if backend is SimulatorBackend.OSKAR:
            if primary_beam is not None:
                warn(
                    """
                    Providing a custom primary beam is not supported by OSKAR.
                    The provided primary beam will be ignored.
                    To configure a primary beam effect with OSKAR,
                    set the InterferometerSimulation primary beam parameters
                    (FWHM and reference frequency) instead.
                    """
                )

            if isinstance(observation, ObservationLong):
                return self.__run_simulation_long(
                    telescope=telescope, sky=sky, observation=observation
                )
            elif isinstance(observation, ObservationParallelized):
                return self.__run_simulation_parallelized_observation(
                    telescope=telescope, sky=sky, observation=observation
                )
            else:
                return self.__setup_run_simulation_oskar(
                    telescope=telescope, sky=sky, observation=observation
                )
        elif backend is SimulatorBackend.RASCIL:
            return self.__run_simulation_rascil(
                telescope=telescope,
                sky=sky,
                observation=observation,
                primary_beam=primary_beam,
            )

        assert_never(backend)

    def set_ionosphere(self, file_path: str) -> None:
        """
        Set the path to an ionosphere screen file generated with ARatmospy. The file
        parameters (times/frequencies) should coincide with the planned observation.
        see https://github.com/timcornwell/ARatmospy

        :param file_path: file path to fits file.
        """
        self.ionosphere_fits_path = file_path

    def __run_simulation_rascil(
        self,
        telescope: Telescope,
        sky: SkyModel,
        observation: ObservationAbstract,
        primary_beam: Optional[RASCILImage],
    ) -> RASCILVisibility:
        """
        Compute visibilities from SkyModel using the RASCIL backend.

        :param telescope: Telescope configuration.
            Should be created using the RASCIL backend
        :param sky: SkyModel to be used in the simulation.
            Will be converted into a RASCIL-compatible list of SkyComponent objects.
        :param observation: Observation details
        :param primary_beam: Primary beam to be included into visibilities.
        """
        # Steps followed in this simulation:
        # Compute hour angles based on Observation details
        # Create an empty visibility according to the observation details
        # Convert SkyModel into RASCIL-compatible list of SkyComponent objects
        # Apply DFT to compute visibilities from SkyComponent list
        # Return visibilities

        # Hour angles and integration time from observation
        observation_hour_angles = observation.compute_hour_angles_of_observation()
        observation_integration_time_seconds = (
            observation.length.total_seconds() / observation.number_of_time_steps
        )
        # Note regarding integration time:
        # If the hour angles array has more than one element,
        # then the integration time parameter is not used,
        # since it can be determined as the delta between successive observation times

        # Compute frequency channels
        frequency_channel_starts = np.linspace(
            observation.start_frequency_hz,
            observation.start_frequency_hz
            + observation.frequency_increment_hz * observation.number_of_channels,
            num=observation.number_of_channels,
            endpoint=False,
        )

        frequency_bandwidths = np.full(
            frequency_channel_starts.shape, observation.frequency_increment_hz
        )
        frequency_channel_centers = frequency_channel_starts + frequency_bandwidths / 2

        # Initialize empty visibilities based on observation details
        vis = create_visibility(
            telescope.RASCIL_configuration,  # Configuration of the interferometer array
            times=observation_hour_angles,  # Hour angles
            frequency=frequency_channel_centers,  # Center channel frequencies in Hz
            channel_bandwidth=frequency_bandwidths,
            phasecentre=SkyCoord(
                observation.phase_centre_ra_deg,
                observation.phase_centre_dec_deg,
                unit="deg",
                frame="icrs",
            ),
            weight=1.0,  # Keep as 1, per recommendation from RASCIL docs
            polarisation_frame=PolarisationFrame(
                "stokesI"
            ),  # TODO handle full stokes as well
            integration_time=observation_integration_time_seconds,
            zerow=self.ignore_w_components,
        )

        # Obtain list of SkyComponent instances
        skycomponents = sky.convert_to_backend(
            backend=SimulatorBackend.RASCIL,
            desired_frequencies_hz=frequency_channel_starts,
            channel_bandwidth_hz=observation.frequency_increment_hz,
        )

        if primary_beam is not None:
            skycomponents = apply_beam_to_skycomponent(skycomponents, primary_beam)

        # Compute visibilities from SkyComponent list using DFT
        vis = dft_skycomponent_visibility(
            vis, skycomponents, dft_compute_kernel="cpu_looped"
        )
        # Save visibilities to disk
        export_visibility_to_hdf5(vis, self.vis_path)

        return vis

    def __run_simulation_parallelized_observation(
        self,
        telescope: Telescope,
        sky: SkyModel,
        observation: ObservationParallelized,
    ) -> List[Visibility]:
        # The following line depends on the mode with which we're loading
        # the sky (explained in documentation)
        array_sky = sky.sources

        # Check if there is a dask client
        if self.client is None:
            self.client = DaskHandler.get_dask_client()

        if array_sky is None:
            raise KaraboInterferometerSimulationError(
                "Sky model has not been loaded. Please load the sky model first."
            )

        input_telpath = telescope.path
        if input_telpath is None:
            raise KaraboInterferometerSimulationError(
                "`telescope.path` must be set but is None."
            )
        settings_tree = observation.get_OSKAR_settings_tree()
        observations = Observation.create_observations_oskar_from_lists(
            settings_tree=settings_tree,
            central_frequencies_hz=observation.center_frequencies_hz,
            channel_bandwidths_hz=observation.channel_bandwidths_hz,
            n_channels=observation.n_channels,
        )

        # Some dask stuff
        run_simu_delayed = delayed(self.__run_simulation_oskar)
        delayed_results = []

        # Scatter sky
        array_sky = self.client.scatter(array_sky)
        tmp_dir = FileHandler().get_tmp_dir(
            prefix="simulation-parallelized-observation-",
            purpose="disk-cache simulation-parallelized-observation",
        )
        ms_dir = os.path.join(tmp_dir, "measurements")
        os.makedirs(ms_dir, exist_ok=False)
        vis_dir = os.path.join(tmp_dir, "visibilities")
        os.makedirs(vis_dir, exist_ok=False)
        for observation_params in observations:
            start_freq = observation_params["observation"]["start_frequency_hz"]
            ms_file_path = os.path.join(ms_dir, f"start_freq_{start_freq}.MS")
            vis_path = os.path.join(ms_dir, f"start_freq_{start_freq}.vis")
            interferometer_params = self.__get_OSKAR_settings_tree(
                input_telpath=input_telpath,
                ms_file_path=ms_file_path,
                vis_path=vis_path,
            )

            params_total = {**interferometer_params, **observation_params}

            # Submit the jobs
            delayed_ = run_simu_delayed(
                os_sky=array_sky,
                params_total=params_total,
                precision=self.precision,
            )
            delayed_results.append(delayed_)

        results = cast(
            List[OskarSettingsTreeType],
            compute(*delayed_results, scheduler="distributed"),
        )
        # Extract visibilities
        visibilities_path = [x["interferometer"]["oskar_vis_filename"] for x in results]
        ms_file_paths = [x["interferometer"]["ms_filename"] for x in results]

        # Save the paths into the class Visibility
        visibilities = []
        for i in range(len(visibilities_path)):
            visibilities.append(
                Visibility(vis_path=visibilities_path[i], ms_file_path=ms_file_paths[i])
            )
        return visibilities

    def __setup_run_simulation_oskar(
        self, telescope: Telescope, sky: SkyModel, observation: ObservationAbstract
    ) -> Visibility:
        """
        Run a single interferometer simulation with a given sky,
        telescope and observation settings.
        :param telescope: telescope model defining its configuration
        :param sky: sky model defining the sources
        :param observation: observation settings
        """
        # The following line depends on the mode with which we're loading
        # the sky (explained in documentation)
        array_sky = sky.sources

        if array_sky is None:
            raise KaraboInterferometerSimulationError(
                "Sky model has not been loaded. Please load the sky model first."
            )

        input_telpath = telescope.path
        if input_telpath is None:
            raise KaraboInterferometerSimulationError(
                "`telescope.path` must be set but is None."
            )
        # Create params for the interferometer
        interferometer_params = self.__get_OSKAR_settings_tree(
            input_telpath=input_telpath,
            ms_file_path=self.ms_file_path,
            vis_path=self.vis_path,
        )

        # Initialise the telescope and observation settings
        observation_params = observation.get_OSKAR_settings_tree()

        params_total = {**interferometer_params, **observation_params}
        params_total = InterferometerSimulation.__run_simulation_oskar(
            array_sky, params_total, self.precision
        )
        ms_file_path = params_total["interferometer"]["ms_filename"]
        vis_path = params_total["interferometer"]["oskar_vis_filename"]
        print(f"Saved visibility to {vis_path}")
        return Visibility(vis_path, ms_file_path)

    @staticmethod
    def __run_simulation_oskar(
        os_sky: Union[oskar.Sky, NDArray[np.float_], xr.DataArray, Delayed],
        params_total: OskarSettingsTreeType,
        precision: PrecisionType = "double",
    ) -> Dict[str, Any]:
        """
        Run a single interferometer simulation with a given sky,
        telescope and observation settings.
        :param params_total: Combined parameters for the interferometer
        :param os_sky: OSKAR sky model as np.array or oskar.Sky
        :param precision: precision of the simulation
        """

        # Create a visibility object
        setting_tree = oskar.SettingsTree("oskar_sim_interferometer")
        setting_tree.from_dict(params_total)

        if isinstance(os_sky, Delayed):
            os_sky = os_sky.persist()
        elif isinstance(os_sky, xr.DataArray):
            os_sky = np.array(os_sky.as_numpy())
        if isinstance(os_sky, np.ndarray):
            os_sky = SkyModel.get_OSKAR_sky(os_sky, precision=precision)

        simulation = oskar.Interferometer(settings=setting_tree)
        simulation.set_sky_model(os_sky)
        simulation.run()

        # Return the params, which contain all the information about the simulation
        return params_total

    def __run_simulation_long(
        self,
        telescope: Telescope,
        sky: SkyModel,
        observation: ObservationLong,
    ) -> Visibility:
        # Initialise the telescope and observation settings
        observation_params = observation.get_OSKAR_settings_tree()
        input_telpath = telescope.path
        runs = []

        if self.beam_polX is None or self.beam_polY is None:
            raise KaraboInterferometerSimulationError(
                "`InterferometerSimulation.beam_polX` and "
                + "`InterferometerSimulation.beam_polY` must be set "
                + "to run a long observation."
            )
        if input_telpath is None:
            raise KaraboInterferometerSimulationError(
                "`telescope.path` must be set but is None."
            )

        tmp_dir = FileHandler().get_tmp_dir(
            prefix="simulation-long-",
            purpose="disk-cache simulation-long",
        )
        ms_dir = os.path.join(tmp_dir, "measurements")
        os.makedirs(ms_dir, exist_ok=False)
        vis_dir = os.path.join(tmp_dir, "visibilities")
        os.makedirs(vis_dir, exist_ok=False)

        # Loop over days
        for i, current_date in enumerate(
            pd.date_range(
                observation.start_date_and_time, periods=observation.number_of_days
            ),
            1,
        ):
            # Convert to date
            current_date = current_date.date()
            print(f"Observing Day: {i}. Date: {current_date}")

            if self.enable_array_beam:
                # ------------ X-coordinate
                pb = deepcopy(self.beam_polX)
                beam = pb.sim_beam()
                pb.save_cst_file(beam[3], telescope=telescope)
                pb.fit_elements(telescope)

                # ------------ Y-coordinate
                pb = deepcopy(self.beam_polY)
                pb.save_cst_file(beam[4], telescope=telescope)
                pb.fit_elements(telescope)

            ms_file_path = os.path.join(ms_dir, f"{current_date}.MS")
            vis_path = os.path.join(ms_dir, f"{current_date}.vis")
            interferometer_params = self.__get_OSKAR_settings_tree(
                input_telpath=input_telpath,
                ms_file_path=ms_file_path,
                vis_path=vis_path,
            )

            params_total = {**interferometer_params, **observation_params}

            # Submit the simulation to the workers
            params_total = InterferometerSimulation.__run_simulation_oskar(
                sky.sources,
                params_total,
                self.precision,
            )
            runs.append(params_total)

        # Combine the visibilities
        visibility_paths = [x["interferometer"]["oskar_vis_filename"] for x in runs]

        Visibility.combine_vis(visibility_paths, self.ms_file_path)

        print("Done with simulation.")
        # Returns currently just one of the visibilities, of the first day.
        return Visibility(visibility_paths[0], self.ms_file_path)

    def simulate_foreground_vis(
        self,
        telescope: Telescope,
        foreground: SkyModel,
        foreground_observation: Observation,
        foreground_vis_file: str,
        write_ms: bool,
        foreground_ms_file: str,
    ) -> Tuple[
        Visibility,
        List[NDArray[np.complex_]],
        oskar.VisHeader,
        oskar.Binary,
        oskar.VisBlock,
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
    ]:
        """
        Simulates foreground sources
        """
        print("### Simulating foreground source....")
        visibility = self.run_simulation(telescope, foreground, foreground_observation)
        (fg_header, fg_handle) = oskar.VisHeader.read(foreground_vis_file)
        foreground_cross_correlation: List[NDArray[np.complex_]] = list()
        # fg_max_channel=fg_header.max_channels_per_block;
        for i in range(fg_header.num_blocks):
            fg_block = oskar.VisBlock.create_from_header(fg_header)
            fg_block.read(fg_header, fg_handle, i)
            foreground_cross_correlation[i] = cast(
                NDArray[np.complex_], fg_block.cross_correlations()
            )
        ff_uu = fg_block.baseline_uu_metres()
        ff_vv = fg_block.baseline_vv_metres()
        ff_ww = fg_block.baseline_ww_metres()
        if write_ms:
            visibility.write_to_file(foreground_ms_file)
        return (
            visibility,
            foreground_cross_correlation,
            fg_header,
            fg_handle,
            fg_block,
            ff_uu,
            ff_vv,
            ff_ww,
        )

    def yes_double_precision(self) -> bool:
        return self.precision != "single"

    def __get_OSKAR_settings_tree(
        self,
        input_telpath: DirPathType,
        ms_file_path: str,
        vis_path: str,
    ) -> OskarSettingsTreeType:
        settings: OskarSettingsTreeType = {
            "simulator": {
                "use_gpus": self.use_gpus,
                "double_precision": self.yes_double_precision(),
            },
            "interferometer": {
                "ms_filename": ms_file_path,
                "oskar_vis_filename": vis_path,
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
                "input_directory": str(input_telpath),
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
                    "external_tec_screen/input_fits_file": str(
                        self.ionosphere_fits_path
                    ),
                    "ionosphere_screen_type": self.ionosphere_screen_type,
                    "isoplanatic_screen": self.ionosphere_isoplanatic_screen,
                    "external_tec_screen/screen_height_km": self.ionosphere_screen_height_km,  # noqa: E501
                    "external_tec_screen/screen_pixel_size_m": self.ionosphere_screen_pixel_size_m,  # noqa: E501
                }
            )

        return settings

    @staticmethod
    def __interpret_uv_filter(uv_filter: float) -> str:
        if uv_filter == float("inf"):
            return "max"
        elif uv_filter <= 0:
            return "min"
        else:
            return str(uv_filter)
