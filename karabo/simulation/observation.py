import datetime, os, sys
import numpy as np
from datetime import timedelta, datetime
from copy import deepcopy
from typing import List
from karabo.error import KaraboError

from karabo.karabo_resource import KaraboResource
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.beam import BeamPattern


class Observation(KaraboResource):
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

    def __init__(
        self, mode:str='Tracking',
        start_frequency_hz:float=0,
        start_date_and_time:datetime=datetime.utcnow(),
        length:timedelta=timedelta(hours=12),
        number_of_channels:float=1,
        frequency_increment_hz:float=0,
        phase_centre_ra_deg:float=0,
        phase_centre_dec_deg:float=0,
        number_of_time_steps:float=1,
    ) -> None:

        # required
        self.start_frequency_hz: float = start_frequency_hz
        self.start_date_and_time: datetime = start_date_and_time
        self.length: timedelta = length
        self.mode: str = mode

        # optional
        self.number_of_channels: float = number_of_channels
        self.frequency_increment_hz: float = frequency_increment_hz
        self.phase_centre_ra_deg: float = phase_centre_ra_deg
        self.phase_centre_dec_deg: float = phase_centre_dec_deg
        self.number_of_time_steps: float = number_of_time_steps

    def set_length_of_observation(
        self,
        hours:float,
        minutes:float,
        seconds:float,
        milliseconds:float,
    ) -> None:
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
                "mode":self.mode,
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

    def __strfdelta(
        self,
        tdelta:timedelta,
    ):
        hours = tdelta.seconds // 3600 + tdelta.days * 24
        rm = tdelta.seconds % 3600
        minutes = rm // 60
        seconds = rm % 60
        milliseconds = tdelta.microseconds // 1000
        return "{}:{}:{}:{}".format(hours, minutes, seconds, milliseconds)

    def get_phase_centre(self):
        return [self.phase_centre_ra_deg, self.phase_centre_dec_deg]


class ObservationLong(KaraboResource):
    def __init__(
        self,
        observation:Observation,
        interferometer_simulation:InterferometerSimulation,
        sky_model:SkyModel,
        telescope:Telescope,
        number_of_days:int,
        enable_array_beam:bool=None,
        xcstfile_path:str=None,
        ycstfile_path:str=None,
        beam_method:str='Gaussian Beam',
        avg_frac_error:float=.8,
    ) -> None:

        self.observation: Observation = observation
        self.interferometer_simulation : InterferometerSimulation = interferometer_simulation
        self.sky_model : SkyModel = sky_model
        self.telescope : Telescope = telescope
        self.number_of_days : int = number_of_days
        self.enable_array_beam : bool = enable_array_beam
        self.xcstfile_path : str = xcstfile_path
        self.ycstfile_path : str = ycstfile_path
        self.beam_method : str = beam_method
        self.avg_frac_error: float = avg_frac_error

        self.sky_data = deepcopy(self.sky_model.sources)
        self.vis_path = self.interferometer_simulation.vis_path
        self.interferometer_simulation.enable_array_beam = self.enable_array_beam
        self.interferometer_simulation.enable_numerical_beam = self.enable_array_beam

        self.__check_input()

    def __check_input(self) -> None:
        if self.enable_array_beam and (self.xcstfile_path is None or self.ycstfile_path is None):
            raise KaraboError(f'`enable_array_beam` is {self.enable_array_beam} \
                but `xcstfile_path` and/or `ycstfile_path` are None!')
        if not isinstance(self.number_of_days, int):
            raise KaraboError(f'`number_of_days` must be of type int but is of type {type(self.number_of_days)}!')
        if self.number_of_days <= 1:
            raise KaraboError(f'`number_of_days` must be >=2 but is {self.number_of_days}!')

    def create_vis_long(self) -> List[str]:
        #days = np.arange(1, self.number_of_days + 1)
        visiblity_files = [0] * self.number_of_days
        ms_files = [0] * self.number_of_days
        current_date = self.observation.start_date_and_time
        tel_type = self.telescope.path.split('/')[-1].split('.tm')[0] # works as long as `read_OSKAR_tm_file` sets telescope.path
        if os.path.exists(self.vis_path):
            ans = input(f'{self.vis_path} already exists. Do you want to replace it? [y/N]')
            if ans != 'y':
                sys.exit(0)
        for i in range(self.number_of_days):
            sky = SkyModel(sources=deepcopy(self.sky_data)) # is deepcopy or copy needed?
            telescope = Telescope.read_OSKAR_tm_file(self.telescope.path)
            # telescope.centre_longitude = 3
            # Remove beam if already present
            test = os.listdir(telescope.path)
            for item in test:
                if item.endswith(".bin"):
                    os.remove(os.path.join(telescope.path, item))
            if self.enable_array_beam:
                # ------------ X-coordinate
                pb = BeamPattern(self.xcstfile_path)  # Instance of the Beam class
                beam = pb.sim_beam(beam_method=self.beam_method)  # Computing beam
                pb.save_cst_file(beam[3], telescope_type=tel_type)  # Saving the beam cst file
                pb.fit_elements(
                    telescope,
                    freq_hz=self.observation.start_frequency_hz,
                    avg_frac_error=self.avg_frac_error,
                    pol='X',
                )  # Fitting the beam using cst file
                # ------------ Y-coordinate
                pb = BeamPattern(self.ycstfile_path)
                pb.save_cst_file(beam[4], telescope_type=tel_type)
                pb.fit_elements(
                    telescope,
                    freq_hz=self.observation.start_frequency_hz,
                    avg_frac_error=self.avg_frac_error,
                    pol='Y',
                )
            print('Observing Day: ' + str(i) + ' the ' + str(current_date))
            # ------------- Simulation Begins
            visiblity_files[i] = os.path.join(self.vis_path, 'beam_vis_' + str(i) + '.vis')
            ms_files[i] = visiblity_files[i].split('.vis')[0] + '.ms'
            os.system('rm -rf ' + visiblity_files[i])
            os.system('rm -rf ' + ms_files[i])
            simulation = deepcopy(self.interferometer_simulation)
            simulation.vis_path = visiblity_files[i]
            # ------------- Design Observation
            observation = deepcopy(self.observation)
            observation.start_date_and_time = current_date
            visibility = simulation.run_simulation(telescope, sky, observation)
            visibility.write_to_file(ms_files[i])
            current_date + timedelta(days=1)
        return visiblity_files
