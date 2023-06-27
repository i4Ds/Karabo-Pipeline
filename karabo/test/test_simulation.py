import numpy as np
from numpy.typing import NDArray

from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation, ObservationParallized
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


def test_oskar_simulation_basic(sky_data: NDArray[np.float64]):
    # Tests oskar simulation. Should use GPU if available and if not, CPU.
    sky = SkyModel()
    sky.add_point_sources(sky_data)
    sky = SkyModel.get_random_poisson_disk_sky((220, -60), (260, -80), 1, 1, 1)
    sky.explore_sky([240, -70], s=10)
    telescope = Telescope.get_OSKAR_Example_Telescope()
    telescope.centre_longitude = 3

    simulation = InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        time_average_sec=10,
    )
    observation = Observation(
        start_frequency_hz=100e6,
        phase_centre_ra_deg=240,
        phase_centre_dec_deg=-70,
        number_of_time_steps=24,
        frequency_increment_hz=20e6,
        number_of_channels=64,
    )

    simulation.run_simulation(telescope, sky, observation)


def test_parallelization_by_observation():
    sky = SkyModel.get_GLEAM_Sky([76])
    phase_center = [250, -80]
    CENTER_FREQUENCIES_HZ = [100e6, 101e6]
    CHANNEL_BANDWIDTHS_HZ = [1, 2]
    N_CHANNELS = [4, 8]

    sky = sky.filter_by_radius(0, 0.55, phase_center[0], phase_center[1])
    telescope = Telescope.get_ASKAP_Telescope()

    simulation = InterferometerSimulation(channel_bandwidth_hz=1e6, time_average_sec=1)

    obs_parallized = ObservationParallized(
        center_frequencies_hz=CENTER_FREQUENCIES_HZ,
        channel_bandwidths_hz=CHANNEL_BANDWIDTHS_HZ,
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_time_steps=24,
        n_channels=N_CHANNELS,
    )

    visibilities = simulation.run_simulation(telescope, sky, obs_parallized)

    for i, vis in enumerate(visibilities):
        imager = Imager(
            vis, imaging_npixel=2048, imaging_cellsize=3.878509448876288e-05
        )  # imaging cellsize is over-written in the Imager based on max uv dist.
        dirty = imager.get_dirty_image()
        dirty.write_to_file(f"result/parallized_by_obs/dirty_{i}.fits", overwrite=True)
        assert dirty.header["CRVAL4"] == CENTER_FREQUENCIES_HZ[i]
        assert dirty.header["NAXIS4"] == N_CHANNELS[i]
        assert dirty.header["CDELT4"] == CHANNEL_BANDWIDTHS_HZ[i]
