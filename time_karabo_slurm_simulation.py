import time

import numpy as np

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


def create_random_sources(num_sources, ranges=None):
    """
    Create a random set of sources.

    :param num_sources: number of sources to create
    :param ranges: list of ranges for each parameter.

    Description of ranges:

    - [0] right ascension (deg)-
    - [1] declination (deg)
    - [2] stokes I Flux (Jy)
    - [3] stokes Q Flux (Jy): defaults to 0
    - [4] stokes U Flux (Jy): defaults to 0
    - [5] stokes V Flux (Jy): defaults to 0
    - [6] reference_frequency (Hz): defaults to 0
    - [7] spectral index (N/A): defaults to 0
    - [8] rotation measure (rad / m^2): defaults to 0
    - [9] major axis FWHM (arcsec): defaults to 0
    - [10] minor axis FWHM (arcsec): defaults to 0
    - [11] position angle (deg): defaults to 0
    - [12] source id (object): defaults to None
    """
    if not ranges:
        ranges = [
            [-0.1, 1.1],
            [-29.5, -30.5],
            [1, 3],
            [0, 0],
            [0, 0],
            [0, 0],
            [80.0e6, 100.0e6],
            [-0.7, -0.7],
            [0.0, 0.0],
            [0, 600],
            [50, 50],
            [45, 45],
        ]

    sources = np.column_stack(
        [
            np.random.uniform(min_val, max_val, num_sources)
            for min_val, max_val in ranges
        ]
    )

    return sources


def main(n_random_sources):
    start = time.time()
    sky = SkyModel()
    sky_data = create_random_sources(
        n_random_sources,
    )

    sky.add_point_sources(sky_data)
    phase_center = [0, -30]
    sky.explore_sky(phase_center, s=0.1)

    sky.setup_default_wcs(phase_center=phase_center)
    telescope = Telescope.get_OSKAR_Example_Telescope()

    observation_settings = Observation(
        start_frequency_hz=100e6,
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=64,
        number_of_time_steps=24,
    )

    interferometer_sim = InterferometerSimulation(
        channel_bandwidth_hz=1e6, split_sky_for_dask_how="randomly"
    )
    _ = interferometer_sim.run_simulation(telescope, sky, observation_settings)

    print(f"Time take for simulation: {(time.time() - start) / 60} minutes")


if __name__ == "__main__":
    main(n_random_sources=100)
