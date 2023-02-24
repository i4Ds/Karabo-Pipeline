import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from karabo.simulation.beam import BeamPattern
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import ObservationLong
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility


def observation_length_integration_time_to_time_steps(
    observation_length, integration_time
):
    return int(observation_length.total_seconds() / integration_time.total_seconds())


def create_random_sources(num_sources, ranges):
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

    sources = np.column_stack(
        (
            np.random.uniform(min_val, max_val, num_sources)
            for min_val, max_val in ranges
        ),
    )

    return sources


def main(
    n_random_sources,
    total_observational_length,
    integration_time,
    daily_observational_length=timedelta(hours=4),
):

    sky = SkyModel()
    sky_data = create_random_sources(
        n_random_sources,
        [
            [-1, 1],
            [-29, -31],
            [1, 3],
            [0, 2],
            [0, 2],
            [0, 2],
            [100.0e6, 100.0e6],
            [-0.7, -0.7],
            [0.0, 0.0],
            [0, 600],
            [50, 50],
            [45, 45],
        ],
    )
    sky.add_point_sources(sky_data)
    sky.explore_sky([0, -30], s=10)

    telescope = Telescope.get_OSKAR_Example_Telescope()
    telescope.plot_telescope()

    number_of_days = int(
        total_observational_length.total_seconds()
        / daily_observational_length.total_seconds()
    )

    observation_long = ObservationLong(
        start_frequency_hz=100e6,
        phase_centre_ra_deg=0,
        phase_centre_dec_deg=-30,
        number_of_channels=2,
        number_of_time_steps=observation_length_integration_time_to_time_steps(
            daily_observational_length, integration_time
        ),
        number_of_days=number_of_days,
        length=daily_observational_length,
    )
    xcstfile_path = "cst_like_beam_port_1.txt"

    ycstfile_path = "cst_like_beam_port_2.txt"
    beam_polX = BeamPattern(
        cst_file_path=xcstfile_path,
        telescope=telescope,
        freq_hz=observation_long.start_frequency_hz,
        pol="X",
        avg_frac_error=0.8,
        beam_method="Gaussian Beam",
    )
    beam_polY = BeamPattern(
        cst_file_path=ycstfile_path,
        telescope=telescope,
        freq_hz=observation_long.start_frequency_hz,
        pol="Y",
        avg_frac_error=0.8,
        beam_method="Gaussian Beam",
    )

    interferometer_sim = InterferometerSimulation(
        channel_bandwidth_hz=1e5,
        vis_path="./data/visibilities.ms",
        beam_polX=beam_polX,
        beam_polY=beam_polY,
    )
    visibilities = interferometer_sim.run_simulation(telescope, sky, observation_long)
    combined_vis_filepath = os.path.join("data", "combined_vis.ms")
    Visibility.combine_vis(number_of_days, visibilities, combined_vis_filepath, True)


if __name__ == "__main__":
    # Set numpy seed
    np.random.seed(0)
    OBSERVATIONAL_TIMES = [
        timedelta(hours=100),
        # timedelta(hours=500),
        # timedelta(hours=750),
        # timedelta(hours=1000),
    ]
    N_POINTS = [1024, 2048]
    # Create logging dataframe
    timings = pd.DataFrame(columns=["n_points", "obs_time", "time"])
    timings.to_csv("timings.csv", index=False)
    for n_points in N_POINTS:
        for obs_time in OBSERVATIONAL_TIMES:
            time_start = time.time()
            main(
                n_random_sources=n_points,
                total_observational_length=obs_time,
                integration_time=timedelta(seconds=10),
                daily_observational_length=timedelta(hours=4),
            )
            time_end = time.time()
            # Read in CSV and append new row
            timings = pd.read_csv("timings.csv")
            timings = timings.append(
                {
                    "n_points": n_points,
                    "obs_time": obs_time,
                    "time": time_end - time_start,
                },
                ignore_index=True,
            )
            # Close all plots
            plt.close("all")
            # Write out CSV
            timings.to_csv("timings.csv", index=False)
