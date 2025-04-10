import time
from datetime import datetime
from typing import Optional

import numpy as np

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.util.dask import DaskHandler
from karabo.util.file_handler import FileHandler


def main(n_channels: int, memory_limit: Optional[int] = None) -> None:
    DaskHandler.memory_limit = memory_limit
    print("Setting up sky model...")
    sky = SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=80e6)
    phase_center = [250, -80]

    print("Filtering sky model...")
    sky = sky.filter_by_radius_euclidean_flat_approximation(
        0, 1, phase_center[0], phase_center[1]
    )

    # Rechunk Sky model
    sky.sources = sky.sources.chunk(np.ceil(len(sky.sources) / 2))  # type: ignore
    print("Size of sky sources: ", sky.sources.nbytes / 1e6, "MB")

    print("Setting up default wcs...")
    sky.setup_default_wcs(phase_center=phase_center)

    print("Setting up telescope...")
    askap_tel = Telescope.constructor("ASKAP")

    print("Setting up observation...")
    observation_settings = Observation(
        start_frequency_hz=100e6,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=n_channels,
        number_of_time_steps=24,
    )

    print("Running simulation...")
    interferometer_sim = InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        use_gpus=False,
        use_dask=True,
        split_observation_by_channels=False,
        n_split_channels="each",
    )

    print(f"Dashboard available here: {interferometer_sim.client.dashboard_link}")  # type: ignore [union-attr] # noqa: E501
    n_workers = len(interferometer_sim.client.scheduler_info()["workers"])  # type: ignore [union-attr] # noqa: E501

    print(f"Number of workers: {n_workers}")
    print(f"Client: {interferometer_sim.client}")

    start = time.time()
    vis = interferometer_sim.run_simulation(
        askap_tel,
        sky,
        observation_settings,
    )

    print(f"MS Vis is {vis.ms_file_path}")

    time_taken = round((time.time() - start) / 60, 2)
    print("Time taken: (minutes)", time_taken)

    # Check that the created visibilities are corresponding to the number of channels

    with open(
        f"output_{str(n_workers)}_nodes_{n_channels}_channels.txt",
        "a",
    ) as file:
        file.write(
            f"Number of channels: {n_channels}. " f"Time taken: {time_taken} min.\n"
        )
        file.flush()

    FileHandler.clean()


if __name__ == "__main__":
    main(n_channels=10, memory_limit=None)
