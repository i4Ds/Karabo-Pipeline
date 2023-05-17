import time

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.util.dask import DaskHandler, get_number_of_nodes


def main(n_channels: int) -> None:
    DaskHandler.min_gb_ram_per_worker = 4

    start = time.time()

    sky = SkyModel.get_GLEAM_Sky()

    phase_center = [250, -80]

    sky = sky.filter_by_radius(0, 0.55, phase_center[0], phase_center[1])
    sky.setup_default_wcs(phase_center=phase_center)

    askap_tel = Telescope.get_ASKAP_Telescope()

    observation_settings = Observation(
        start_frequency_hz=100e6,
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=n_channels,
        number_of_time_steps=24,
    )

    interferometer_sim = InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        use_gpus=True,
        use_dask=True,
        split_observation_by_channels=True,
        n_split_channels="each",
    )

    visibility_askap = interferometer_sim.run_simulation(
        askap_tel,
        sky,
        observation_settings,
    )

    assert len(visibility_askap) == n_channels
    time_taken = round((time.time() - start) / 60, 2)
    print("Time taken: (minutes)", time_taken)

    with open(f"output_{str(get_number_of_nodes())}_nodes_{str(n_channels)}_channels.txt", "a") as file:
        file.write(
            f"Number of channels: {str(n_channels)}. "
            "Time taken: {str(time_taken)} min.\n"
        )
        file.flush()  # Optional: Flush the buffer to ensure immediate writing


if __name__ == "__main__":
    main(n_channels=10000)
