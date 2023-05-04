import time

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope


def main() -> None:
    start = time.time()
    path = "/scratch/snx3000/vtimmel/karabo/point_sources_OSKAR1.h5"
    phase_center = [0, -30]

    prefix_mapping = {
        "ra": "Right Ascension",
        "dec": "Declination",
        "i": "Flux",
        "q": None,
        "u": None,
        "v": None,
        "ref_freq": None,
        "spectral_index": None,
        "rm": None,
        "major": None,
        "minor": None,
        "pa": None,
        "id": None,
    }

    sky = SkyModel.get_sky_model_from_h5_to_dask(
        path=path, prefix_mapping=prefix_mapping
    )

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
    main()
