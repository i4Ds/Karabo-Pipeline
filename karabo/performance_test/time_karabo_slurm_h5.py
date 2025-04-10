import time
from datetime import datetime

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel, SkyPrefixMapping
from karabo.simulation.telescope import Telescope


def main() -> None:
    start = time.time()
    path = "/scratch/snx3000/vtimmel/karabo/point_sources_OSKAR1.h5"
    phase_center = [0, -30]

    prefix_mapping = SkyPrefixMapping(
        ra="Right Ascension",
        dec="Declination",
        stokes_i="Flux",
    )

    sky = SkyModel.get_sky_model_from_h5_to_xarray(
        path=path, prefix_mapping=prefix_mapping
    )

    sky.setup_default_wcs(phase_center=phase_center)
    telescope = Telescope.constructor("EXAMPLE")

    observation_settings = Observation(
        start_frequency_hz=100e6,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=64,
        number_of_time_steps=24,
    )

    interferometer_sim = InterferometerSimulation(channel_bandwidth_hz=1e6)
    _ = interferometer_sim.run_simulation(telescope, sky, observation_settings)

    print(f"Time take for simulation: {(time.time() - start) / 60} minutes")


if __name__ == "__main__":
    main()
