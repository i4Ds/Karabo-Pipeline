from datetime import datetime

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.simulator_backend import SimulatorBackend


def run_sample_simulation(
    phase_center: list[float] = [250, -80], verbose: bool = False
) -> tuple[Visibility, SkyModel]:
    """
    Creates example visibilities for use in tests, experiments and examples.

    Args:
        phase_center: ra and dec of the sky
        verbose: Boolean to decide if console outputs are made during simulation
        (e.g. for use in ipynb)

    Returns:
        Visibility: visibilities created by the simulation
        SkyModel: Sky model used for the simulation
    """

    SIMULATOR_BACKEND = SimulatorBackend.OSKAR

    if verbose:
        print("Getting Sky Survey")
    # Get GLEAM Survey Sky
    gleam_sky = SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=80e6)

    if verbose:
        print("Filtering Sky Model")
    sky = gleam_sky.filter_by_radius(0, 0.55, phase_center[0], phase_center[1])
    sky.setup_default_wcs(phase_center=phase_center)

    if verbose:
        print("Setting Up Telescope")
    askap_tel = Telescope.constructor("ASKAP", version=None, backend=SIMULATOR_BACKEND)

    if verbose:
        print("Setting Up Observation")
    observation_settings = Observation(
        start_frequency_hz=100e6,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=16,
        number_of_time_steps=24,
    )

    if verbose:
        print("Generating Visibilities")

    vis = Visibility()
    vis_path = str(vis.vis_path)

    print(f"Saving file to {vis_path}")
    interferometer_sim = InterferometerSimulation(
        vis_path=vis_path, channel_bandwidth_hz=1e6
    )
    visibility_askap = interferometer_sim.run_simulation(
        askap_tel, sky, observation_settings, backend=SIMULATOR_BACKEND
    )

    # In case run_simulation returns a list of vis (allowed by type hint)
    if isinstance(visibility_askap, list):
        visibility_askap = visibility_askap[0]

    return visibility_askap, sky
