from datetime import datetime
from typing import Tuple

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility, VisibilityFormat
from karabo.simulator_backend import SimulatorBackend
from karabo.util._types import IntFloatList


def run_sample_simulation(
    *,
    simulator_backend: SimulatorBackend = SimulatorBackend.OSKAR,
    visibility_format: VisibilityFormat = "MS",
    verbose: bool = False,
) -> Tuple[
    Visibility, IntFloatList, SkyModel, Telescope, Observation, InterferometerSimulation
]:
    """
    TODO update
    Creates example visibilities for use in tests, experiments and examples.

    Args:
        phase_center: ra and dec of the sky. Defaults to [250, -80] if not provided.
        verbose: Boolean to decide if console outputs are made during simulation
        (e.g. for use in ipynb)

    Returns:
        Visibility: visibilities created by the simulation
        SkyModel: Sky model used for the simulation
    """
    if simulator_backend == SimulatorBackend.RASCIL:
        raise NotImplementedError(
            "RASCIL simulations are currently not supported in this sample"
        )

    phase_center = [250, -80]

    if verbose:
        print("Getting Sky Survey")
    # Get GLEAM Survey Sky
    sky = SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=80e6)

    if verbose:
        print("Filtering Sky Model")
    sky = sky.filter_by_radius(0, 0.55, phase_center[0], phase_center[1])
    sky.setup_default_wcs(phase_center=phase_center)

    if verbose:
        print("Setting Up Telescope")
    telescope = Telescope.constructor(
        "ASKAP", version=None, backend=simulator_backend
    )  # type: ignore[call-overload]

    if verbose:
        print("Setting Up Observation")
    observation = Observation(
        start_frequency_hz=100e6,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=16,
        number_of_time_steps=24,
    )

    if verbose:
        print("Generating Visibilities")

    interferometer_sim = InterferometerSimulation(channel_bandwidth_hz=1e6)
    visibility = interferometer_sim.run_simulation(
        telescope,
        sky,
        observation,
        backend=simulator_backend,
        visibility_format=visibility_format,
    )  # type: ignore[call-overload]

    # In case run_simulation returns a list of vis (allowed by type hint)
    if isinstance(visibility, list):
        visibility = visibility[0]

    return visibility, phase_center, sky, telescope, observation, interferometer_sim
