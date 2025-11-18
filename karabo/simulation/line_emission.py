import math
import os
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union, overload

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from ska_sdp_datamodels.image import create_image
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

from karabo.imaging.image import Image
from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_factory import ImagingBackend, get_imager
from karabo.imaging.imager_interface import ImageSpec
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.line_emission_helpers import convert_frequency_to_z
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.simulator_backend import SimulatorBackend

CircleSkyRegion = namedtuple("CircleSkyRegion", ["center", "radius"])


@overload
def line_emission_pipeline(
    output_base_directory: Union[Path, str],
    pointings: List[CircleSkyRegion],
    sky_model: SkyModel,
    observation_details: Observation,
    telescope: Telescope,
    interferometer: InterferometerSimulation,
    simulator_backend: SimulatorBackend,
    dirty_imager_config: DirtyImagerConfig,
    primary_beams: List[NDArray[np.float_]],
    imaging_backend: ImagingBackend = ImagingBackend.RASCIL,
) -> Tuple[List[List[Visibility]], List[List[Image]]]:
    ...


@overload
def line_emission_pipeline(
    output_base_directory: Union[Path, str],
    pointings: List[CircleSkyRegion],
    sky_model: SkyModel,
    observation_details: Observation,
    telescope: Telescope,
    interferometer: InterferometerSimulation,
    simulator_backend: SimulatorBackend,
    dirty_imager_config: DirtyImagerConfig,
    primary_beams: Optional[List[NDArray[np.float_]]] = ...,
    imaging_backend: ImagingBackend = ImagingBackend.RASCIL,
) -> Tuple[List[List[Visibility]], List[List[Image]]]:
    ...


@overload
def line_emission_pipeline(
    output_base_directory: Union[Path, str],
    pointings: List[CircleSkyRegion],
    sky_model: SkyModel,
    observation_details: Observation,
    telescope: Telescope,
    interferometer: InterferometerSimulation,
    simulator_backend: SimulatorBackend,
    dirty_imager_config: DirtyImagerConfig,
    primary_beams: Optional[List[NDArray[np.float_]]] = ...,
    should_perform_primary_beam_correction: Optional[bool] = True,
    imaging_backend: ImagingBackend = ImagingBackend.RASCIL,
) -> Tuple[List[List[Visibility]], List[List[Image]]]:
    ...


def line_emission_pipeline(
    output_base_directory: Union[Path, str],
    pointings: List[CircleSkyRegion],
    sky_model: SkyModel,
    observation_details: Observation,
    telescope: Telescope,
    interferometer: InterferometerSimulation,
    simulator_backend: SimulatorBackend,
    dirty_imager_config: DirtyImagerConfig,
    primary_beams: Optional[List[NDArray[np.float_]]] = None,
    should_perform_primary_beam_correction: Optional[bool] = True,
    imaging_backend: ImagingBackend = ImagingBackend.RASCIL,
) -> Tuple[List[List[Visibility]], List[List[Image]]]:
    """Perform a line emission simulation, to compute visibilities and dirty images.
    A line emission simulation involves assuming every source in the input SkyModel
    only emits within one frequency channel.

    If requested, include primary beam effects into the visibilities and dirty images.

    For the actual primary beam effect in OSKAR,
    set the relevant parameter in the InterferometerSimulation constructor.

    For RASCIL, the provided primary beams are used
    for the primary beam effect.
    """
    print(f"Selected backend: {simulator_backend}")

    output_base_directory = Path(output_base_directory)
    # If output filepath does not exist, mkdir
    print(f"Creating {output_base_directory} directory if it does not exist yet.")
    output_base_directory.mkdir(exist_ok=True, parents=True)

    # Compute frequency channels
    frequency_channel_starts = np.linspace(
        observation_details.start_frequency_hz,
        observation_details.start_frequency_hz
        + observation_details.frequency_increment_hz
        * observation_details.number_of_channels,
        num=observation_details.number_of_channels,
        endpoint=False,
    )

    # Verify that, if primary beams are provided,
    # we have one primary beam per frequency channel
    if primary_beams is not None:
        assert (
            len(primary_beams) == observation_details.number_of_channels
        ), f"""Did not provide same number of primary beams
            as number of desired frequency channels:
        {len(primary_beams)}, {observation_details.number_of_channels}"""

    # Create observation instance to be used for each pointing and each channel
    observation = deepcopy(observation_details)

    print("Computing visibilities...")

    # Loop through pointings
    visibilities: List[List[Visibility]] = []

    for index_freq, frequency_start in enumerate(frequency_channel_starts):
        print(f"Processing frequency channel {index_freq}...")
        visibilities.append([])

        for index_p, p in enumerate(pointings):
            print(f"Processing pointing {index_p}...")
            center = p.center
            radius = p.radius.to(u.deg).value
            # Create observation details
            observation.phase_centre_ra_deg = center.ra.deg
            observation.phase_centre_dec_deg = center.dec.deg
            observation.number_of_channels = 1  # For line emission
            observation.start_frequency_hz = frequency_start

            # Convert provided primary beam to required image format
            if primary_beams is None:
                primary_beam = None
            else:
                # Currently supported backend for custom primary beams: RASCIL
                primary_beam = create_image(
                    npixel=dirty_imager_config.imaging_npixel,
                    cellsize=dirty_imager_config.imaging_cellsize,
                    phasecentre=center,
                    polarisation_frame=PolarisationFrame(
                        "stokesI"
                    ),  # TODO support full stokes
                    frequency=observation.start_frequency_hz,
                    channel_bandwidth=observation.frequency_increment_hz,
                    nchan=1,
                )

                # TODO support full stokes,
                # instead of hardcoding index 0 for the polarisation
                primary_beam["pixels"][0][0] = primary_beams[index_freq]

            # Filter sky based on pointing and on frequency channel
            filtered_sky = sky_model.filter_by_radius_euclidean_flat_approximation(
                inner_radius_deg=0,
                outer_radius_deg=radius,
                ra0_deg=center.ra.deg,
                dec0_deg=center.dec.deg,
            )
            z_min = convert_frequency_to_z(
                frequency_start + observation.frequency_increment_hz
            )
            z_max = convert_frequency_to_z(frequency_start)

            filtered_sky = filtered_sky.filter_by_column(
                col_idx=13,
                min_val=z_min,
                max_val=z_max,
            )

            assert (
                filtered_sky.num_sources > 0
            ), f"""For frequency channel {index_freq}
                    and pointing {index_p}, there are 0 sources in the sky model.
                    Setting visibility to None, and skipping analysis."""

            vis = interferometer.run_simulation(
                telescope=telescope,
                sky=filtered_sky,
                observation=observation,
                backend=simulator_backend,
                primary_beam=primary_beam,
                visibility_format="MS",
                visibility_path=os.path.join(
                    output_base_directory,
                    f"visibilities_f{index_freq}_p{index_p}.MS",
                ),
            )  # type: ignore[call-overload]

            visibilities[-1].append(vis)

    assert len(visibilities) == observation_details.number_of_channels
    assert len(visibilities[0]) == len(pointings)

    print("Creating dirty images from visibilities...")

    imager = get_imager(imaging_backend)
    dirty_images: List[List[Image]] = []
    for index_freq, _ in enumerate(frequency_channel_starts):
        print(f"Processing frequency channel {index_freq}...")
        dirty_images.append([])
        for index_p, _ in enumerate(pointings):
            print(f"Processing pointing {index_p}...")
            vis = visibilities[index_freq][index_p]

            image_spec = ImageSpec(
                npix=dirty_imager_config.imaging_npixel,
                cellsize_arcsec=math.degrees(dirty_imager_config.imaging_cellsize)
                * 3600.0,
                phase_centre_deg=(center.ra.deg, center.dec.deg),
                polarisation="I",
                nchan=1,
            )
            dirty, _ = imager.invert(vis, image_spec)

            if simulator_backend is SimulatorBackend.OSKAR:
                backend = "OSKAR"
            else:
                backend = "RASCIL"

            dirty_output_path = os.path.join(
                output_base_directory, f"dirty_{backend}_{index_p}.fits"
            )
            dirty.write_to_file(dirty_output_path, overwrite=True)
            dirty = Image(path=dirty_output_path)

            dirty_images[-1].append(dirty)

            # dirty.data.shape meaning:
            # (frequency channels, polarisations, pixels_x, pixels_y)
            assert dirty.data.ndim == 4
            # I.e. only one frequency channel,
            # since we are performing a line emission analysis
            assert dirty.data.shape[0] == 1

    assert len(dirty_images) == observation_details.number_of_channels
    assert len(dirty_images[0]) == len(
        pointings
    ), f"{len(dirty_images[0])}, {len(pointings)}"

    return visibilities, dirty_images
