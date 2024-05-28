import time
from datetime import datetime
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from karabo.imaging.imager_rascil import RascilImageCleaner, RascilImageCleanerConfig
from karabo.imaging.util import project_sky_to_image
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.sourcedetection.evaluation import SourceDetectionEvaluation
from karabo.sourcedetection.result import PyBDSFSourceDetectionResult
from karabo.util._types import IntFloat


def create_random_sources(
    num_sources: int,
    ranges: Optional[List[List[IntFloat]]] = None,
) -> NDArray[np.float64]:
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
            [-1, 1],
            [-25, -35],
            [1, 3],
            [0, 0],
            [0, 0],
            [0, 0],
            [100.0e6, 100.0e6],
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


def main(n_random_sources: int) -> None:
    start = time.time()

    sky = SkyModel()
    sky_data = create_random_sources(
        n_random_sources,
    )

    sky.add_point_sources(sky_data)
    phase_center = [0, -30]
    sky.explore_sky(phase_center, s=0.1)

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
    visibility_askap = interferometer_sim.run_simulation(
        telescope, sky, observation_settings
    )

    # Imaging
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    # Try differnet algorithm
    # More sources
    (
        deconvolved,
        restored,
        residual,
    ) = RascilImageCleaner(
        RascilImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            ingest_vis_nchan=16,
            clean_nmajor=0,
            clean_algorithm="mmclean",
            clean_scales=[0, 6, 10, 30, 60],
            clean_threshold=0.12e-3,
            clean_nmoment=5,
            clean_psf_support=640,
            clean_restored_output="integrated",
            use_dask=False,
        )
    ).create_cleaned_image_variants(
        ms_file_path=visibility_askap.ms_file_path,
    )

    # Source detection
    detection_result = PyBDSFSourceDetectionResult.detect_sources_in_image(restored)
    if detection_result is None:
        raise ValueError("`detection_result` is None.")

    ground_truth, sky_idxs = project_sky_to_image(
        sky=sky,
        phase_center=phase_center,
        imaging_cellsize=imaging_cellsize,
        imaging_npixel=imaging_npixel,
        filter_outlier=True,
        invert_ra=True,
    )

    # Eval
    assignments_restored = (
        SourceDetectionEvaluation.automatic_assignment_of_ground_truth_and_prediction(
            ground_truth=ground_truth.T,
            detected=detection_result.get_pixel_position_of_sources(),
            max_dist=10,
            top_k=3,
        )
    )

    # Plot
    # Create mapping plots
    sde_restored = SourceDetectionEvaluation(
        sky=sky,
        ground_truth=ground_truth,
        assignments=assignments_restored,
        sky_idxs=sky_idxs,
        source_detection=detection_result,
    )

    sde_restored.plot(filename="sources_restored.png")

    # Give out time
    print("Time taken: (minutes)", (time.time() - start) / 60)


if __name__ == "__main__":
    main(n_random_sources=20000)
