import os
import time

from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.sourcedetection.evaluation import SourceDetectionEvaluation
from karabo.sourcedetection.result import PyBDSFSourceDetectionResult
from karabo.util.dask import setup_dask_for_slurm


def main():
    # Setup Dask
    client = setup_dask_for_slurm()

    # Print out the slurm node name this scirpt is running on
    print(f'Node Name for main script: {os.getenv("SLURMD_NODENAME")}')
    start = time.time()
    # Get GLEAM Survey Sky
    phase_center = [250, -80]
    gleam_sky = SkyModel.get_GLEAM_Sky()
    gleam_sky.explore_sky(phase_center, s=0.1)

    sky = gleam_sky.filter_by_radius(0, 0.5, phase_center[0], phase_center[1])
    sky.setup_default_wcs(phase_center=phase_center)
    askap_tel = Telescope.get_ASKAP_Telescope()

    observation_settings = Observation(
        start_frequency_hz=100e6,
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=64,
        number_of_time_steps=24,
    )

    interferometer_sim = InterferometerSimulation(channel_bandwidth_hz=1e6)
    visibility_askap = interferometer_sim.run_simulation(
        askap_tel, sky, observation_settings
    )

    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    imager_askap = Imager(
        visibility_askap,
        ingest_chan_per_vis=1,
        ingest_vis_nchan=16,
        imaging_npixel=imaging_npixel,
        imaging_cellsize=imaging_cellsize,
    )

    # Try differnet algorithm
    # More sources
    deconvolved, restored, residual = imager_askap.imaging_rascil(
        clean_nmajor=0,
        clean_algorithm="mmclean",
        clean_scales=[0, 6, 10, 30, 60],
        clean_fractional_threshold=0.3,
        clean_threshold=0.12e-3,
        clean_nmoment=5,
        clean_psf_support=640,
        clean_restored_output="integrated",
        use_cuda=False,
        use_dask=True,
        client=client,
    )

    # Source detection
    detection_result = PyBDSFSourceDetectionResult.detect_sources_in_image(restored)

    ground_truth, sky_idxs = Imager.project_sky_to_image(
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
            detected=detection_result.get_pixel_position_of_sources().T,
            max_dist=10,
            top_k=3,
        )
    )
    print(assignments_restored)

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
    main()
