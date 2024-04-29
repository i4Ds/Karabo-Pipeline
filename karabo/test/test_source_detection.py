import os
import tempfile
from datetime import datetime

import numpy as np
import pytest

from karabo.data.external_data import (
    SingleFileDownloadObject,
    cscs_karabo_public_testing_base_url,
)
from karabo.imaging.image import Image
from karabo.imaging.imager import Imager
from karabo.imaging.imager_rascil import RascilImageCleaner, RascilImageCleanerConfig
from karabo.imaging.util import project_sky_to_image
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.sourcedetection.evaluation import SourceDetectionEvaluation
from karabo.sourcedetection.result import (
    PyBDSFSourceDetectionResult,
    PyBDSFSourceDetectionResultList,
    SourceDetectionResult,
)
from karabo.test.conftest import RUN_GPU_TESTS, NNImageDiffCallable, TFiles


@pytest.fixture
def restored_filtered_example_gleam() -> str:
    return "restored_filtered_example_gleam.fits"


@pytest.fixture
def test_restored_filtered_example_gleam_downloader(
    restored_filtered_example_gleam,
) -> SingleFileDownloadObject:
    return SingleFileDownloadObject(
        remote_file_path=restored_filtered_example_gleam,
        remote_base_url=cscs_karabo_public_testing_base_url,
    )


def test_source_detection_plot(
    tobject: TFiles, normalized_norm_diff: NNImageDiffCallable
):
    phase_center = [250, -80]
    sky = SkyModel.read_from_file(tobject.filtered_sky_csv)
    sky.setup_default_wcs(phase_center=phase_center)
    detection = SourceDetectionResult.read_from_file(tobject.detection_zip)
    with tempfile.TemporaryDirectory() as tmpdir:
        detection.write_to_file(os.path.join(tmpdir, "detection.zip"))

    img = detection.get_source_image()
    imaging_npixel = img.header["NAXIS1"]
    imaging_cellsize = img.get_cellsize()

    ground_truth, sky_idxs = project_sky_to_image(
        sky=sky,
        phase_center=phase_center,
        imaging_cellsize=imaging_cellsize,
        imaging_npixel=imaging_npixel,
        filter_outlier=True,
        invert_ra=True,
    )
    assignments = SourceDetectionEvaluation.automatic_assignment_of_ground_truth_and_prediction(  # noqa
        ground_truth=ground_truth.T,
        detected=detection.get_pixel_position_of_sources(),
        max_dist=10,
        top_k=3,
    )
    # Compare the assignment
    np.testing.assert_array_equal(
        assignments,
        np.load(tobject.gt_assigment),
        err_msg="The assignment has changed!",
    )
    mapping = SourceDetectionEvaluation(
        sky=sky,
        ground_truth=ground_truth,
        assignments=assignments,
        sky_idxs=sky_idxs,
        source_detection=detection,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        mapping.plot(filename=os.path.join(tmpdir, "plot.png"))
        mapping.plot_error_ra_dec(
            filename=os.path.join(tmpdir, "plot_error_ra_dec.png")
        )
        mapping.plot_quiver_positions(
            filename=os.path.join(tmpdir, "plot_quiver_positions.png")
        )
        mapping.plot_flux_ratio_to_distance(
            filename=os.path.join(tmpdir, "plot_flux_ratio_to_distance.png")
        )
        mapping.plot_flux_ratio_to_ra_dec(
            filename=os.path.join(tmpdir, "plot_flux_ratio_to_ra_dec.png")
        )
        mapping.plot_flux_histogram(
            filename=os.path.join(tmpdir, "plot_flux_histogram.png")
        )

        # Compare the images
        assert (
            normalized_norm_diff(os.path.join(tmpdir, "plot.png"), tobject.gt_plot)
            < 0.1
        )
        assert (
            normalized_norm_diff(
                os.path.join(tmpdir, "plot_error_ra_dec.png"),
                tobject.gt_plot_error_ra_dec,
            )
            < 0.1
        )
        assert (
            normalized_norm_diff(
                os.path.join(tmpdir, "plot_flux_ratio_to_distance.png"),
                tobject.gt_plot_flux_ratio_to_distance,
            )
            < 0.1
        )
        assert (
            normalized_norm_diff(
                os.path.join(tmpdir, "plot_flux_ratio_to_ra_dec.png"),
                tobject.gt_plot_flux_ratio_to_ra_dec,
            )
            < 0.1
        )
        assert (
            normalized_norm_diff(
                os.path.join(tmpdir, "plot_quiver_positions.png"),
                tobject.gt_plot_quiver_positions,
            )
            < 0.1
        )
        assert (
            normalized_norm_diff(
                os.path.join(tmpdir, "plot_flux_histogram.png"),
                tobject.gt_plot_flux_histogram,
            )
            < 0.1
        )


def test_bdsf_image_blanked():
    """
    Tests if bdsf error message in
    `PyBDSFSourceDetectionResult.detect_sources_in_image` has changed.
    If it has changed, it will throw a `RuntimeError`
    Test could maybe be changed if .fits file with blanked pixels is available and
    therefore you can just create an `Image` from that file
    """
    phase_center = [250, -80]
    gleam_sky = SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=80e6)
    sky = gleam_sky.filter_by_radius(0, 0.01, phase_center[0], phase_center[1])
    sky.setup_default_wcs(phase_center=phase_center)
    askap_tel = Telescope.constructor("ASKAP")
    observation_settings = Observation(
        start_frequency_hz=100e6,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=2,
        number_of_time_steps=24,
    )
    interferometer_sim = InterferometerSimulation(channel_bandwidth_hz=1e6)
    visibility_askap = interferometer_sim.run_simulation(
        askap_tel, sky, observation_settings
    )
    imaging_npixel = 512
    imaging_cellsize = 3.878509448876288e-05
    imager_askap = Imager(
        visibility_askap,
        imaging_npixel=imaging_npixel,
        imaging_cellsize=imaging_cellsize,
    )
    image_blanked = imager_askap.get_dirty_image()
    ret = PyBDSFSourceDetectionResult.detect_sources_in_image(image=image_blanked)
    if ret is not None:
        pytest.fail(
            "The return value is not None as expected due to PyBDSF RuntimeError!"
        )


def test_automatic_assignment_of_ground_truth_and_prediction():
    # Test that the automatic assignment of ground truth and prediction works
    # Create matrices of ground truth and prediction
    gtruth = np.random.randn(5000, 2) * 100
    detected = np.flipud(gtruth)

    # Calculate result
    assigment = SourceDetectionEvaluation.automatic_assignment_of_ground_truth_and_prediction(  # noqa
        gtruth, detected, 0.5, top_k=3
    )

    # Check that the result is correct by flipping the assigment
    # and checking that it is equal
    assert np.all(
        assigment[:, 0] == np.flipud(assigment[:, 1])
    ), "Automatic assignment of ground truth and detected is not correct"

    # Check reassigment of detected points, i.e. that the same detected point is
    # not assigned to multiple ground truth points
    # Create matrices of ground truth and prediction
    gtruth = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])

    detected = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])

    # Calculate result
    assigment = SourceDetectionEvaluation.automatic_assignment_of_ground_truth_and_prediction(  # noqa
        gtruth, detected, np.inf, top_k=4
    )

    # Check that the result is correct by flipping the assigment
    # and checking that it is equal
    assert np.all(
        assigment[:, 0] == np.flipud(assigment[:, 1])
    ), "Automatic assignment of ground truth and detected is not correct"


def test_full_source_detection(
    test_restored_filtered_example_gleam_downloader: SingleFileDownloadObject,
):
    restored = Image.read_from_file(
        test_restored_filtered_example_gleam_downloader.get()
    )
    detection_result = PyBDSFSourceDetectionResult.detect_sources_in_image(
        restored, thresh_isl=15, thresh_pix=20
    )
    gtruth = np.array(
        [
            [981.74904041, 843.23261492],
            [923.99869192, 856.80790319],
            [875.39219674, 889.2266872],
            [811.14161381, 929.42900662],
            [1018.00786977, 925.23273295],
            [1045.25482933, 1039.90727384],
            [1212.06660484, 930.03800074],
        ]
    )
    detected = detection_result.get_pixel_position_of_sources()
    mse = np.linalg.norm(gtruth - detected, axis=1)
    assert np.all(mse < 1), "Source detection is not correct"

    # Now compare it with splitting the image
    restored_cuts = restored.split_image(N=2, overlap=100)
    detection_results = PyBDSFSourceDetectionResultList.detect_sources_in_images(
        restored_cuts, thresh_isl=15, thresh_pix=20
    )
    detected = detection_results.get_pixel_position_of_sources()
    # Sometimes the order of the sources is different, so we need to sort them
    detected = detected[np.argsort(detected[:, 0])]
    gtruth = gtruth[np.argsort(gtruth[:, 0])]
    mse = np.linalg.norm(gtruth - detected, axis=1)
    assert np.all(mse < 1), "Source detection is not correct"


@pytest.mark.skipif(not RUN_GPU_TESTS, reason="GPU tests are disabled")
def test_create_detection_from_ms_cuda():
    phasecenter = np.array([225, -65])
    sky = SkyModel.get_random_poisson_disk_sky(
        phasecenter + np.array([-5, -5]),
        phasecenter + np.array([+5, +5]),
        100,
        200,
        0.4,
    )

    telescope = Telescope.constructor("MeerKAT")
    # telescope.centre_longitude = 3

    simulation = InterferometerSimulation(channel_bandwidth_hz=1e6, time_average_sec=1)

    observation = Observation(
        start_frequency_hz=100e6,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        phase_centre_ra_deg=phasecenter[0],
        phase_centre_dec_deg=phasecenter[1],
        number_of_time_steps=1,
        frequency_increment_hz=20e6,
        number_of_channels=3,
    )

    visibility = simulation.run_simulation(telescope, sky, observation)

    (
        convolved,
        restored,
        residual,
    ) = RascilImageCleaner().create_cleaned_image_variants(
        RascilImageCleanerConfig(
            imaging_npixel=2048,
            imaging_cellsize=0.0003,
            ms_file_path=visibility.ms_file_path,
            ingest_vis_nchan=3,
            ingest_average_blockvis=True,
            imaging_weighting="natural",
            imaging_robustness=-0.5,
            use_cuda=True,
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        convolved.write_to_file(os.path.join(tmpdir, "convolved.fits"))
        restored.write_to_file(os.path.join(tmpdir, "restored.fits"))
        residual.write_to_file(os.path.join(tmpdir, "residual.fits"))

        result = PyBDSFSourceDetectionResult.detect_sources_in_image(restored)
        result.write_to_file(os.path.join(tmpdir, "sources.zip"))
