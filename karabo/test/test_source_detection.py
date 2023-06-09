import os

import numpy as np
import pytest

from karabo.imaging.image import Image
from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.sourcedetection.evaluation import SourceDetectionEvaluation
from karabo.sourcedetection.result import (
    PyBDSFSourceDetectionResult,
    SourceDetectionResult,
)
from karabo.test import data_path

RUN_GPU_TESTS = os.environ.get("RUN_GPU_TESTS", "false").lower() == "true"


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    # make dir for result files
    if not os.path.exists("result/"):
        os.makedirs("result/")

    if not os.path.exists("result/test_dec"):
        os.makedirs("result/test_dec")


@pytest.mark.skipif(not RUN_GPU_TESTS, reason="GPU tests are disabled")
def test_source_detection_plot():
    phase_center = [250, -80]
    sky = SkyModel.read_from_file(f"{data_path}/filtered_sky.csv")
    sky.setup_default_wcs(phase_center=phase_center)
    detection = SourceDetectionResult.read_from_file(f"{data_path}/detection.zip")
    detection.write_to_file("./result/detection.zip")

    img = detection.get_source_image()
    imaging_npixel = img.header["NAXIS1"]
    imaging_cellsize = img.get_cellsize()

    ground_truth, sky_idxs = Imager.project_sky_to_image(
        sky=sky,
        phase_center=phase_center,
        imaging_cellsize=imaging_cellsize,
        imaging_npixel=imaging_npixel,
        filter_outlier=True,
        invert_ra=True,
    )
    assignments = SourceDetectionEvaluation.automatic_assignment_of_ground_truth_and_prediction(  # noqa
        ground_truth=ground_truth.T,
        detected=detection.get_pixel_position_of_sources().T,
        max_dist=10,
        top_k=3,
    )
    mapping = SourceDetectionEvaluation(
        sky=sky,
        ground_truth=ground_truth,
        assignments=assignments,
        sky_idxs=sky_idxs,
        source_detection=detection,
    )
    mapping.plot()
    mapping.plot_error_ra_dec()
    mapping.plot_quiver_positions()
    mapping.plot_flux_ratio_to_distance()
    mapping.plot_flux_ratio_to_ra_dec()
    mapping.plot_flux_histogram()


def test_bdsf_image_blanked():
    """
    Tests if bdsf error message in
    `PyBDSFSourceDetectionResult.detect_sources_in_image` has changed.
    If it has changed, it will throw a `RuntimeError`
    Test could maybe be changed if .fits file with blanked pixels is available and
    therefore you can just create an `Image` from that file
    """
    phase_center = [250, -80]
    gleam_sky = SkyModel.get_GLEAM_Sky([76])
    sky = gleam_sky.filter_by_radius(0, 0.01, phase_center[0], phase_center[1])
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
        imaging_npixel=imaging_npixel,
        imaging_cellsize=imaging_cellsize,
    )
    image_blanked = imager_askap.get_dirty_image()
    beam_guess = (0.06414627663254034, 0.05891435806172773, 69.63573045562626)
    ret = PyBDSFSourceDetectionResult.detect_sources_in_image(
        image=image_blanked, beam=beam_guess
    )
    if ret is not None:
        raise Exception(
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


@pytest.mark.skipif(not RUN_GPU_TESTS, reason="GPU tests are disabled")
def test_detection():
    image = Image(path=f"{data_path}/restored.fits")
    detection = PyBDSFSourceDetectionResult.detect_sources_in_image(image)
    detection.write_to_file("result/result.zip")
    # detection_read = PyBDSFSourceDetectionResult.open_from_file('result/result.zip')  # noqa
    pixels = detection.get_pixel_position_of_sources()
    print(pixels)


@pytest.mark.skipif(not RUN_GPU_TESTS, reason="GPU tests are disabled")
def test_create_detection_from_ms_dask():
    phasecenter = np.array([225, -65])
    sky = SkyModel.get_random_poisson_disk_sky(
        phasecenter + np.array([-5, -5]),
        phasecenter + np.array([+5, +5]),
        100,
        200,
        0.4,
    )

    # sky = SkyModel.get_GLEAM_Sky([76])
    # sky.filter_by_flux(0.4, 1)
    sky.explore_sky(phasecenter)

    telescope = Telescope.get_MEERKAT_Telescope()
    # telescope.centre_longitude = 3

    simulation = InterferometerSimulation(channel_bandwidth_hz=1e6, time_average_sec=1)

    observation = Observation(
        start_frequency_hz=100e6,
        phase_centre_ra_deg=phasecenter[0],
        phase_centre_dec_deg=phasecenter[1],
        number_of_time_steps=1,
        frequency_increment_hz=20e6,
        number_of_channels=3,
    )

    visibility = simulation.run_simulation(telescope, sky, observation)
    visibility.write_to_file("./result/test_dec/poisson_vis.ms")

    imager = Imager(
        visibility,
        ingest_vis_nchan=3,
        ingest_chan_per_vis=1,
        ingest_average_blockvis=True,
        imaging_npixel=2048,
        imaging_cellsize=0.0003,
        imaging_weighting="natural",
        imaging_robustness=-0.5,
    )
    # Get Dask client
    print("Starting Rascil with Dask.")
    convolved, restored, residual = imager.imaging_rascil(use_dask=True, use_cuda=False)

    convolved.write_to_file("result/test_dec/convolved.fits")
    restored.write_to_file("result/test_dec/restored.fits")
    residual.write_to_file("result/test_dec/residual.fits")

    result = PyBDSFSourceDetectionResult.detect_sources_in_image(restored)
    result.write_to_file("result/test_dec/sources.zip")

    evaluation = SourceDetectionEvaluation.evaluate_result_with_sky_in_pixel_space(
        result, sky, 1
    )
    evaluation.plot(filename="result/test_dec/matching_plot.png")
    evaluation.plot_error_ra_dec(filename="result/test_dec/error_ra_dec_plot.png")
    evaluation.plot_quiver_positions(filename="result/test_dec/quiver_position.png")
    evaluation.plot_flux_histogram(filename="result/test_dec/flux_histogram.png")
    evaluation.plot_flux_ratio_to_distance(
        filename="result/test_dec/flux_ratio_distance.png"
    )
    evaluation.plot_flux_ratio_to_ra_dec(
        filename="result/test_dec/flux_ratio_ra_dec.png"
    )


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

    # sky = SkyModel.get_GLEAM_Sky([76])
    # sky.filter_by_flux(0.4, 1)
    sky.explore_sky(phasecenter)

    telescope = Telescope.get_MEERKAT_Telescope()
    # telescope.centre_longitude = 3

    simulation = InterferometerSimulation(channel_bandwidth_hz=1e6, time_average_sec=1)

    observation = Observation(
        start_frequency_hz=100e6,
        phase_centre_ra_deg=phasecenter[0],
        phase_centre_dec_deg=phasecenter[1],
        number_of_time_steps=1,
        frequency_increment_hz=20e6,
        number_of_channels=3,
    )

    visibility = simulation.run_simulation(telescope, sky, observation)
    visibility.write_to_file("./result/test_dec/poisson_vis.ms")

    imager = Imager(
        visibility,
        ingest_vis_nchan=3,
        ingest_chan_per_blockvis=1,
        ingest_average_blockvis=True,
        imaging_npixel=2048,
        imaging_cellsize=0.0003,
        imaging_weighting="natural",
        imaging_robustness=-0.5,
    )

    convolved, restored, residual = imager.imaging_rascil(
        client=None, use_dask=False, use_cuda=True
    )

    convolved.write_to_file("result/test_dec/convolved.fits")
    restored.write_to_file("result/test_dec/restored.fits")
    residual.write_to_file("result/test_dec/residual.fits")

    result = PyBDSFSourceDetectionResult.detect_sources_in_image(restored)
    result.write_to_file("result/test_dec/sources.zip")

    evaluation = SourceDetectionEvaluation.evaluate_result_with_sky_in_pixel_space(
        result, sky, 1
    )
    evaluation.plot(filename="result/test_dec/matching_plot.png")
    evaluation.plot_error_ra_dec(filename="result/test_dec/error_ra_dec_plot.png")
    evaluation.plot_quiver_positions(filename="result/test_dec/quiver_position.png")
    evaluation.plot_flux_histogram(filename="result/test_dec/flux_histogram.png")
    evaluation.plot_flux_ratio_to_distance(
        filename="result/test_dec/flux_ratio_distance.png"
    )
    evaluation.plot_flux_ratio_to_ra_dec(
        filename="result/test_dec/flux_ratio_ra_dec.png"
    )


@pytest.mark.skipif(not RUN_GPU_TESTS, reason="GPU tests are disabled")
def test_create_image_waag_gridder():
    import wagg as wg

    # Read test data.
    test_data = np.load(f"{data_path}/vla_d.npz")
    vis = test_data["vis"]
    freqs = test_data["freqs"]
    uvw = test_data["uvw"]

    # Image parameters.
    image_size = 1024
    pixsize_deg = 1.94322419749866394e-02
    pixsize_rad = pixsize_deg * np.pi / 180.0

    # Convert data to single precision.
    vis = vis.astype(np.complex64)
    weights = np.ones_like(vis, dtype=np.float32)
    epsilon = 1e-6

    image_input = wg.ms2dirty(
        uvw,
        freqs,
        vis,
        weights,
        image_size,
        image_size,
        pixsize_rad,
        pixsize_rad,
        epsilon,
        False,
    )

    vis_gpu = wg.dirty2ms(
        uvw, freqs, image_input, weights, pixsize_rad, pixsize_rad, epsilon, False
    )

    image_gpu = wg.ms2dirty(
        uvw,
        freqs,
        vis_gpu,
        weights,
        image_size,
        image_size,
        pixsize_rad,
        pixsize_rad,
        epsilon,
        False,
    )
    image_gpu /= len(vis_gpu)
    image_input /= len(vis)

    MAE = np.linalg.norm(image_gpu - image_input, ord=1) / (
        np.shape(image_gpu)[0] * np.shape(image_gpu)[1]
    )

    # Asserts
    assert np.sum(np.isinf(image_gpu)) == 0
    assert np.sum(np.isnan(image_gpu)) == 0
    assert type(image_gpu) == np.ndarray
    assert MAE < 1.0e-02, "WAGG does not correctly reconstruct the image"
