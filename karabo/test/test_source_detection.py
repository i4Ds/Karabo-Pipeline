import os
import unittest

import numpy as np

from karabo.imaging.image import Image
from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.sourcedetection.evaluation import SourceDetectionEvaluation
from karabo.sourcedetection.result import (
    SourceDetectionResult,
)
from karabo.test import data_path


class TestSourceDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists("result/"):
            os.makedirs("result/")

        if not os.path.exists("result/test_dec"):
            os.makedirs("result/test_dec")

    # # # TODO: move these on to CSCS Test Infrastructure once we have it.
    # def test_detection(self):
    #     image = Image.read_from_file(f"{data_path}/restored.fits")
    #     detection = SourceDetectionResult.detect_sources_in_image(image)
    #     detection.write_to_file("result/result.zip")
    #     # detection_read = PyBDSFSourceDetectionResult.open_from_file('result/result.zip')
    #     pixels = detection.get_pixel_position_of_sources()
    #     print(pixels)
    #
    # def test_create_detection_from_ms(self):
    #     phasecenter = np.array([225, -65])
    #     sky = SkyModel.get_random_poisson_disk_sky(
    #         phasecenter + np.array([-5, -5]),
    #         phasecenter + np.array([+5, +5]),
    #         100,
    #         200,
    #         0.4,
    #     )
    #     print(sky.sources)
    #     # sky = SkyModel.get_GLEAM_Sky()
    #     # sky.filter_by_flux(0.4, 1)
    #     sky.plot_sky(phasecenter)
    #     sky.explore_sky(phasecenter, xlim=(-10, 10), ylim=(-10, 10))
    #
    #     telescope = Telescope.get_MEERKAT_Telescope()
    #     # telescope.centre_longitude = 3
    #
    #     simulation = InterferometerSimulation(
    #         channel_bandwidth_hz=1e6, time_average_sec=10
    #     )
    #     observation = Observation(
    #         100e6,
    #         phase_centre_ra_deg=phasecenter[0],
    #         phase_centre_dec_deg=phasecenter[1],
    #         number_of_time_steps=1,
    #         frequency_increment_hz=20e6,
    #         number_of_channels=1,
    #     )
    #
    #     visibility = simulation.run_simulation(telescope, sky, observation)
    #     visibility.write_to_file("./result/test_dec/poisson_vis.ms")
    #
    #     imager = Imager(
    #         visibility,
    #         ingest_vis_nchan=1,
    #         ingest_chan_per_blockvis=1,
    #         ingest_average_blockvis=True,
    #         imaging_npixel=2048,
    #         imaging_cellsize=0.0003,
    #         imaging_weighting="natural",
    #         imaging_robustness=-0.5,
    #     )
    #     convolved, restored, residual = imager.imaging_rascil()
    #
    #     convolved.write_to_file("result/test_dec/convolved.fits")
    #     restored.write_to_file("result/test_dec/restored.fits")
    #     residual.write_to_file("result/test_dec/residual.fits")
    #
    #     result = SourceDetectionResult.detect_sources_in_image(restored)
    #     result.write_to_file("result/test_dec/sources.zip")
    #
    #     evaluation = SourceDetectionEvaluation.evaluate_result_with_sky_in_pixel_space(
    #         result, sky, 1
    #     )
    #     evaluation.plot(filename="result/test_dec/matching_plot.png")
    #     evaluation.plot_error_ra_dec(filename="result/test_dec/error_ra_dec_plot.png")
    #     evaluation.plot_quiver_positions(filename="result/test_dec/quiver_position.png")
    #     evaluation.plot_flux_histogram(filename="result/test_dec/flux_histogram.png")
    #     evaluation.plot_flux_ratio_to_distance(filename="result/test_dec/flux_ratio_distance.png")
    #     evaluation.plot_flux_ratio_to_ra_dec(filename="result/test_dec/flux_ratio_ra_dec.png")

    def test_source_detection_plot(self):
        sky = SkyModel.read_from_file(f"{data_path}/filtered_sky.csv")
        sky.setup_default_wcs([250, -80])
        detection = SourceDetectionResult.read_from_file(
            f"{data_path}/detection.zip",
        )
        detection.write_to_file("./result/detection.zip")
        mapping = SourceDetectionEvaluation.evaluate_result_with_sky_in_pixel_space(
            detection, sky, 5
        )
        mapping.plot()
        mapping.plot_error_ra_dec()
        mapping.plot_quiver_positions()
        mapping.plot_flux_ratio_to_distance()
        mapping.plot_flux_ratio_to_ra_dec()
        mapping.plot_flux_histogram()
    
    def test_automatic_assignment_of_ground_truth_and_prediction(self):
        ## Test that the automatic assignment of ground truth and prediction works
        # Create matrices of ground truth and prediction
        gtruth = np.random.randn(5000, 2) * 100
        detected = np.flipud(gtruth)

        # Calculate result
        assigment = SourceDetectionEvaluation.automatic_assignment_of_ground_truth_and_prediction(gtruth, detected, 0.5, top_k=3)

        # Check that the result is correct by flipping the assigment and checking that it is equal
        assert np.all(assigment[:,0]==np.flipud(assigment[:,1])), "Automatic assignment of ground truth and detected is not correct"

        ## Check reassigment of detected points, i.e. that the same detected point is not assigned to multiple ground truth points
        # Create matrices of ground truth and prediction
        gtruth = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0]
            ]
        )

        detected = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0]
            ]
        )

        # Calculate result
        assigment = SourceDetectionEvaluation.automatic_assignment_of_ground_truth_and_prediction(gtruth, detected, np.inf, top_k=4)

        # Check that the result is correct by flipping the assigment and checking that it is equal
        assert np.all(assigment[:,0]==np.flipud(assigment[:,1])), "Automatic assignment of ground truth and detected is not correct"