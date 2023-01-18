import os, pickle
import unittest

import numpy as np

from karabo.imaging.imager import Imager
from karabo.imaging.image import Image
from karabo.simulation.sky_model import SkyModel
from karabo.sourcedetection.evaluation import SourceDetectionEvaluation
from karabo.sourcedetection.result import SourceDetectionResult, PyBDSFSourceDetectionResult

from karabo.test import data_path

# import wagg as wg #This gives an ImportError if no GPU is available
import numpy as np

class TestSourceDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists("result/"):
            os.makedirs("result/")

        if not os.path.exists("result/test_dec"):
            os.makedirs("result/test_dec")
            
    def test_source_detection_plot(self):
        phase_center = [250, -80]
        sky = SkyModel.read_from_file(f"{data_path}/filtered_sky.csv")
        sky.setup_default_wcs(phase_center=phase_center)
        detection = SourceDetectionResult.read_from_file(f"{data_path}/detection.zip")
        detection.write_to_file("./result/detection.zip")

        img = detection.get_source_image()
        imaging_npixel = img.header['NAXIS1']
        imaging_cellsize = img.get_cellsize()

        ground_truth, sky_idxs = Imager.project_sky_to_image(
            sky=sky,
            phase_center=phase_center,
            imaging_cellsize=imaging_cellsize,
            imaging_npixel=imaging_npixel,
            filter_outlier=True,
            invert_ra=True,
        )
        assignments = SourceDetectionEvaluation.automatic_assignment_of_ground_truth_and_prediction(
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

    def test_bdsf_image_blanked(self):
        """
        Tests if bdsf error message in `PyBDSFSourceDetectionResult.detect_sources_in_image` has changed
        If it has changed, it will throw a `RuntimeError`
        """
        with open('image_blank.pkl', 'rb') as f:
            image_blanked: Image = pickle.load(f)
        beam_guess = (0.06414627663254034, 0.05891435806172773, 69.63573045562626)
        PyBDSFSourceDetectionResult.detect_sources_in_image(image=image_blanked, beam=beam_guess)
        
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
        
    # # # TODO: move these on to CSCS Test Infrastructure once we have it.
    # def test_detection(self):
    #     image = Image(path=f"{data_path}/restored.fits")
    #     detection = PyBDSFSourceDetectionResult.detect_sources_in_image(image)
    #     detection.write_to_file("result/result.zip")
    #     # detection_read = PyBDSFSourceDetectionResult.open_from_file('result/result.zip')
    #     pixels = detection.get_pixel_position_of_sources()
    #     print(pixels)
    
    
    #TODO: Investigate why this error sometimes fails and sometimes doesn't, especially when running it on github hosted instances.
    """
    def test_create_detection_from_ms_small(self):
        phasecenter = np.array([225, -65])
        np.random.seed(0)
        sky = SkyModel.get_random_poisson_disk_sky(
            phasecenter + np.array([-0.1, -0.1]),
            phasecenter + np.array([+0.2, +0.2]),
            100,
            200,
            0.4,
        )

        sky.explore_sky(phasecenter)
    
        telescope = Telescope.get_MEERKAT_Telescope()
    
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=1e6,
            time_average_sec=1
            )

        observation = Observation(
            start_frequency_hz=100e6,
            phase_centre_ra_deg=phasecenter[0],
            phase_centre_dec_deg=phasecenter[1],
            number_of_time_steps=1,
            frequency_increment_hz=20e6,
            number_of_channels=1
        )
    
        visibility = simulation.run_simulation(telescope, sky, observation)
        visibility.write_to_file("./result/test_dec/poisson_vis.ms")
    
        imager = Imager(
            visibility,
            ingest_vis_nchan=1,
            ingest_chan_per_blockvis=1,
            ingest_average_blockvis=True,
            imaging_npixel=2*3*5,
            imaging_cellsize=0.0003,
            imaging_weighting="natural",
            imaging_robustness=-0.5,
        )
        convolved, restored, residual = imager.imaging_rascil()
    
        convolved.write_to_file("result/test_dec/convolved.fits")
        restored.write_to_file("result/test_dec/restored.fits")
        residual.write_to_file("result/test_dec/residual.fits")
    
        result = PyBDSFSourceDetectionResult.detect_sources_in_image(restored)
        result.write_to_file("result/test_dec/sources.zip")
    
        evaluation = SourceDetectionEvaluation.evaluate_result_with_sky_in_pixel_space( # DOESN't WORK ANYMORE
            result, sky, 10
        )
        evaluation.plot(filename="result/test_dec/matching_plot.png")
        evaluation.plot_error_ra_dec(filename="result/test_dec/error_ra_dec_plot.png")
        evaluation.plot_quiver_positions(filename="result/test_dec/quiver_position.png")

        evaluation.plot_flux_histogram(filename="result/test_dec/flux_histogram.png")
        evaluation.plot_flux_ratio_to_distance(filename="result/test_dec/flux_ratio_distance.png")
        evaluation.plot_flux_ratio_to_ra_dec(filename="result/test_dec/flux_ratio_ra_dec.png")
        
        
        TODO: DEPLOY ON CSCS:
    def test_create_detection_from_ms_dask(self):
        phasecenter = np.array([225, -65])
        sky = SkyModel.get_random_poisson_disk_sky(
            phasecenter + np.array([-5, -5]),
            phasecenter + np.array([+5, +5]),
            100,
            200,
            0.4,
        )

        # sky = SkyModel.get_GLEAM_Sky()
        # sky.filter_by_flux(0.4, 1)
        sky.explore_sky(phasecenter)
    
        telescope = Telescope.get_MEERKAT_Telescope()
        # telescope.centre_longitude = 3
    
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=1e6,
            time_average_sec=1
            )

        observation = Observation(
            start_frequency_hz=100e6,
            phase_centre_ra_deg=phasecenter[0],
            phase_centre_dec_deg=phasecenter[1],
            number_of_time_steps=1,
            frequency_increment_hz=20e6,
            number_of_channels=3
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
        # Get Dask client
        print('Starting Rascil with Dask.')
        client = get_global_client(2, 2)
        convolved, restored, residual = imager.imaging_rascil(
            client=client,
            use_dask=True, 
            use_cuda=False)
        
        convolved.write_to_file("result/test_dec/convolved.fits")
        restored.write_to_file("result/test_dec/restored.fits")
        residual.write_to_file("result/test_dec/residual.fits")
    
        result = PyBDSFSourceDetectionResult.detect_sources_in_image(restored)
        result.write_to_file("result/test_dec/sources.zip")
    
        evaluation = SourceDetectionEvaluation.evaluate_result_with_sky_in_pixel_space( # DOESN't WORK ANYMORE
            result, sky, 1
        )
        evaluation.plot(filename="result/test_dec/matching_plot.png")
        evaluation.plot_error_ra_dec(filename="result/test_dec/error_ra_dec_plot.png")
        evaluation.plot_quiver_positions(filename="result/test_dec/quiver_position.png")
        evaluation.plot_flux_histogram(filename="result/test_dec/flux_histogram.png")
        evaluation.plot_flux_ratio_to_distance(filename="result/test_dec/flux_ratio_distance.png")
        evaluation.plot_flux_ratio_to_ra_dec(filename="result/test_dec/flux_ratio_ra_dec.png")
        
    def test_create_detection_from_ms_cuda(self):
        phasecenter = np.array([225, -65])
        sky = SkyModel.get_random_poisson_disk_sky(
            phasecenter + np.array([-5, -5]),
            phasecenter + np.array([+5, +5]),
            100,
            200,
            0.4,
        )

        # sky = SkyModel.get_GLEAM_Sky()
        # sky.filter_by_flux(0.4, 1)
        sky.explore_sky(phasecenter)
    
        telescope = Telescope.get_MEERKAT_Telescope()
        # telescope.centre_longitude = 3
    
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=1e6,
            time_average_sec=1
            )

        observation = Observation(
            start_frequency_hz=100e6,
            phase_centre_ra_deg=phasecenter[0],
            phase_centre_dec_deg=phasecenter[1],
            number_of_time_steps=1,
            frequency_increment_hz=20e6,
            number_of_channels=3
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
            client=None,
            use_dask=False, 
            use_cuda=True)
        
        convolved.write_to_file("result/test_dec/convolved.fits")
        restored.write_to_file("result/test_dec/restored.fits")
        residual.write_to_file("result/test_dec/residual.fits")
    
        result = PyBDSFSourceDetectionResult.detect_sources_in_image(restored)
        result.write_to_file("result/test_dec/sources.zip")
    
        evaluation = SourceDetectionEvaluation.evaluate_result_with_sky_in_pixel_space( # DOESN't WORK ANYMORE
            result, sky, 1
        )
        evaluation.plot(filename="result/test_dec/matching_plot.png")
        evaluation.plot_error_ra_dec(filename="result/test_dec/error_ra_dec_plot.png")
        evaluation.plot_quiver_positions(filename="result/test_dec/quiver_position.png")
        evaluation.plot_flux_histogram(filename="result/test_dec/flux_histogram.png")
        evaluation.plot_flux_ratio_to_distance(filename="result/test_dec/flux_ratio_distance.png")
        evaluation.plot_flux_ratio_to_ra_dec(filename="result/test_dec/flux_ratio_ra_dec.png")

    def test_source_detection_plot(self):
        sky = SkyModel.read_from_file(f"{data_path}/filtered_sky.csv")
        sky.setup_default_wcs([250, -80])
        detection = SourceDetectionResult.read_from_file(
            f"{data_path}/detection.zip",
        )
        detection.write_to_file("./result/detection.zip")
        mapping = SourceDetectionEvaluation.evaluate_result_with_sky_in_pixel_space( # DOESN't WORK ANYMORE
            detection, sky, 5
        )
        mapping.plot()
        mapping.plot_error_ra_dec()
        mapping.plot_quiver_positions()
        mapping.plot_flux_ratio_to_distance()
        mapping.plot_flux_ratio_to_ra_dec()
        mapping.plot_flux_histogram()
        
        # TODO: Deploy this somewhere with a GPU (CSCS?)
    def test_create_image_waag_gridder(self):
        # Read test data.
        test_data = np.load(f"{data_path}/vla_d.npz")
        vis = test_data["vis"]
        freqs = test_data["freqs"]
        uvw = test_data["uvw"]

        # Image parameters.
        image_size = 1024
        pixsize_deg = 1.94322419749866394E-02
        pixsize_rad = pixsize_deg * np.pi / 180.0

        # Convert data to single precision.
        vis = vis.astype(np.complex64)
        weights = np.ones_like(vis, dtype=np.float32)
        epsilon = 1e-6
        
        image_input = wg.ms2dirty(uvw, freqs, vis, weights, image_size, image_size,
                            pixsize_rad, pixsize_rad, epsilon, False)
        
        vis_gpu = wg.dirty2ms(uvw, freqs, image_input, weights,
                        pixsize_rad, pixsize_rad, epsilon, False)
        
        image_gpu = wg.ms2dirty(uvw, freqs, vis_gpu, weights, image_size, image_size,
                                pixsize_rad, pixsize_rad, epsilon, False)
        image_gpu /= len(vis_gpu)
        image_input /= len(vis)
        
        MAE = np.linalg.norm(image_gpu-image_input, ord=1) / (np.shape(image_gpu)[0]*np.shape(image_gpu)[1])
        
        ## Asserts
        assert np.sum(np.isinf(image_gpu)) == 0
        assert np.sum(np.isnan(image_gpu)) == 0
        assert type(image_gpu) == np.ndarray
        assert MAE < 1.e-02, "WAGG does not correctly reconstruct the image" 
        """

        
        