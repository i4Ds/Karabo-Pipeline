import os
import unittest

from karabo.Imaging.image import Image
from karabo.Imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.sourcedetection import SourceDetectionEvaluation, read_detection_from_sources_file_csv, \
    SourceDetectionResult
from karabo.test import data_path


class TestSourceDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists('result/'):
            os.makedirs('result/')

        if not os.path.exists('result/test_dec'):
            os.makedirs('result/test_dec')

    # # TODO: move these on to CSCS Test Infrastructure once we have it.
    def test_detection(self):
        image = Image.read_from_file(f"{data_path}/restored.fits")
        detection = SourceDetectionResult.detect_sources_in_image(image)
        detection.__save_sources_to_csv(f"result/detection_result_512px.csv")
        detection.write_to_file('result/result.zip')
        # detection_read = PyBDSFSourceDetectionResult.open_from_file('result/result.zip')
        pixels = detection.get_pixel_position_of_sources()
        print(pixels)

    def test_create_detection_from_ms(self):
        phasecenter = (30, -20)
        sky = SkyModel.get_GLEAM_Sky()
        sky.filter_by_flux(0.4, 1)
        sky.plot_sky(phasecenter)
        sky.explore_sky(phasecenter, xlim=(-10, 10), ylim=(-10, 10))

        telescope = Telescope.get_MEERKAT_Telescope()
        # telescope.centre_longitude = 3

        simulation = InterferometerSimulation(channel_bandwidth_hz=1e6,
                                              time_average_sec=10)
        observation = Observation(100e6,
                                  phase_centre_ra_deg=phasecenter[0],
                                  phase_centre_dec_deg=phasecenter[1],
                                  number_of_time_steps=24,
                                  frequency_increment_hz=20e6,
                                  number_of_channels=64)

        visibility = simulation.run_simulation(telescope, sky, observation)
        visibility.write_to_file("./result/poisson_vis.ms")

        imager = Imager(visibility,
                        ingest_vis_nchan=16,
                        ingest_chan_per_blockvis=1,
                        ingest_average_blockvis=True,
                        imaging_npixel=2048,
                        imaging_cellsize=0.0003,
                        imaging_weighting='robust',
                        imaging_robustness=-.5)
        convolved, restored, residual = imager.imaging_rascil()

        convolved.write_to_file("result/test_dec/convolved.fits")
        restored.write_to_file("result/test_dec/restored.fits")
        residual.write_to_file("result/test_dec/residual.fits")

        result = SourceDetectionResult.detect_sources_in_image(restored)
        result.write_to_file("/result/test_dec/sources.csv")

    # def test_save_detection(self):
    #     image = open_fits_image("./data/restored.fits")
    #     detection = source_detection.detect_sources_in_image(image)
    #     detection.save_sources_file_as_csv("./result/detection.csv")

    def test_read_detection(self):
        detection = read_detection_from_sources_file_csv(filepath=f"{data_path}/detection_result_512px.csv")
        assert len(detection.detected_sources) == 37

    def test_source_detection_plot(self):
        sky = SkyModel.read_from_file(f"{data_path}/filtered_sky.csv")
        sky.setup_default_wcs([250, -80])
        detection = read_detection_from_sources_file_csv(f"{data_path}/detection_result_512px.csv",
                                                         source_image_path=f"{data_path}/restored.fits")
        detection.write_to_file("./result/detection.csv")
        mapping = SourceDetectionEvaluation.evaluate_result_with_sky_in_pixel_space(detection, sky, 5)
        mapping.plot()
        mapping.plot_error_ra_dec()
        mapping.plot_quiver_positions()
        mapping.plot_flux_ratio_to_distance()
        mapping.plot_flux_ratio_to_ra_dec()
        mapping.plot_flux_histogram()

    def test_get_arrays(self):
        sky = SkyModel.read_from_file(f"{data_path}/filtered_sky.csv")
        sky.setup_default_wcs([250, -80])
        detection = read_detection_from_sources_file_csv(f"{data_path}/detection_result_512px.csv",
                                                         source_image_path=f"{data_path}/restored.fits")
        detection.write_to_file("./result/detection.csv")

        mapping = SourceDetectionEvaluation.evaluate_result_with_sky_in_pixel_space(detection, sky, 10)
        arr = mapping.mapped_array
        print(arr)

    # def test_full_workflow(self):
    #     sky_data = np.array([
    #         [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
    #         [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
    #         [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10]])
    #     sky = sky_model.SkyModel(sky_data)
    #     askap_tel = telescope.get_ASKAP_Telescope()
    #     observation_settings = observation.Observation(100e6,
    #                                                    phase_centre_ra_deg=20,
    #                                                    phase_centre_dec_deg=-30,
    #                                                    number_of_channels=64,
    #                                                    number_of_time_steps=24)
    #
    #     interferometer_sim = interferometer.InterferometerSimulation(channel_bandwidth_hz=1e6)
    #     visibility_askap = interferometer_sim.run_simulation(askap_tel, sky, observation_settings)
    #     imager_askap = imager.Imager(visibility_askap, imaging_npixel=2048,
    #                                  imaging_cellsize=3.878509448876288e-05)
    #     imager_askap.ingest_chan_per_blockvis = 1
    #     imager_askap.ingest_vis_nchan = 16
    #     deconvolved, restored, residual = imager_askap.imaging_rascil(
    #         clean_nmajor=0,
    #         clean_algorithm='mmclean',
    #         clean_scales=[0, 6, 10, 30, 60],
    #         clean_fractional_threshold=.3,
    #         clean_threshold=.12e-3,
    #         clean_nmoment=5,
    #         clean_psf_support=640,
    #         clean_restored_output='integrated')
    #     # detection_result = source_detection.detect_sources_in_image(restored)
    #     # restored.save_as_fits("./restored_corrupt.fits")
    #     # deconvolved.save_as_fits("./deconv_corrupt.fits")
    #     # residual.save_as_fits("./redisual_corrupt.fits")
    #     restored.plot()
    #     restored.plot()
    #     # deconvolved.plot()
    #     # residual.plot()
    #     # sky.setup_default_wcs()
    #     # detection_evaluation = source_detection.map_sky_to_detection(sky,
    #     #                                                              3.878509448876288e-05,
    #     #                                                              2048,
    #     #                                                              detection_result,
    #     #                                                              10)
    #     # detection_evaluation.plot()
    #
    #
    # def test_full_workflo2w(self):
    #     sky_data = np.array([
    #         [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
    #         [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
    #         [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10]])
    #     sky = sky_model.SkyModel(sky_data)
    #     askap_tel = telescope.get_ASKAP_Telescope()
    #     observation_settings = observation.Observation(100e6,
    #                                                    phase_centre_ra_deg=20,
    #                                                    phase_centre_dec_deg=-30,
    #                                                    number_of_channels=64,
    #                                                    number_of_time_steps=24)
    #
    #     interferometer_sim = interferometer.InterferometerSimulation(channel_bandwidth_hz=1e6)
    #     visibility_askap = interferometer_sim.run_simulation(askap_tel, sky, observation_settings)
    #     imager_askap = imager.Imager(visibility_askap, imaging_npixel=2048,
    #                                  imaging_cellsize=3.878509448876288e-05)
    #     imager_askap.ingest_chan_per_blockvis = 1
    #     imager_askap.ingest_vis_nchan = 16
    #     deconvolved, restored, residual = imager_askap.imaging_rascil(
    #         clean_nmajor=0,
    #         clean_algorithm='mmclean',
    #         clean_scales=[0, 6, 10, 30, 60],
    #         clean_fractional_threshold=.3,
    #         clean_threshold=.12e-3,
    #         clean_nmoment=5,
    #         clean_psf_support=640,
    #         clean_restored_output='integrated')
    #     restored.save_as_fits("./restored_before.fits")
    #     detection_result = source_detection.detect_sources_in_image(restored)
    #     restored.save_as_fits("./restored_after.fits")
    #     # deconvolved.save_as_fits("./deconv_corrupt.fits")
    #     # residual.save_as_fits("./redisual_corrupt.fits")
    #     # restored.plot()
    #     # restored.plot()
    #     # deconvolved.plot()
    #     # residual.plot()
    #     sky.setup_default_wcs()
    #     detection_evaluation = source_detection.map_sky_to_detection(sky,
    #                                                                  3.878509448876288e-05,
    #                                                                  2048,
    #                                                                  detection_result,
    #                                                                  10)
    #     detection_evaluation.plot()
    #
