import unittest

from karabo.Imaging import source_detection
from karabo.Imaging.image import open_fits_image
from karabo.Imaging.source_detection import read_detection_from_sources_file_csv
from karabo.simulation.sky_model import read_sky_model_from_csv


class TestSourceDetection(unittest.TestCase):

    def test_detection(self):
        image = open_fits_image("./data/restored.fits")
        detection = source_detection.detect_sources_in_image(image)
        pixels = detection.get_pixel_position_of_sources()
        print(pixels)

    def test_save_detection(self):
        image = open_fits_image("./data/restored.fits")
        detection = source_detection.detect_sources_in_image(image)
        detection.save_sources_file_as_csv("./result/detection.csv")

    def test_read_detection(self):
        detection = source_detection.SourceDetectionResult(file_path_csv="./data/detection.csv")
        assert len(detection.detected_sources) == 8

    def test_source_detection_plot(self):
        sky = read_sky_model_from_csv("./data/filtered_sky.csv")
        sky.setup_default_wcs([250, -80])
        detection = read_detection_from_sources_file_csv("./data/detection.csv",
                                                         source_image_path="./data/restored.fits")
        mapping = source_detection.map_sky_to_detection(sky, 3.878509448876288e-05, 2048, detection, 10)
        mapping.plot()

    def test_source_detection_plot_no_image(self):
        sky = read_sky_model_from_csv("./data/filtered_sky.csv")
        sky.setup_default_wcs([250, -80])
        detection = read_detection_from_sources_file_csv("./data/detection.csv")
        mapping = source_detection.map_sky_to_detection(sky, 3.878509448876288e-05, 2048, detection, 10)
        mapping.plot()
