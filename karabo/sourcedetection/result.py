import shutil

import numpy
import numpy as np
import bdsf
from bdsf import image as bdsf_image

from karabo.imaging.image import Image
from karabo.karabo_resource import KaraboResource
from karabo.util.FileHandle import FileHandle
from karabo.util.data_util import read_CSV_to_ndarray


class SourceDetectionResult(KaraboResource):
    def __init__(self, detected_sources: np.ndarray, source_image: Image):
        """
        Generic Source Detection Result Class.
        Inputting your Source Detection Result as an Array with specified shape and columns

        +-------+----+-----+----------------+--------------+------------+-----------+
        | index | ra | dec | pos X (pixel) | pos Y (pixel) | total_flux | peak_flux |
        +=======+====+=====+================+==============+============+===========+
        | 0     | 30 | 200 | 400           | 500           | 0.345     |    0.34540 |
        +-------+----+-----+----------------+--------------+------------+-----------+

        Rows can also be left empty if the specified value is not found by your source detection algorithm.
        More rows can also be added at the end. As they are not used for any internal algorithm.

        :param detected_sources: detected sources in array
        :param source_image: Image, where the source detection was performed on
        """
        self.source_image = source_image
        self.detected_sources = detected_sources

    @staticmethod
    def detect_sources_in_image(image: Image, beam=None, quiet=False) -> any:
        """
        Detecting sources in an image. The Source detection is implemented with the PyBDSF.process_image function.
        See https://www.astron.nl/citt/pybdsf/process_image.html for more information.

        :param image: Image to perform source detection on
        :param beam: FWHM of restoring beam. Specify as (maj, min. pos angle E of N).
                     None means it will try to be extracted from the Image data. (Might fail)
        :return: Source Detection Result containing the found sources
        """
        detection = bdsf.process_image(
            image.file.path, beam=beam, quiet=quiet, format="csv"
        )
        return PyBDSFSourceDetectionResult(detection)

    def write_to_file(self, path: str) -> None:
        """
        Save Source Detection Result to ZIP Archive containing the .fits source image and source-finding catalog.
        :param path: path to save the zip archive as.
        """
        if path.endswith(".zip"):
            path = path[0 : len(path) - 4]
        tempdir = FileHandle(is_dir=True)
        self.source_image.write_to_file(tempdir.path + "/source_image.fits")
        self.__save_sources_to_csv(tempdir.path + "/detected_sources.csv")
        shutil.make_archive(path, "zip", tempdir.path)

    @staticmethod
    def read_from_file(path) -> any:
        tempdir = FileHandle(is_dir=True)
        shutil.unpack_archive(path, tempdir.path)
        source_image = Image.read_from_file(tempdir.path + "/source_image.fits")
        source_catalouge = numpy.loadtxt(
            tempdir.path + "/detected_sources.csv", delimiter=","
        )
        return SourceDetectionResult(source_catalouge, source_image)

    def __save_sources_to_csv(self, filepath: str):
        """
        Save detected Sources to CSV
        :param filepath:
        :return:
        """
        numpy.savetxt(filepath, self.detected_sources, delimiter=",")

    def has_source_image(self) -> bool:
        """
        Check if source image is present.
        :return: True if present, False if not present
        """
        if self.source_image is not None:
            return True
        return False

    def get_source_image(self) -> Image:
        """
        Return the source image, where the source detection was performed on.
        :return: Karabo Image or None (if not supplied)
        """
        if self.has_source_image():
            return self.source_image

    def get_pixel_position_of_sources(self):
        x_pos = self.detected_sources[:, 3]
        y_pos = self.detected_sources[:, 4]
        result = np.vstack((np.array(x_pos), np.array(y_pos)))
        return result


class PyBDSFSourceDetectionResult(SourceDetectionResult):
    def __init__(self, bdsf_detection: bdsf_image):
        """
        Source Detection Result Wrapper for source detection results from PyBDSF.
        The Object allows the use of all Karabo-Source Detection functions on PyBDSF results
        :param bdsf_detection: PyBDSF result image
        """
        sources_file = FileHandle()
        bdsf_detection.write_catalog(
            outfile=sources_file.path, catalog_type="gaul", format="csv", clobber=True
        )
        bdsf_detected_sources = read_CSV_to_ndarray(sources_file.path)

        detected_sources = self.__transform_bdsf_to_reduced_result_array(
            bdsf_detected_sources
        )

        self.bdsf_detected_sources = bdsf_detected_sources
        self.bdsf_result = bdsf_detection
        source_image = self.__get_result_image("ch0")
        super().__init__(detected_sources, source_image)

    @staticmethod
    def __transform_bdsf_to_reduced_result_array(bdsf_detected_sources):
        sources = bdsf_detected_sources[:, [0, 4, 6, 12, 14, 8, 9]]
        return sources

    def __get_result_image(self, image_type: str) -> Image:
        image = Image()
        self.bdsf_result.export_image(
            outfile=image.file.path,
            img_format="fits",
            img_type=image_type,
            clobber=True,
        )
        return image

    def get_RMS_map_image(self) -> Image:
        return self.__get_result_image("rms")

    def get_mean_map_image(self) -> Image:
        return self.__get_result_image("mean")

    def get_polarized_intensity_image(self):
        return self.__get_result_image("pi")

    def get_gaussian_residual_image(self) -> Image:
        return self.__get_result_image("gaus_resid")

    def get_gaussian_model_image(self) -> Image:
        return self.__get_result_image("gaus_model")

    def get_shapelet_residual_image(self) -> Image:
        return self.__get_result_image("shap_resid")

    def get_shapelet_model_image(self) -> Image:
        return self.__get_result_image("shap_model")

    def get_major_axis_FWHM_variation_image(self) -> Image:
        return self.__get_result_image("psf_major")

    def get_minor_axis_FWHM_variation_image(self) -> Image:
        return self.__get_result_image("psf_minor")

    def get_position_angle_variation_image(self) -> Image:
        return self.__get_result_image("psf_pa")

    def get_peak_to_total_flux_variation_image(self) -> Image:
        return self.__get_result_image("psf_ratio")

    def get_peak_to_aperture_flux_variation_image(self) -> Image:
        return self.__get_result_image("psf_ratio_aper")

    def get_island_mask(self) -> Image:
        return self.__get_result_image("island_mask")
