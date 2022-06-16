import shutil

import numpy as np

from karabo.Imaging.image import Image
from bdsf import image as bdsf_image

from karabo.util.FileHandle import FileHandle


class SourceDetectionResult:
    def __init__(self, detection: bdsf_image = None, file_path_csv: str = None, source_image: Image = None):
        if detection is not None:
            """Create Source Detection Result from bdsf_image output."""
            self.detected_sources = np.array([])
            self.detection = detection
            self.sources_file = FileHandle()
            detection.write_catalog(outfile=self.sources_file.path, catalog_type="gaul", format="csv", clobber=True)
            self.__read_CSV_sources(self.sources_file.path)
            # Explicitly pass the source image
            if source_image is not None:
                self.source_image = source_image
        elif file_path_csv is not None:
            """Create a SourceDetectionResult object from a CSV list. 
               If SourceDetectionResult is created like this the get_image_<any> functions cannot be used.
               Rerun the detection to look at the images.
               However the map_sky_to_detection() can be used, with this source detection result."""
            self.sources_file = FileHandle(existing_file_path=file_path_csv)
            self.detected_sources = np.array([])
            self.__read_CSV_sources(file_path_csv)
            self.detection = None
            self.source_image = source_image

    def save_sources_file_as_csv(self, filepath: str):
        if not filepath.endswith(".csv"):
            raise EnvironmentError("The passed path and name of file must end with .csv")

        shutil.copy(self.sources_file.path, filepath)

    def __read_CSV_sources(self, file: str):
        import csv
        sources = []
        with open(file, newline='') as sourcefile:
            spamreader = csv.reader(sourcefile, delimiter=',', quotechar='|')
            for row in spamreader:
                if len(row) == 0:
                    continue
                if row[0].startswith("#"):
                    continue
                else:
                    n_row = []
                    for cell in row:
                        try:
                            value = float(cell)
                        except:
                            value = cell
                        n_row.append(value)
                    sources.append(n_row)
        self.detected_sources = np.array(sources)

    def has_source_image(self) -> bool:
        if self.detection is not None or self.source_image is not None:
            return True
        return False

    def __get_result_image(self, image_type: str) -> Image:
        image = Image()
        self.detection.export_image(outfile=image.file.path, img_format='fits', img_type=image_type, clobber=True)
        return image

    def get_source_image(self) -> Image:
        if self.source_image is not None:
            return self.source_image
        return self.__get_result_image('cho0')

    def get_RMS_map_image(self) -> Image:
        return self.__get_result_image('rms')

    def get_mean_map_image(self) -> Image:
        return self.__get_result_image('mean')

    def get_polarized_intensity_image(self):
        return self.__get_result_image('pi')

    def get_gaussian_residual_image(self) -> Image:
        return self.__get_result_image('gaus_resid')

    def get_gaussian_model_image(self) -> Image:
        return self.__get_result_image('gaus_model')

    def get_shapelet_residual_image(self) -> Image:
        return self.__get_result_image('shap_resid')

    def get_shapelet_model_image(self) -> Image:
        return self.__get_result_image('shap_model')

    def get_major_axis_FWHM_variation_image(self) -> Image:
        return self.__get_result_image('psf_major')

    def get_minor_axis_FWHM_variation_image(self) -> Image:
        return self.__get_result_image('psf_minor')

    def get_position_angle_variation_image(self) -> Image:
        return self.__get_result_image('psf_pa')

    def get_peak_to_total_flux_variation_image(self) -> Image:
        return self.__get_result_image('psf_ratio')

    def get_peak_to_aperture_flux_variation_image(self) -> Image:
        return self.__get_result_image('psf_ratio_aper')

    def get_island_mask(self) -> Image:
        return self.__get_result_image('island_mask')

    def get_pixel_position_of_sources(self):
        x_pos = self.detected_sources[:, 12]
        y_pos = self.detected_sources[:, 14]
        result = np.vstack((np.array(x_pos), np.array(y_pos)))
        return result

