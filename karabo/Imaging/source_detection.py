from bdsf import image as bdsfImage

from karabo.Imaging.image import Image
from karabo.util.FileHandle import FileHandle


class SourceDetectionResult:
    def __init__(self, detection: bdsfImage):
        self.sources = []
        self.detection = detection
        self.sources_file = FileHandle()
        detection.write_catalog(outfile=self.sources_file.path, catalog_type="srl", format="csv", clobber=True)
        self.__read_CSV_sources()

    def __read_CSV_sources(self):
        import csv
        with open(self.sources_file.path, newline='') as sourcefile:
            spamreader = csv.reader(sourcefile, delimiter=',', quotechar='|')
            for row in spamreader:
                if len(row) == 0:
                    continue
                if row[0].startswith("#"):
                    continue
                else:
                    print(row)
                    n_row = []
                    for cell in row:

                        try:
                            value = float(cell)
                        except:
                            value = cell
                        n_row.append(value)
                    self.sources.append(n_row)

    def __get_result_image(self, image_type: str) -> Image:
        image = Image()
        self.detection.export_image(outfile=image.file.path, img_format='fits', img_type=image_type, clobber=True)
        return image

    def get_source_image(self) -> Image:
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


def detect_sources_in_image(image: Image, beam=None) -> SourceDetectionResult:
    """
    Detecting sources in an image. The Source detection is impemented with the PyBDSF.process_image function.
    See https://www.astron.nl/citt/pybdsf/process_image.html for more information.

    :param image: Image to perform source detection on
    :param beam: FWHM of restoring beam. Specify as (maj, min. pos angle E of N). None means it will try to be extracted from the Image data.
    :return: Source Detection Result containing the found sources
    """
    import bdsf
    detection = bdsf.process_image(image.file.path, beam=beam)
    return SourceDetectionResult(detection)

#
# def use_dao_star_finder(image: Image):
#     from photutils.detection import DAOStarFinder
#     from photutils.datasets import load_star_image
#     data = image.get_data()
#     from astropy.stats import sigma_clipped_stats
#     mean, median, std = sigma_clipped_stats(data)
#     finder = DAOStarFinder(0.1 * std, 3.0)
#     sources = finder(data)
#     print(sources)
#
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from astropy.visualization import SqrtStretch
#     from astropy.visualization.mpl_normalize import ImageNormalize
#     from photutils.aperture import CircularAperture
#     positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
#     apertures = CircularAperture(positions, r=4.)
#     norm = ImageNormalize(stretch=SqrtStretch())
#     plt.imshow(data, cmap='Greys', origin='lower', norm=norm,
#                interpolation='nearest')
#     apertures.plot(color='blue', lw=1.5, alpha=0.5)
#     plt.show()
#
