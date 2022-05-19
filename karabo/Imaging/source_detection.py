import numpy as np
from bdsf import image as bdsf_image
from scipy.spatial.distance import cdist

from karabo.Imaging.image import Image
from karabo.simulation.sky_model import SkyModel
from karabo.util.FileHandle import FileHandle


class SourceDetectionResult:
    def __init__(self, detection: bdsf_image):
        self.detected_sources = []
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
                    self.detected_sources.append(n_row)

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


def map_sky_to_detection(sky: SkyModel, prediction: SourceDetectionResult, max_dist: float) -> np.ndarray:
    return automatic_assignment_of_ground_truth_and_prediction(sky.sources, np.array(prediction.detected_sources),
                                                               max_dist)


def automatic_assignment_of_ground_truth_and_prediction(ground_truth: np.ndarray, predicted: np.ndarray,
                                                        max_dist: float) -> np.ndarray:
    """
    Automatic assignment of the predicted sources `predicted` to the ground truth `gtruth`.
    The strategy is the following
    (similar to AUTOMATIC SOURCE DETECTION IN ASTRONOMICAL IMAGES, P.61, Marc MASIAS MOYSET, 2014):

    Each distance between the predicted and the ground truth sources is calculated.
    Any distances > `max_dist` are deleted.
    Assign the closest distance from the predicted and ground truth.
    Repeat the assignment, until every source from the gtruth has an assigment if possible,
        not allowing any double assignments from the predicted sources to the ground truth and vice versa.
    So each ground truth source should be assigned with a predicted source if at leas one was in range
        and the predicted source assigned to another ground truth source before.

    :param ground_truth: nx2 np.ndarray with the ground truth pixel coordinates of the catalog
    :param predicted: kx2 np.ndarray with the predicted pixel coordinates of the image
    :param max_dist: maximal allowed distance for assignment

    :return: jx3 np.ndarray where each row represents an assignment
                 - first column represents the ground truth index
                 - second column represents the predicted index
                 - third column represents the euclidean distance between the assignment
    """
    euclidian_distances = cdist(ground_truth, predicted)
    ground_truth_assignments = np.array([None] * ground_truth.shape[0])
    # gets the euclidian_distances sorted values indices as (m*n of euclidian_distances) x 2 matrix
    argsort_2dIndexes = np.array(
        np.unravel_index(np.argsort(euclidian_distances, axis=None), euclidian_distances.shape)).transpose()
    max_dist_2dIndexes = np.array(np.where(euclidian_distances <= max_dist)).transpose()
    # can slice it since argsort_2dIndexes is sorted. it is to ensure to not assign sources outside of max_dist
    argsort_2dIndexes = argsort_2dIndexes[:max_dist_2dIndexes.shape[0]]
    # to get the closes assignment it is the task to get the first indices pair which each index in each column
    # occured just once
    assigned_ground_truth_indexes, assigned_predicted_idxs, eucl_dist = [], [], []
    for i in range(argsort_2dIndexes.shape[0]):
        # could maybe perform better if possible assignments argsort_2dIndexes is very large by filtering the
        # selected idxs after assignment
        assignment_idxs = argsort_2dIndexes[i]
        if (assignment_idxs[0] not in assigned_ground_truth_indexes) and (
                assignment_idxs[1] not in assigned_predicted_idxs):
            assigned_ground_truth_indexes.append(assignment_idxs[0])
            assigned_predicted_idxs.append(assignment_idxs[1])
            eucl_dist.append(euclidian_distances[assignment_idxs[0], assignment_idxs[1]])
    assignments = np.array([assigned_ground_truth_indexes, assigned_predicted_idxs, eucl_dist]).transpose()
    return assignments


def calculate_evaluation_measures(ground_truth: np.ndarray, predicted: np.ndarray, max_dist: float) -> tuple:
    """
    Calculates the True Positive (TP), False Positive (FP) and False Negative (FN) of the ground truth and predictions.
    - TP are the detections associated with a source
    - FP are detections without any associated source
    - FN are sources with no associations with a detection

    :param ground_truth: nx2 np.ndarray with the ground truth pixel coordinates of the catalog
    :param predicted: kx2 np.ndarray with the predicted pixel coordinates of the image
    :param max_dist: maximal allowed distance for assignment

    :return: TP, FP, FN
    """
    assignments = automatic_assignment_of_ground_truth_and_prediction(ground_truth, predicted, max_dist)
    tp = assignments.shape[0]
    fp = predicted.shape[0] - assignments.shape[0]
    fn = ground_truth.shape[0] - assignments.shape[0]
    return tp, fp, fn
