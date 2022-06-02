import shutil

import numpy
import numpy as np
from astropy.wcs import WCS
from bdsf import image as bdsf_image
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from karabo.Imaging.image import Image, open_fits_image
from karabo.simulation.sky_model import SkyModel
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
            self.detected_sources = np.array([])
            self.__read_CSV_sources(file_path_csv)
            self.detection = None
            self.source_image = source_image

    def save_sources_file_as_csv(self, filepath: str):
        if not filepath.endswith(".csv"):
            raise EnvironmentError("The passed path and name of file must end with .fits")

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
        if self.detection is None and self.source_image is not None:
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


def read_detection_from_sources_file_csv(filepath: str, source_image_path: str = None) -> SourceDetectionResult:
    """
    Reads in a CSV table and saves it in the Source Detection Result.
    The format of the CSV is according to the PyBDSF definition.:
    https://www.astron.nl/citt/pybdsf/write_catalog.html#definition-of-output-columns

    Karabo creates the output from write_catalog(format='csv', catalogue_type='gaul').
    We suggest to only read in CSV that are created with Karabo (or with PyBDSF itself with the above configuration).

    This method is mainly for convenience.
    It allows that one can save the CSV with the SourceDetectionResult.save_sources_as_csv_file("./sources.csv")
    and then read it back in.
    This helps save runtime and potential wait time, when working with the output of the source detection

    :param source_image_path: (Optional), you can also read in the source image for the detection.
            If you read this back in you can use plot() function on the SkyModelToSourceDetectionMapping
    :param filepath: file of CSV sources in the format that
    :return: SourceDetectionResult
    """
    image = None
    if source_image_path is not None:
        image = open_fits_image(source_image_path)
    detection = SourceDetectionResult(file_path_csv=filepath, source_image=image)
    return detection


def detect_sources_in_image(image: Image, beam=None) -> SourceDetectionResult:
    """
    Detecting sources in an image. The Source detection is impemented with the PyBDSF.process_image function.
    See https://www.astron.nl/citt/pybdsf/process_image.html for more information.

    :param image: Image to perform source detection on
    :param beam: FWHM of restoring beam. Specify as (maj, min. pos angle E of N).
                 None means it will try to be extracted from the Image data. (Might fail)
    :return: Source Detection Result containing the found sources
    """
    import bdsf
    detection = bdsf.process_image(image.file.path, beam=beam, quiet=True, format='csv')
    return SourceDetectionResult(detection, source_image=image)


def map_sky_to_detection(sky: SkyModel,
                         sky_projection_cell_size: float,
                         sky_projection_pixel_per_side: float,
                         prediction: SourceDetectionResult,
                         max_dist: float):
    truth = sky.project_sky_to_2d_image(sky_projection_cell_size, sky_projection_pixel_per_side)[:2].astype('float64')
    pred = np.array(prediction.get_pixel_position_of_sources()).astype('float64')
    assignment = automatic_assignment_of_ground_truth_and_prediction(truth, pred, max_dist)
    tp, fp, fn = calculate_evaluation_measures(assignment, truth, pred)
    result = SourceDetectionEvaluation(assignment, truth, sky, pred, prediction, tp, fp, fn)
    return result


def automatic_assignment_of_ground_truth_and_prediction(ground_truth: np.ndarray, detected: np.ndarray,
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
    :param detected: kx2 np.ndarray with the predicted pixel coordinates of the image
    :param max_dist: maximal allowed distance for assignment

    :return: jx3 np.ndarray where each row represents an assignment
                 - first column represents the ground truth index
                 - second column represents the predicted index
                 - third column represents the euclidean distance between the assignment
    """
    ground_truth = ground_truth.transpose()
    detected = detected.transpose()
    euclidian_distances = cdist(ground_truth, detected)
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


def calculate_evaluation_measures(assignments: np.ndarray, ground_truth: np.ndarray, detected: np.ndarray) -> tuple:
    """
    Calculates the True Positive (TP), False Positive (FP) and False Negative (FN) of the ground truth and predictions.
    - TP are the detections associated with a source
    - FP are detections without any associated source
    - FN are sources with no associations with a detection

    :param assignments:
    :param ground_truth: nx2 np.ndarray with the ground truth pixel coordinates of the catalog
    :param detected: kx2 np.ndarray with the predicted pixel coordinates of the image
    :param max_dist: maximal allowed distance for assignment

    :return: TP, FP, FN
    """
    tp = assignments.shape[0]
    fp = detected.shape[0] - assignments.shape[0]
    fn = ground_truth.shape[0] - assignments.shape[0]
    return tp, fp, fn


class SourceDetectionEvaluation:

    def __init__(self, assignment: np.array, pixel_coordinates_sky: np.array, sky: SkyModel,
                 pixel_coordinates_detection: np.array, source_detection: SourceDetectionResult,
                 true_positives, false_negatives, false_positives):
        """
        Class that holds the mapping of a source detection to truth mapping.
        :param assignment: jx3 np.ndarray where each row represents an assignment
                 - first column represents the ground truth index
                 - second column represents the predicted index
                 - third column represents the euclidean distance between the assignment
        :param pixel_coordinates_sky: array that holds the pixel coordinates of the ground truth sources
        :param sky: sky model that is the ground truth
        :param pixel_coordinates_detection: array that holds the pixel coordinates of the detected sources
        :param source_detection: Source Detection Result from a previous source detection.
        """
        self.assignment = assignment
        self.pixel_coordinates_sky = pixel_coordinates_sky
        self.sky = sky
        self.pixel_coordinates_detection = pixel_coordinates_detection
        self.source_detection = source_detection
        self.true_positives = true_positives
        self.false_negatives = false_negatives
        self.false_positives = false_positives

    def plot(self):
        """
        Plot the found sources as green x's and the source truth as red 'o' on the original image,
         that the source detection was performed on.
        """

        if self.source_detection.has_source_image():
            image = self.source_detection.get_source_image()
            wcs = WCS(image.header)
            fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=wcs, slices=('y', 'x')))
            squeezed = numpy.squeeze(image.data[:1, :1, :, :])  # remove any (1) size dimensions
            ax.imshow(squeezed, cmap="jet", origin='lower', extent=[0, 2000, 0, 2000])

            self.__plot_truth_and_prediction(ax)

            plt.show()
        else:
            fig, ax = plt.subplots(1, 1, subplot_kw=dict())

            self.__plot_truth_and_prediction(ax)
            plt.show()

    def __plot_truth_and_prediction(self, ax):
        truth_and_pred_coords = self.get_truth_to_detection_pixel_coordinate_array().transpose()
        ax.plot(truth_and_pred_coords[0, :], truth_and_pred_coords[1, :], 'o', linewidth=5, color='firebrick')
        ax.plot(truth_and_pred_coords[2, :], truth_and_pred_coords[3, :], 'x', linewidth=5, color='green')

    def get_truth_to_detection_pixel_coordinate_array(self) -> np.ndarray:
        """
        Get a np.ndarray holding the pixel coordinates of the truth and the detection mapped.
        Can be used for further analysis.
        :return: nx4 np.ndarray
            - 1. column x direction pixel of truth
            - 2. column y direction pixel of truth
            - 3. column x direction pixel of detection
            - 4. column y direction pixel of detection
        """
        truth_indexes = np.array(self.assignment[:, 0], dtype=int)
        truths = self.pixel_coordinates_sky[:, truth_indexes]
        pred_indexes = np.array(self.assignment[:, 0], dtype=int)
        preds = self.pixel_coordinates_sky[:, pred_indexes]
        return np.vstack((truths, preds)).transpose()
