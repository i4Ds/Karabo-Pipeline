import numpy as np
import numpy.typing as npt
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from karabo.simulation.sky_model import SkyModel
from karabo.sourcedetection.result import SourceDetectionResult
from karabo.util.plotting_util import get_slices


class SourceDetectionEvaluation:

    def __init__(self, assignment: np.array,
                 sky: SkyModel,
                 source_detection: SourceDetectionResult,
                 true_positives,
                 false_negatives,
                 false_positives):
        """
        Class that holds the mapping of a source detection to truth mapping.
        :param assignment: jx3 np.ndarray where each row represents an assignment
                 - first column represents the ground truth index
                 - second column represents the predicted index
                 - third column represents the euclidean distance between the assignment
        :param sky: sky model that is the ground truth
        :param source_detection: Source Detection Result from a previous source detection.
        """
        # self.assignment = assignment
        # self.pixel_coordinates_sky = pixel_coordinates_sky
        # self.sky = sky
        # self.pixel_coordinates_detection = pixel_coordinates_detection
        self.source_detection = source_detection
        self.true_positives = true_positives
        self.false_negatives = false_negatives
        self.false_positives = false_positives

        self.mapped_array = self.__map_sky_to_detection_array(assignment, sky)

    @staticmethod
    def evaluate_result_with_sky_in_pixel_space(source_detection_result: SourceDetectionResult,
                                                sky: SkyModel,
                                                distance_threshold: float,
                                                filter_outliers=False):
        """
        Evaluate Result of this Source Detection Result by comparing it with a Sky Model.
        The Sky Model will be converted to Pixel space based on the dimensions of the original Image.
        Then the pixel space of the found sources and sky model sources will be compared and paired automatically.
        But only if so possible based on the

        The mapping uses the automatic_assignment_of_ground_truth_and_prediction() function
        and calculate_evaluation_measures() to create the evaluation

        :param filter_outliers: If True outliers will be filtered , defaults to False
        :param source_detection_result: result that was produced with a source detection algorithm
        :param sky: sky that was used to create the image
        :param source_image_cell_size: cellsize in the original source image (used for mapping),
                                       cannot be read out from the fits file (unfortunately)
        :param distance_threshold: threshold of distance between two sources,
                                   so that they are still considered in mathching (pixel distance).
        :return:
        """
        image = source_detection_result.get_source_image()

        truth = sky.project_sky_to_image(image, filter_outliers)[
                :2].astype(
            'float64')
        pred = np.array(source_detection_result.get_pixel_position_of_sources()).astype('float64')
        assignment = SourceDetectionEvaluation. \
            automatic_assignment_of_ground_truth_and_prediction(truth, pred, distance_threshold)
        tp, fp, fn = SourceDetectionEvaluation.calculate_evaluation_measures(assignment, truth, pred)
        result = SourceDetectionEvaluation(assignment, sky, source_detection_result, tp, fp, fn)
        return result

    @staticmethod
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
        :param max_dist: maximal allowed distance for assignment (in pixel)

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

    @staticmethod
    def calculate_evaluation_measures(assignments: np.ndarray, ground_truth: np.ndarray,
                                      detected: np.ndarray) -> tuple:
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

    def plot(self):
        """
        Plot the found sources as green x's and the source truth as red 'o' on the original image,
         that the source detection was performed on.
        """

        if self.source_detection.has_source_image():
            image = self.source_detection.get_source_image()
            wcs = WCS(image.header)

            slices = get_slices(wcs)

            fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=wcs, slices=slices))
            ax.imshow(image.data[0][0], cmap="jet", origin='lower', interpolation=None)

            self.__plot_truth_and_prediction(ax)

            plt.show()
        else:
            fig, ax = plt.subplots(1, 1, subplot_kw=dict())

            self.__plot_truth_and_prediction(ax)
            plt.show()

    def __plot_truth_and_prediction(self, ax):
        truth = self.get_truth_array()[:, [3, 4]].transpose()
        pred = self.get_detected_array()[:, [3, 4]].transpose()
        ax.plot(truth[0], truth[1], 'o', linewidth=5,
                color="firebrick", alpha=0.5)
        ax.plot(pred[0], pred[1], 'x', linewidth=5, color='green')

    def __map_sky_to_detection_array(self, assignment, sky: SkyModel) -> np.ndarray:
        truth_indexes = np.array(assignment[:, 0], dtype=int)
        pred_indexes = np.array(assignment[:, 1], dtype=int)
        meta = assignment[:, 2]

        predictions = self.source_detection.detected_sources[pred_indexes]
        truths = self.__sky_array_to_same_shape_as_detection(truth_indexes, sky)
        meta = np.vstack((meta, np.zeros((6, meta.shape[0])))).transpose()

        result = np.stack((truths, predictions, meta))
        return result

    def __sky_array_to_same_shape_as_detection(self,
                                               sky_indexes: npt.NDArray,
                                               sky: SkyModel) -> npt.NDArray:
        pixel_coords_sky = sky.project_sky_to_image(self.source_detection.get_source_image())
        pixel_coords = pixel_coords_sky[:, sky_indexes].transpose()
        filtered = sky[sky_indexes.astype(dtype='uint32')]
        ra = filtered[:, 0]
        dec = filtered[:, 1]
        flux = filtered[:, 2]
        x_pos = pixel_coords[:, 0]
        y_pos = pixel_coords[:, 1]
        peak = np.zeros((len(filtered)))
        indexes = sky_indexes.transpose()
        return np.vstack((indexes, ra, dec, x_pos, y_pos, flux, peak)).transpose()

    def get_confusion_matrix(self) -> npt.NDArray:
        return np.array([[self.true_positives, self.false_negatives],
                         [self.false_positives, 0.0]])

    def get_accuracy(self) -> float:
        return self.true_positives / (self.true_positives + self.false_positives + self.false_negatives)

    def get_precision(self) -> float:
        return self.true_positives / (self.true_positives + self.false_positives)

    def get_sensitivity(self) -> float:
        return self.true_positives / (self.true_positives + self.false_negatives)

    def get_f_score(self) -> float:
        p = self.get_precision()
        sn = self.get_sensitivity()
        return 2 * (p * sn / (p + sn))

    def get_truth_array(self) -> npt.NDArray:
        return self.mapped_array[0]

    def get_detected_array(self) -> npt.NDArray:
        return self.mapped_array[1]

    def get_meta_data_array(self) -> npt.NDArray:
        return self.mapped_array[2]

    def plot_error_ra_dec(self):
        ra_dec_truth = self.get_truth_array()[:, [1, 2]].transpose()
        ra_dec_det = self.get_detected_array()[:, [1, 2]].transpose()
        error = ra_dec_truth - ra_dec_det
        ra_error = error[0]
        dec_error = error[1]

        err_r = max(np.max(ra_error), np.max(dec_error))
        err_l = min(np.min(ra_error), np.min(dec_error))
        err = max(err_l, err_r)
        err *= 1.1 #scale to add a small border
        plt.xlim([-err, err])
        plt.ylim([-err, err])
        plt.xlabel("RA (deg) error")
        plt.ylabel("DEC (deg) error")
        plt.plot(error[0], error[1], 'o', markersize=8, color='r', alpha=0.5)
        plt.show()

    def quiver_plot_error_ra_dec(self):
        ra_dec_truth = self.get_truth_array()[:, [1, 2]].transpose()
        ra_dec_det = self.get_detected_array()[:, [1, 2]].transpose()
        error = ra_dec_truth - ra_dec_det
        ra_error = error[0]
        dec_error = error[1]

        plt.quiver(ra_dec_truth[0], ra_dec_truth[1], ra_error, dec_error, color='b')
        plt.scatter(ra_dec_truth[0], ra_dec_truth[1], color='r', s=8)
        plt.show()



# class SourceDetectionEvaluationBlock:
#
#     def __init__(self, evaluations: [SourceDetectionEvaluation]):
#         if len(evaluations) != 0:
#             self.evaluations: [SourceDetectionEvaluation] = evaluations
#         else:
#             self.evaluations: [SourceDetectionEvaluation] = []
#
#     def add_evaluation(self, evaluation: SourceDetectionEvaluation):
#         self.evaluations.append(evaluation)
#
#     def flatten_plot(self, index_sky, index_result, index_distance):
#
#         sky = np.concatenate([t.mapped_array[0] for t in self.evaluations])
#         result = np.concatenate([t.mapped_array[1] for t in self.evaluations])
#         distance = np.concatenate([t.mapped_array[2] for t in self.evaluations])
#
#         if not (len(sky) > index_sky >= 0):
#             raise IndexError("Sky Index is not in range")
#
#         if not (len(result) > index_result >= 0):
#             raise IndexError("")
#
#         if not (len(distance) > index_distance >= 0):
#             raise IndexError("")
#
#         x = range(0, len(distance))
#
#         sky_select = sky[:, index_sky]
#
#         result_select = result[:, index_result]
#
#         distance_select = result[:, index_distance]
#
#         fig, ax = plt.subplots()
#         ax.plot(x, sky_select)
#         ax.plot(x, result_select)
#         fig.savefig("./result_full.png")
