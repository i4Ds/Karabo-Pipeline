import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from karabo.simulation.sky_model import SkyModel
from karabo.sourcedetection.result import SourceDetectionResult
from karabo.util.plotting_util import get_slices


class SourceDetectionEvaluation:
    def __init__(
        self,
        assignment: np.array,
        sky: SkyModel,
        source_detection: SourceDetectionResult,
        true_positives,
        false_negatives,
        false_positives,
    ):
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
    def evaluate_result_with_sky_in_pixel_space(
        source_detection_result: SourceDetectionResult,
        sky: SkyModel,
        distance_threshold: float,
        filter_outliers=False,
    ):
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

        truth = sky.project_sky_to_image(image, filter_outliers)[:2].astype("float64")
        pred = np.array(source_detection_result.get_pixel_position_of_sources()).astype(
            "float64"
        )
        assignment = SourceDetectionEvaluation.automatic_assignment_of_ground_truth_and_prediction(
            truth, pred, distance_threshold
        )
        tp, fp, fn = SourceDetectionEvaluation.calculate_evaluation_measures(
            assignment, truth, pred
        )
        result = SourceDetectionEvaluation(
            assignment, sky, source_detection_result, tp, fp, fn
        )
        return result

    @staticmethod
    def automatic_assignment_of_ground_truth_and_prediction(
        ground_truth: np.ndarray, detected: np.ndarray, max_dist: float
    ) -> np.ndarray:
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
            np.unravel_index(
                np.argsort(euclidian_distances, axis=None), euclidian_distances.shape
            )
        ).transpose()
        max_dist_2dIndexes = np.array(
            np.where(euclidian_distances <= max_dist)
        ).transpose()
        # can slice it since argsort_2dIndexes is sorted. it is to ensure to not assign sources outside of max_dist
        argsort_2dIndexes = argsort_2dIndexes[: max_dist_2dIndexes.shape[0]]
        # to get the closes assignment it is the task to get the first indices pair which each index in each column
        # occured just once
        assigned_ground_truth_indexes, assigned_predicted_idxs, eucl_dist = [], [], []
        for i in range(argsort_2dIndexes.shape[0]):
            # could maybe perform better if possible assignments argsort_2dIndexes is very large by filtering the
            # selected idxs after assignment
            assignment_idxs = argsort_2dIndexes[i]
            if (assignment_idxs[0] not in assigned_ground_truth_indexes) and (
                assignment_idxs[1] not in assigned_predicted_idxs
            ):
                assigned_ground_truth_indexes.append(assignment_idxs[0])
                assigned_predicted_idxs.append(assignment_idxs[1])
                eucl_dist.append(
                    euclidian_distances[assignment_idxs[0], assignment_idxs[1]]
                )
        assignments = np.array(
            [assigned_ground_truth_indexes, assigned_predicted_idxs, eucl_dist]
        ).transpose()
        return assignments

    @staticmethod
    def calculate_evaluation_measures(
        assignments: np.ndarray, ground_truth: np.ndarray, detected: np.ndarray
    ) -> tuple:
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

    def plot(self, filename=None):
        """
        Plot the found sources as green x's and the source truth as red 'o' on the original image,
         that the source detection was performed on.
        """

        if self.source_detection.has_source_image():
            image = self.source_detection.get_source_image()
            wcs = WCS(image.header)

            slices = get_slices(wcs)

            fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=wcs, slices=slices))
            ax.imshow(image.data[0][0], cmap="jet", origin="lower", interpolation=None)

            self.__plot_truth_and_prediction(ax)

            if filename:
                plt.savefig(filename)
                plt.show(block=False)
            else:
                plt.show()
        else:
            fig, ax = plt.subplots(1, 1, subplot_kw=dict())

            self.__plot_truth_and_prediction(ax)
            if filename:
                plt.savefig(filename)
                plt.show(block=False)
            else:
                plt.show()

    def __plot_truth_and_prediction(self, ax):
        truth = self.get_truth_array()[:, [3, 4]].transpose()
        pred = self.get_detected_array()[:, [3, 4]].transpose()
        ax.plot(truth[0], truth[1], "o", linewidth=5, color="firebrick", alpha=0.5)
        ax.plot(pred[0], pred[1], "x", linewidth=5, color="green")

    def __map_sky_to_detection_array(self, assignment, sky: SkyModel) -> np.ndarray:
        truth_indexes = np.array(assignment[:, 0], dtype=int)
        pred_indexes = np.array(assignment[:, 1], dtype=int)
        meta = assignment[:, 2]

        predictions = self.source_detection.detected_sources[pred_indexes]
        truths = self.__sky_array_to_same_shape_as_detection(truth_indexes, sky)
        meta = np.vstack((meta, np.zeros((6, meta.shape[0])))).transpose()

        result = np.stack((truths, predictions, meta))
        return result

    def __sky_array_to_same_shape_as_detection(
        self, sky_indexes: npt.NDArray, sky: SkyModel
    ) -> npt.NDArray:
        pixel_coords_sky = sky.project_sky_to_image(
            self.source_detection.get_source_image(), filter_outlier=False
        )
        pixel_coords = pixel_coords_sky[:, sky_indexes].transpose()
        filtered = sky[sky_indexes.astype(dtype="uint32")]
        ra = filtered[:, 0]
        dec = filtered[:, 1]
        flux = filtered[:, 2]
        x_pos = pixel_coords[:, 0]
        y_pos = pixel_coords[:, 1]
        peak = np.zeros((len(filtered)))
        indexes = sky_indexes.transpose()
        return np.vstack((indexes, ra, dec, x_pos, y_pos, flux, peak)).transpose()

    def get_confusion_matrix(self) -> npt.NDArray:
        return np.array(
            [[self.true_positives, self.false_negatives], [self.false_positives, 0.0]]
        )

    def get_accuracy(self) -> float:
        return self.true_positives / (
            self.true_positives + self.false_positives + self.false_negatives
        )

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

    def plot_error_ra_dec(self, filename=None):
        truth = self.get_truth_array().astype(float)
        detection = self.get_detected_array().astype(float)

        # get ra-dec error
        ra_dec_truth = truth[:, [1, 2]].transpose()
        ra_dec_det = detection[:, [1, 2]].transpose()
        error = ra_dec_truth - ra_dec_det
        ra_error = error[0]
        dec_error = error[1]

        # get pixel error
        pixel_truth = truth[:, [3, 4]].transpose()
        pixel_detection = detection[:, [3, 4]].transpose()
        error_pixel = pixel_truth - pixel_detection
        pixel_x_error = error_pixel[0]
        pixel_y_error = error_pixel[1]

        ra_div_pix_error = ra_error / pixel_x_error
        dec_div_pix_error = ra_error * np.cos(dec_error) / pixel_y_error

        error_max_ra = max(np.max(ra_div_pix_error), np.max(dec_div_pix_error))
        error_max_dec = max(np.max(ra_div_pix_error), np.max(dec_div_pix_error))

        plt.xlim([-error_max_ra, error_max_ra])
        plt.ylim([-error_max_dec, error_max_ra])
        plt.xlabel("RA (deg) error / x")
        plt.ylabel("DEC (deg) error / y")
        plt.plot(error[0], error[1], "o", markersize=8, color="r", alpha=0.5)
        if filename:
            plt.savefig(filename)
            plt.show(block=False)
        else:
            plt.show()

    def plot_quiver_positions(self, filename=None):
        ref = self.get_truth_array()[:, [1, 2]].transpose().astype(float)
        pred = self.get_detected_array()[:, [1, 2]].transpose().astype(float)
        ra_ref = np.array(ref[0], dtype=float)
        dec_ref = np.array(ref[1], dtype=float)
        num = len(ra_ref)

        error = ref - pred
        ra_error = error[0] * (np.cos(np.deg2rad(dec_ref)))
        dec_error = error[1]
        fig, ax = plt.subplots()
        if np.mean(np.deg2rad(dec_ref)) != 0.0:
            ax.set_aspect(1.0 / np.cos(np.mean(np.deg2rad(dec_ref))))
        q = ax.quiver(ra_ref, dec_ref, ra_error, dec_error, color="b")

        ax.scatter(ra_ref, dec_ref, color="r", s=8)
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        plt.title(f"Matched {num} sources")
        if filename:
            plt.savefig(filename)
            plt.show(block=False)
        else:
            plt.show()

    def plot_flux_ratio_to_distance(self, filename=None):
        ref = self.get_truth_array()[:, [1, 2, 5]].transpose().astype(float)
        pred = self.get_detected_array()[:, [1, 2, 5]].transpose().astype(float)
        # ra_dec_ref = ref[0, 1]
        ra_dec_pred = ref[[0, 1]]
        ra_ref = ref[0]
        dec_ref = ref[1]
        flux_ref = ref[2]
        ra_pred = pred[0]
        dec_pred = pred[1]
        flux_pred = pred[2]
        phase_center = self.source_detection.get_source_image().get_phase_centre()

        flux_ratio = flux_pred / flux_ref

        sky_coords_pred = [
            SkyCoord(ra=p[0], dec=p[1], frame="icrs", unit="deg")
            for p in ra_dec_pred.transpose()
        ]
        sky_coord_center = SkyCoord(
            phase_center[0] * u.degree, phase_center[1] * u.degree, frame="icrs"
        )
        dist = [coord.separation(sky_coord_center).degree for coord in sky_coords_pred]

        plt.plot(dist, flux_ratio, "o", color="b", markersize=5, alpha=0.5)
        plt.title("Flux ratio vs. distance")
        plt.xlabel("Distance to center (Deg)")
        plt.ylabel("Flux Ratio (Out/In)")
        if filename:
            plt.savefig(filename)
            plt.show(block=False)
        else:
            plt.show()

    def plot_flux_ratio_to_ra_dec(self, filename=None):
        ref = self.get_truth_array()[:, [1, 2, 5]].transpose().astype(float)
        pred = self.get_detected_array()[:, [1, 2, 5]].transpose().astype(float)
        ra_pred = pred[0]
        dec_pred = pred[1]
        flux_ref = ref[2]
        flux_pred = pred[2]

        flux_ratio = flux_pred / flux_ref

        # Flux ratio vs. RA & Dec
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        fig.suptitle("Flux ratio vs. Position")
        ax1.plot(ra_pred, flux_ratio, "o", color="b", markersize=5, alpha=0.5)
        ax2.plot(dec_pred, flux_ratio, "o", color="b", markersize=5, alpha=0.5)

        ax1.set_xlabel("RA (deg)")
        ax2.set_xlabel("Dec (deg)")
        ax1.set_ylabel("Flux ratio (Out/In)")
        if filename:
            plt.savefig(filename)
            plt.show(block=False)
        else:
            plt.show()

    def plot_flux_histogram(self, nbins=10, filename=None):

        flux_in = self.get_truth_array()[:, [5]].transpose().astype(float)
        flux_out = self.get_detected_array()[:, [5]].transpose().astype(float)

        flux_in = flux_in[flux_in > 0.0]
        flux_out = flux_out[flux_out > 0.0]

        hist = [flux_in, flux_out]
        labels = ["Flux In", "Flux Out"]
        colors = ["r", "b"]
        hist_min = min(np.min(flux_in), np.min(flux_out))
        hist_max = max(np.max(flux_in), np.max(flux_out))

        hist_bins = np.logspace(np.log10(hist_min), np.log10(hist_max), nbins)

        fig, ax = plt.subplots()
        ax.hist(hist, bins=hist_bins, log=True, color=colors, label=labels)

        ax.set_title("Flux histogram")
        ax.set_xlabel("Flux (Jy)")
        ax.set_xscale("log")
        ax.set_ylabel("Source Count")
        plt.legend(loc="best")
        if filename:
            plt.savefig(filename)
            plt.show(block=False)
        else:
            plt.show()
