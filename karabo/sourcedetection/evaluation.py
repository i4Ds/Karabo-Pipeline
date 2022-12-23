from __future__ import annotations
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from karabo.simulation.sky_model import SkyModel
from karabo.sourcedetection.result import SourceDetectionResult
from karabo.util.plotting_util import get_slices
from typing import Union, Tuple, Optional
from numpy.typing import NDArray


class SourceDetectionEvaluation:
    def __init__(
        self,
        sky: SkyModel,
        ground_truth: Tuple[NDArray[np.float64], NDArray[np.int64]],
        assignments: NDArray[np.float64],
        sky_idxs: NDArray[np.int64],
        source_detection: SourceDetectionResult,
    ) -> None:
        """
        Class that holds the mapping of a source detection to truth mapping.
        :param sky: `SkyModel` where the `assignment` comes from
        :param ground_truth: 2xn array of pixel positions of ground truth
        :param assignments: jx3 np.ndarray where each row represents an assignment
            - first column represents the `ground_truth` index
            - second column represents the predicted `source_detection.detected_sources` index
            - third column represents the euclidean distance between the assignment
        :param sky_idxs: Sky sources indices of `SkyModel` from `assignment`
        :param source_detection: Source Detection Result from a previous source detection
        """
        self.sky = sky
        self.ground_truth = ground_truth
        self.assignments = assignments
        self.sky_idxs = sky_idxs
        self.source_detection = source_detection

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
            truth.T, pred.T, distance_threshold
        )
        tp, fp, fn = SourceDetectionEvaluation.calculate_evaluation_measures(assignment)
        result = SourceDetectionEvaluation(
            assignment, sky, source_detection_result, tp, fp, fn
        )
        return result

    @staticmethod
    def __return_multiple_assigned_detected_points(
        assigments: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the indices of the predicted sources that are assigned to more than one ground truth source.
        """
        # Check if a ground truth point is assigned to more than one predicted point
        unique_counts = np.unique(assigments[:, 0], return_counts=True)  # O(nlogn)
        pred_multiple_assignment = unique_counts[0][unique_counts[1] > 1]
        # Don't check unassigned points (When no points are below the max distance by kdtree, they are assigned to input.shape, which we replace to -1).
        pred_multiple_assignment = pred_multiple_assignment[
            pred_multiple_assignment != -1
        ]
        return pred_multiple_assignment

    @staticmethod
    def automatic_assignment_of_ground_truth_and_prediction(
        ground_truth: Union[NDArray[np.int64], NDArray[np.float64]],
        detected: Union[NDArray[np.int64], NDArray[np.float64]],
        max_dist: float,
        top_k: int = 3,
    ) -> NDArray[np.float64]:
        """
        Automatic assignment of the predicted sources `predicted` to the ground truth `gtruth`.
        The strategy is the following:
        (similar to AUTOMATIC SOURCE DETECTION IN ASTRONOMICAL IMAGES, P.61, Marc MASIAS MOYSET, 2014):

        Each distance between the predicted and the ground truth sources is calculated.
        Any distances > `max_dist` are not considered.
        Assign the closest distance from the predicted and ground truth.
        Repeat the assignment, until every source from the gtruth has an assigment if possible,
            not allowing any double assignments from the predicted sources to the ground truth and vice versa.
        So each ground truth source should be assigned with a predicted source if at least one was in range
            and the predicted source assigned to another ground truth source before.

        :param ground_truth: nx2 np.ndarray with the ground truth pixel coordinates of the catalog
        :param detected: kx2 np.ndarray with the predicted pixel coordinates of the image
        :param max_dist: maximal allowed euclidean distance for assignment (in pixel domain)
        :param top_k: number of top predictions to be considered in scipy.spatial.KDTree. A small value could lead to inperfect results.

        :return: nx3 np.ndarray where each row represents an assignment
            - first column represents the ground truth index (return is sorted by this column)
                a minus index means a ground-truth source with no allocated prediction
            - second column represents the predicted index
                a minus index means a predicted source with no allocated ground-truth
            - third column represents the euclidean distance between the assignment
                a "inf" means no allocation between ground-truth and prediction of that source
        """
        # With scipy.spatial.KDTree get the closest detection point for each ground truth point
        tree = KDTree(ground_truth)
        distance, idx_assigment_pred = tree.query(
            detected, k=top_k, distance_upper_bound=max_dist
        )
        # Replace unassigned points with -1
        idx_assigment_pred[distance == np.inf] = -1
        # Check if a ground truth point is assigned to more than one predicted point
        pred_multiple_assignments = (
            SourceDetectionEvaluation.__return_multiple_assigned_detected_points(
                idx_assigment_pred
            )
        )
        while len(pred_multiple_assignments) > 0:
            for pred_multiple_assignment in pred_multiple_assignments:
                # Get idx
                idx_pred_multiple_assigment = np.where(
                    idx_assigment_pred[:, 0] == pred_multiple_assignment
                )
                idx_max_distance_multiple_assigment = np.argmax(
                    distance[idx_pred_multiple_assigment, 0]
                )
                idx_max_distance_multiple_assigment = idx_pred_multiple_assigment[0][
                    idx_max_distance_multiple_assigment
                ]
                # Switch the assignment to the next closest point by rolling the row with the highest distance one to the left
                distance[idx_max_distance_multiple_assigment, :] = np.roll(
                    distance[idx_max_distance_multiple_assigment, :], -1
                )
                # To avoid infinite loops, we set the last element to np.inf.
                distance[idx_max_distance_multiple_assigment, -1] = np.inf
                idx_assigment_pred[idx_max_distance_multiple_assigment, :] = np.roll(
                    idx_assigment_pred[idx_max_distance_multiple_assigment, :], -1
                )
                # Update points with no assignment with -1
                idx_assigment_pred[distance == np.inf] = -1
                # Check if a ground truth point is assigned to more than one predicted point
                pred_multiple_assignments = SourceDetectionEvaluation.__return_multiple_assigned_detected_points(
                    idx_assigment_pred
                )

        assigments = np.array(
            [idx_assigment_pred[:, 0], np.arange(detected.shape[0]), distance[:, 0]]
        ).T

        # If there are more predicitons than GTs, we need to add the missing GTs.
        missing_gts = np.setdiff1d(np.arange(ground_truth.shape[0]), assigments[:, 0])
        missing_gts = missing_gts[missing_gts != -1]
        if len(missing_gts) > 0:
            missing_gts = np.array(
                [
                    missing_gts,
                    np.full(len(missing_gts), -1),
                    np.full(len(missing_gts), np.inf),
                ]
            )
            assigments = np.vstack([assigments, missing_gts.T])
        return assigments[assigments[:, 0].argsort()]

    @staticmethod
    def calculate_evaluation_measures(assignments: np.ndarray) -> tuple:
        """
        Calculates the True Positive (TP), False Positive (FP) and False Negative (FN) of the ground truth and predictions.
        - TP are the detections associated with a source
        - FP are detections without any associated source
        - FN are sources with no associations with a detection

        :param assignments: nx3 did np.ndarray where each row represents an assignment
            The `assignments` is expected to be as `automatic_assignment_of_ground_truth_and_prediction` return.
            Therefore, the non-assigned sources must have a value of "-1".

        :return: TP, FP, FN
        """
        tp = assignments[
            np.logical_and(assignments[:, 1] != -1, assignments[:, 0] != -1), :
        ].shape[0]
        fp = assignments[assignments[:, 1] == -1, :].shape[0]
        fn = assignments[assignments[:, 0] == -1, :].shape[0]
        return tp, fp, fn

    def plot(
        self,
        exclude_img: bool = False,
        show_legend: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot the found sources as green x's and the source truth as red 'o' on the original image,
         that the source detection was performed on.
        """

        if self.source_detection.has_source_image() and not exclude_img:
            image = self.source_detection.get_source_image()
            wcs = WCS(image.header)
            slices = get_slices(wcs)

            _, ax = plt.subplots(1, 1, subplot_kw=dict(projection=wcs, slices=slices))
            ax.imshow(image.data[0][0], cmap="jet", origin="lower", interpolation=None)
        else:
            _, ax = plt.subplots(1, 1, subplot_kw=dict())
        self.__plot_truth_and_prediction(ax, show_legend=show_legend)

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
        source_matches = assignment[assignment[:, 2] != np.inf, :]
        truth_indexes = np.array(source_matches[:, 0], dtype=int)
        pred_indexes = np.array(source_matches[:, 1], dtype=int)
        meta = source_matches[:, 2]

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
            [[0.0, self.fn], [self.fp, self.tp]]
        )

    def get_accuracy(self) -> float:
        return self.tp / (
            self.tp + self.fp + self.fn
        )

    def get_precision(self) -> float:
        return self.tp / (self.tp + self.fp)

    def get_sensitivity(self) -> float:
        return self.tp / (self.tp + self.fn)

    def get_f_score(self) -> float:
        p = self.get_precision()
        sn = self.get_sensitivity()
        return 2 * (p * sn / (p + sn))

    def plot_error_ra_dec(
        self,
        filename: Optional[str] = None,
    ) -> None:
        truth_assigned = self.sky_array_gt_assigned[:,:-1].astype(np.float64)
        detection_assigned = self.detected_sources_array_pred_assigned.astype(np.float64)

        # get ra-dec error
        ra_dec_truth = truth[:, [1, 2]].transpose()
        ra_dec_det = detection[:, [1, 2]].transpose()
        error = ra_dec_truth - ra_dec_det

        plt.xlabel("RA (deg) error / x")
        plt.ylabel("DEC (deg) error / y")
        plt.plot(assignment_error[0], assignment_error[1], "o", markersize=8, color="r", alpha=0.5)
        if filename:
            plt.savefig(filename)
            plt.show(block=False)
        else:
            plt.show()

    def plot_confusion_matrix(self, file_name=None):
        conf_matrix = self.get_confusion_matrix()
        _, ax = plt.subplots()
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(
                    x=j,
                    y=i,
                    s=int(conf_matrix[i, j]),
                    va="center",
                    ha="center",
                    size="x-large",
                )

        plt.xlabel("Predicted", fontsize=13)
        plt.ylabel("Reference", fontsize=13)
        plt.title("Confusion Matrix", fontsize=13)

        if file_name:
            plt.savefig(file_name)
            plt.show(block=False)
        else:
            plt.show()

    def plot_quiver_positions(self, filename=None):
        ref = self.get_truth_array()[:, [1, 2]].transpose().astype(float)
        pred = self.get_detected_array()[:, [1, 2]].transpose().astype(float)
        ra_ref = np.array(ref[0], dtype=float)
        dec_ref = np.array(ref[1], dtype=float)
        num = len(ra_ref)

        error = truth - pred
        ra_error = error[0] * (np.cos(np.deg2rad(dec_ref)))
        dec_error = error[1]
        _, ax = plt.subplots()
        if np.mean(np.deg2rad(dec_ref)) != 0.0:
            ax.set_aspect(1.0 / np.cos(np.mean(np.deg2rad(dec_ref))))
        _ = ax.quiver(ra_ref, dec_ref, ra_error, dec_error, color="b")

        ax.scatter(ra_ref, dec_ref, color="r", s=8)
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        plt.title(f"Matched {num} sources")
        if filename:
            plt.savefig(filename)
        plt.show(block=False)
        plt.pause(1)

    def plot_flux_ratio_to_distance(
        self,
        filename: Optional[str] = None,
    ) -> None:
        truth = self.sky_array_gt_assigned[:,[0,1,2]].astype(np.float64).T # used to be 5 instead of 2!?
        pred = self.detected_sources_array_pred_assigned[:,[1,2,5]].astype(np.float64).T
        ra_dec_pred = truth[[0, 1]]
        ra_ref = truth[0]
        dec_ref = truth[1]
        flux_ref = truth[2]
        ra_pred = pred[0]
        dec_pred = pred[1]
        flux_pred = pred[2]
        phase_center = self.source_detection.get_source_image().get_phase_center()

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
        plt.ylabel("Flux Ratio (Pred/Ref)")
        if filename:
            plt.savefig(filename)
        plt.show(block=False)
        plt.pause(1)


    def plot_flux_ratio_to_ra_dec(
        self,
        filename: Optional[str] = None,
    ) -> None:
        truth = self.sky_array_gt_assigned[:,[0,1,2]].astype(np.float64).T
        pred = self.detected_sources_array_pred_assigned[:,[1,2,5]].astype(np.float64).T
        ra_pred = pred[0]
        dec_pred = pred[1]
        flux_ref = truth[2]
        flux_pred = pred[2]

        flux_ratio = flux_pred / flux_ref

        # Flux ratio vs. RA & Dec
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        fig.suptitle("Flux ratio vs. Position")
        ax1.plot(ra_pred, flux_ratio, "o", color="b", markersize=5, alpha=0.5)
        ax2.plot(dec_pred, flux_ratio, "o", color="b", markersize=5, alpha=0.5)

        ax1.set_xlabel("RA (deg)")
        ax2.set_xlabel("Dec (deg)")
        ax1.set_ylabel("Flux ratio (Pred/Ref)")
        if filename:
            plt.savefig(filename)
        plt.show(block=False)
        plt.pause(1)


    def plot_flux_histogram(
        self,
        nbins: int = 10,
        filename: Optional[str] = None,
    ) -> None:
        flux_in = self.sky_array_gt_assigned[:,2].astype(np.float64)
        flux_out = self.detected_sources_array_pred_assigned[:,5].astype(np.float64)

        flux_in = flux_in[flux_in > 0.0]
        flux_out = flux_out[flux_out > 0.0]

        hist = [flux_in, flux_out]
        labels = ["Flux Reference", "Flux Predicted"]
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
        plt.pause(1)

