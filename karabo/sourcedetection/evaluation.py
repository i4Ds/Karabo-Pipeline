import numpy as np
from astropy.wcs import WCS
from matplotlib import pyplot as plt

from karabo.simulation.sky_model import SkyModel
from karabo.sourcedetection.result import SourceDetectionResult
from karabo.util.plotting_util import get_slices


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
        truth_and_pred_coords = self.get_truth_to_detection_pixel_coordinate_array().transpose()
        ax.plot(truth_and_pred_coords[0, :], truth_and_pred_coords[1, :], 'o', linewidth=5,
                color="firebrick", alpha=0.5)
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
        pred_indexes = np.array(self.assignment[:, 2], dtype=int)
        preds = self.pixel_coordinates_sky[:, pred_indexes]
        return np.vstack((truths, preds)).transpose()
