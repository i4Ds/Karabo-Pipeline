"""Pytest global fixtures needs to be here!"""
import os
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

from karabo.test import data_path


@dataclass
class TFiles:
    """Read-only repo-artifact paths.

    It is assumed that all artifacts here exist in the repo
     and therefore are not in the .gitignore.

    The defined artifacts should reflect the `karabo/test/data/* file
     and dir paths.
    """

    cst_like_beam_port_1_txt: str = os.path.join(data_path, "cst_like_beam_port_1.txt")
    cst_like_beam_port_2_txt: str = os.path.join(data_path, "cst_like_beam_port_2.txt")
    detection_csv: str = os.path.join(data_path, "detection.csv")
    detection_zip: str = os.path.join(data_path, "detection.zip")
    detection_result_csv: str = os.path.join(data_path, "detection_result_512.px.csv")
    filtered_sky_csv: str = os.path.join(data_path, "filtered_sky.csv")
    restored_fits: str = os.path.join(data_path, "restored.fits")
    run5_cst: str = os.path.join(data_path, "run5.cst")
    visibilities_gleam_ms: str = os.path.join(data_path, "visibilities_gleam.ms")
    poission_vis_ms: str = os.path.join(data_path, "poisson_vis.ms")

    # Source Detection Plot (sdp) related files
    gt_assigment: str = os.path.join(data_path, "sdp", "gt_assigment.npy")
    gt_plot: str = os.path.join(data_path, "sdp", "gt_plot.png")
    gt_plot_error_ra_dec: str = os.path.join(
        data_path, "sdp", "gt_plot_error_ra_dec.png"
    )
    gt_plot_flux_histogram: str = os.path.join(
        data_path, "sdp", "gt_plot_flux_histogram.png"
    )
    gt_plot_flux_ratio_to_distance: str = os.path.join(
        data_path, "sdp", "gt_plot_flux_ratio_to_distance.png"
    )
    gt_plot_flux_ratio_to_ra_dec: str = os.path.join(
        data_path, "sdp", "gt_plot_flux_ratio_to_ra_dec.png"
    )
    gt_plot_quiver_positions: str = os.path.join(
        data_path, "sdp", "gt_plot_quiver_positions.png"
    )


@pytest.fixture(scope="session")
def tobject() -> TFiles:
    return TFiles()


@pytest.fixture(scope="function")
def sky_data_with_ids() -> NDArray[np.object_]:
    return np.array(
        [
            [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0, "source1"],
            [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45, "source2"],
            [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10, "source3"],
        ]
    )


@pytest.fixture(scope="function")
def sky_data(sky_data_with_ids: NDArray[np.object_]) -> NDArray[np.float64]:
    return sky_data_with_ids[:, :-1].astype(np.float64)


@pytest.fixture(scope="session")
def compare_images() -> Callable:
    """Compare two images."""

    def _compare_images(img_path_1, img_path_2):
        img1 = plt.imread(img_path_1)
        img2 = plt.imread(img_path_2)
        assert img1.shape == img2.shape
        # Calculate the error between the two images
        diff = np.linalg.norm(img1 - img2) / (img1.shape[0] * img1.shape[1])
        print(f"Max difference: {diff}")
        assert diff < 1e-3

    return _compare_images
