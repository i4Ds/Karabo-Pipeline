"""Pytest global fixtures needs to be here!"""
import os
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray
from pytest import Config, Item, Parser

from karabo.test import data_path
from karabo.util.file_handler import FileHandler

NNImageDiffCallable = Callable[[str, str], float]

IS_GITHUB_RUNNER = os.environ.get("IS_GITHUB_RUNNER", "false").lower() == "true"
RUN_GPU_TESTS = os.environ.get("RUN_GPU_TESTS", "false").lower() == "true"


def pytest_addoption(parser: Parser) -> None:
    """Pytest custom argparse hook.

    Add custom argparse options here.

    Pytest argparse-options have to be declared in the root conftest.py.
    For some reason, the root conftest.py has to live near the project-root, even if
    only a single conftest.py exists. However, this prevents using `pytest .` with
    custom argparse-coptions from the root. Instead, either specify the test-dir
    or leave it out entirely.

    Args:
        parser: pytest.Parser
    """
    parser.addoption(
        "--only-mpi",
        action="store_true",
        default=False,
        help="run only mpi tests",
    )


def pytest_configure(config: Config) -> None:
    """Pytest add ini-values.

    Args:
        config: pytest.Config
    """
    config.addinivalue_line("markers", "mpi: mark mpi-tests as mpi")


def pytest_collection_modifyitems(config: Config, items: Iterable[Item]) -> None:
    """Pytest modify-items hook.

    Change pytest-behavior dependent on parsed input.

    See https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option

    Args:
        config: pytest.Config
        items: iterable of pytest.Item
    """  # noqa: E501
    if not config.getoption("--only-mpi"):
        skipper = pytest.mark.skip(reason="Only run when --only-mpi is given")
        for item in items:
            if "mpi" in item.keywords:
                item.add_marker(skipper)
    else:
        skipper = pytest.mark.skip(reason="Don't run when --only-mpi is given")
        for item in items:
            if "mpi" not in item.keywords:
                item.add_marker(skipper)


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

    blank_image_file_for_source_detection: str = os.path.join(
        data_path, "blank_image.fits"
    )


@pytest.fixture(scope="session")
def tobject() -> TFiles:
    return TFiles()


@pytest.fixture(scope="function", autouse=True)
def clean_disk() -> Generator[None, None, None]:
    """Automatically clears FileHandler's short-term-memory after each test.

    Needed in some cases where the underlying functions do use FileHanlder
     which could lead to IOError because of disk-space limitations.
    """
    # Setup: fill with logic
    yield  # testing happens here
    # Teardown: fill with logic
    FileHandler.clean()
    plt.close("all")


@pytest.fixture(scope="function")
def sky_data_with_ids() -> NDArray[np.object_]:
    return np.array(
        [
            [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0, 1, 1.01, "source1"],
            [
                20.0,
                -30.5,
                3,
                2,
                2,
                0,
                100.0e6,
                -0.7,
                0.0,
                600,
                50,
                45,
                1,
                1.01,
                "source2",
            ],
            [
                20.5,
                -30.5,
                3,
                0,
                0,
                2,
                100.0e6,
                -0.7,
                0.0,
                700,
                10,
                -10,
                1,
                1.01,
                "source3",
            ],
        ]
    )


@pytest.fixture(scope="function")
def sky_data(sky_data_with_ids: NDArray[np.object_]) -> NDArray[np.float64]:
    return sky_data_with_ids[:, :-1].astype(np.float64)


@pytest.fixture(scope="session")
def normalized_norm_diff() -> NNImageDiffCallable:
    """Compare two images."""

    def _normalized_norm_diff(img_path_1: str, img_path_2: str) -> float:
        img1 = plt.imread(img_path_1)
        img2 = plt.imread(img_path_2)
        assert img1.shape == img2.shape
        # Calculate the error between the two images
        return float(np.linalg.norm(img1 - img2) / (img1.shape[0] * img1.shape[1]))

    return _normalized_norm_diff
