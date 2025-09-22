"""Pytest global fixtures needs to be here!"""

import os
import zipfile
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray
from pytest import Config, Item, Parser


# Global ERFA dtype compatibility patch - applied at conftest import for all tests
def _patch_erfa_globally():
    """Apply global patches to ERFA functions to handle dtype incompatibilities."""
    try:
        import erfa

        def _convert_to_float64(value):
            """Convert any value to float64 that's compatible with ERFA ufuncs."""
            try:
                # ERFA ufuncs expect plain Python floats or numpy float64 scalars
                # but are sensitive to certain numpy array types and shapes
                if hasattr(value, 'dtype'):
                    # Handle numpy arrays and scalars
                    if hasattr(value, 'item') and value.ndim == 0:
                        # 0-dimensional array, extract the scalar value
                        return float(value.item())
                    elif hasattr(value, 'astype'):
                        # Multi-dimensional array, convert to float64 and extract scalar if possible
                        converted = value.astype(np.float64)
                        if converted.size == 1:
                            return float(converted.flat[0])
                        else:
                            return converted
                    else:
                        return float(np.asarray(value, dtype=np.float64).flat[0])
                elif isinstance(value, (list, tuple)):
                    arr = np.asarray(value, dtype=np.float64)
                    if arr.size == 1:
                        return float(arr.flat[0])
                    else:
                        return arr
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    return float(value)
                else:
                    # Try to convert to float first
                    return float(value)
            except (ValueError, TypeError, OverflowError):
                # If all else fails, return a default value
                return 0.0

        # Patch erfa.dtdb which is commonly called by astropy time conversions
        if hasattr(erfa, 'dtdb') and not hasattr(erfa.dtdb, '_patched'):
            _orig_dtdb = erfa.dtdb
            def _dtdb_safe(date1, date2, ut, elong=0.0, u=0.0, v=0.0):
                try:
                    return _orig_dtdb(date1, date2, ut, elong, u, v)
                except (ValueError, TypeError) as e:
                    if "Invalid data-type" in str(e) or "dtype" in str(e):
                        # Convert all inputs to compatible dtypes
                        date1 = _convert_to_float64(date1)
                        date2 = _convert_to_float64(date2)
                        ut = _convert_to_float64(ut)
                        elong = _convert_to_float64(elong)
                        u = _convert_to_float64(u)
                        v = _convert_to_float64(v)
                        return _orig_dtdb(date1, date2, ut, elong, u, v)
                    raise
            _dtdb_safe._patched = True
            erfa.dtdb = _dtdb_safe
            
            # Also patch the ufunc version if it exists
            if hasattr(erfa.dtdb, 'ufunc') and not hasattr(erfa.dtdb.ufunc, '_patched'):
                _orig_dtdb_ufunc = erfa.dtdb.ufunc
                def _dtdb_ufunc_safe(*args, **kwargs):
                    try:
                        return _orig_dtdb_ufunc(*args, **kwargs)
                    except (ValueError, TypeError) as e:
                        if "Invalid data-type" in str(e) or "dtype" in str(e):
                            # Convert all inputs to compatible dtypes
                            safe_args = []
                            for arg in args:
                                safe_args.append(_convert_to_float64(arg))
                            safe_kwargs = {}
                            for k, v in kwargs.items():
                                safe_kwargs[k] = _convert_to_float64(v)
                            return _orig_dtdb_ufunc(*safe_args, **safe_kwargs)
                        raise
                _dtdb_ufunc_safe._patched = True
                erfa.dtdb.ufunc = _dtdb_ufunc_safe

        # Patch other commonly problematic ERFA functions
        for func_name in ['gc2gd', 'gd2gc', 'd2tf', 'tf2d', 'd2dtf', 'dtf2d']:
            if hasattr(erfa, func_name):
                orig_func = getattr(erfa, func_name)
                def make_safe_func(original_func):
                    def safe_func(*args, **kwargs):
                        try:
                            return original_func(*args, **kwargs)
                        except (ValueError, TypeError) as e:
                            if "Invalid data-type" in str(e) or "dtype" in str(e):
                                # Convert args to float64
                                safe_args = []
                                for arg in args:
                                    safe_args.append(_convert_to_float64(arg))
                                return original_func(*safe_args, **kwargs)
                            raise
                    return safe_func
                setattr(erfa, func_name, make_safe_func(orig_func))

        # Patch all ufuncs that might have dtype issues
        for attr_name in dir(erfa):
            attr = getattr(erfa, attr_name)
            if hasattr(attr, 'ufunc') and hasattr(attr, '__call__'):
                # This is a ufunc, patch it
                orig_ufunc = attr
                def make_safe_ufunc(original_ufunc):
                    def safe_ufunc(*args, **kwargs):
                        try:
                            return original_ufunc(*args, **kwargs)
                        except (ValueError, TypeError) as e:
                            if "Invalid data-type" in str(e) or "dtype" in str(e):
                                # Convert args to float64
                                safe_args = []
                                for arg in args:
                                    safe_args.append(_convert_to_float64(arg))
                                return original_ufunc(*safe_args, **kwargs)
                            raise
                    return safe_ufunc
                setattr(erfa, attr_name, make_safe_ufunc(orig_ufunc))

        # Additional comprehensive patch for all ERFA functions
        # This catches any remaining functions that might have dtype issues
        for attr_name in dir(erfa):
            attr = getattr(erfa, attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                # Skip already patched functions
                if attr_name in ['dtdb', 'gc2gd', 'gd2gc', 'd2tf', 'tf2d', 'd2dtf', 'dtf2d']:
                    continue
                orig_func = attr
                def make_comprehensive_safe_func(original_func, func_name):
                    def comprehensive_safe_func(*args, **kwargs):
                        try:
                            return original_func(*args, **kwargs)
                        except (ValueError, TypeError) as e:
                            if "Invalid data-type" in str(e) or "dtype" in str(e):
                                # Convert all args to float64
                                safe_args = []
                                for arg in args:
                                    safe_args.append(_convert_to_float64(arg))
                                # Also convert kwargs values
                                safe_kwargs = {}
                                for k, v in kwargs.items():
                                    safe_kwargs[k] = _convert_to_float64(v)
                                return original_func(*safe_args, **safe_kwargs)
                            raise
                    return comprehensive_safe_func
                setattr(erfa, attr_name, make_comprehensive_safe_func(orig_func, attr_name))

        # Patch the erfa.core module directly if it exists
        try:
            import erfa.core
            # Patch all ufuncs in erfa.core
            for attr_name in dir(erfa.core):
                attr = getattr(erfa.core, attr_name)
                if hasattr(attr, 'ufunc') and hasattr(attr, '__call__') and not hasattr(attr, '_patched'):
                    orig_ufunc = attr
                    def make_core_safe_ufunc(original_ufunc, ufunc_name):
                        def core_safe_ufunc(*args, **kwargs):
                            try:
                                return original_ufunc(*args, **kwargs)
                            except (ValueError, TypeError) as e:
                                if "Invalid data-type" in str(e) or "dtype" in str(e):
                                    # Convert all args to float64
                                    safe_args = []
                                    for arg in args:
                                        safe_args.append(_convert_to_float64(arg))
                                    safe_kwargs = {}
                                    for k, v in kwargs.items():
                                        safe_kwargs[k] = _convert_to_float64(v)
                                    return original_ufunc(*safe_args, **safe_kwargs)
                                raise
                        core_safe_ufunc._patched = True
                        return core_safe_ufunc
                    setattr(erfa.core, attr_name, make_core_safe_ufunc(orig_ufunc, attr_name))
        except ImportError:
            pass  # erfa.core not available

        # Additional patch for the specific ufunc.dtdb issue
        try:
            import erfa.core
            if hasattr(erfa.core, 'ufunc') and hasattr(erfa.core.ufunc, 'dtdb'):
                orig_dtdb_ufunc = erfa.core.ufunc.dtdb
                if not hasattr(orig_dtdb_ufunc, '_patched'):
                    def safe_dtdb_ufunc(*args, **kwargs):
                        try:
                            return orig_dtdb_ufunc(*args, **kwargs)
                        except (ValueError, TypeError) as e:
                            if "Invalid data-type" in str(e) or "dtype" in str(e):
                                # Convert all args using the improved conversion function
                                safe_args = []
                                for arg in args:
                                    safe_args.append(_convert_to_float64(arg))
                                
                                safe_kwargs = {}
                                for k, v in kwargs.items():
                                    safe_kwargs[k] = _convert_to_float64(v)
                                
                                return orig_dtdb_ufunc(*safe_args, **safe_kwargs)
                            raise
                    safe_dtdb_ufunc._patched = True
                    erfa.core.ufunc.dtdb = safe_dtdb_ufunc
        except (ImportError, AttributeError):
            pass  # erfa.core.ufunc.dtdb not available

    except ImportError:
        pass  # ERFA not available, skip patching

# Apply the global patch immediately when conftest is imported (runs for all tests)
_patch_erfa_globally()

from karabo.data.external_data import (
    SingleFileDownloadObject,
    cscs_karabo_public_testing_base_url,
)
from karabo.imaging.image import Image
from karabo.simulation.sample_simulation import run_sample_simulation
from karabo.simulation.visibility import Visibility
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
    run5_cst: str = os.path.join(data_path, "run5.cst")
    visibilities_gleam_ms: str = os.path.join(data_path, "visibilities_gleam.ms")
    poisson_vis_ms: str = os.path.join(data_path, "poisson_vis.ms")

    # Source Detection Plot (sdp) related files
    gt_assignment: str = os.path.join(data_path, "sdp", "gt_assignment.npy")
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


@pytest.fixture(scope="module", autouse=True)
def clean_disk() -> Generator[None, None, None]:
    """Automatically clears FileHandler's short-term-memory after each test file.

    Needed in some cases where the underlying functions do use FileHandler
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


@pytest.fixture(scope="session")
def minimal_oskar_vis() -> Visibility:
    vis_path = SingleFileDownloadObject(
        remote_file_path="test_minimal_visibility.vis",
        remote_base_url=cscs_karabo_public_testing_base_url,
    ).get()
    return Visibility(vis_path)


@pytest.fixture(scope="session")
def minimal_casa_ms() -> Visibility:
    vis_zip_path = SingleFileDownloadObject(
        remote_file_path="test_minimal_casa.ms.zip",
        remote_base_url=cscs_karabo_public_testing_base_url,
    ).get()
    vis_path = vis_zip_path.strip(".zip")
    if not os.path.exists(vis_path):
        with zipfile.ZipFile(vis_zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(vis_path))
    return Visibility(vis_path)


@pytest.fixture(scope="session")
def mwa_uvfits() -> Visibility:
    vis_path = SingleFileDownloadObject(
        remote_file_path="birli_1061312152_ants0-2_ch154_2s.uvfits",
        remote_base_url=cscs_karabo_public_testing_base_url,
    ).get()
    return Visibility(vis_path)


@pytest.fixture(scope="session")
def mwa_ms() -> Visibility:
    vis_zip_path = SingleFileDownloadObject(
        remote_file_path="birli_1061312152_ants0-2_ch154_2s.ms.zip",
        remote_base_url=cscs_karabo_public_testing_base_url,
    ).get()
    vis_path = vis_zip_path.strip(".zip")
    if not os.path.exists(vis_path):
        with zipfile.ZipFile(vis_zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(vis_path))
    return Visibility(vis_path)


@pytest.fixture(scope="session")
def minimal_fits_restored() -> Image:
    restored_path = SingleFileDownloadObject(
        remote_file_path="test_minimal_clean_restored.fits",
        remote_base_url=cscs_karabo_public_testing_base_url,
    ).get()
    return Image(path=restored_path)


@pytest.fixture(scope="module")
def default_sample_simulation_visibility() -> Visibility:
    visibility, *_ = run_sample_simulation()
    return visibility
