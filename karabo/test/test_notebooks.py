import os

import pytest

from karabo.test.conftest import IS_GITHUB_RUNNER
from karabo.util.plotting_util import Font

RUN_NOTEBOOK_TESTS = os.environ.get("RUN_NOTEBOOK_TESTS", "false").lower() == "true"

# get notebook-dir not matter cwd
notebook_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def _run_notebook(notebook: str) -> None:
    # Imported lazily: these are dev-only dependencies (see pyproject.toml) and
    # are not part of the conda runtime deps, so importing them at module level
    # would break test collection in the conda-build pipeline.
    import nbformat
    import nest_asyncio
    from nbconvert.preprocessors import ExecutePreprocessor

    nest_asyncio.apply()

    notebook = os.path.join(notebook_dir, notebook)
    print(Font.BOLD + Font.BLUE + "Testing notebook " + notebook + Font.END)

    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=-1)
        cwd = os.getcwd()
        os.chdir(notebook_dir)
        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except AssertionError as e:
            pytest.fail(reason=f"Assertion error, details: {e}")
        except Exception as e:
            pytest.fail(reason=f"Failed executing {notebook}, Exception: {e}")
        finally:
            os.chdir(cwd)


@pytest.mark.skipif(
    IS_GITHUB_RUNNER or not RUN_NOTEBOOK_TESTS,
    reason="Error: Process completed with exit code 143",
)
def test_source_detection_notebook() -> None:
    _run_notebook(notebook="source_detection.ipynb")


@pytest.mark.skipif(
    not RUN_NOTEBOOK_TESTS,
    reason="'Error: The operation was canceled' when running this test on the package",
)
def test_source_detection_big_files_notebook() -> None:
    _run_notebook(notebook="source_detection_big_files.ipynb")


@pytest.mark.skipif(
    IS_GITHUB_RUNNER or not RUN_NOTEBOOK_TESTS,
    reason="'Error: The operation was canceled' when running this test on the package",
)
def test_source_detection_assessment_notebook() -> None:
    _run_notebook(notebook="source_detection_assessment.ipynb")


@pytest.mark.skipif(
    not RUN_NOTEBOOK_TESTS,
    reason="'Error: The operation was canceled' when running this test on the package",
)
def test_line_emission_notebook() -> None:
    _run_notebook(notebook="LineEmissionSimulation.ipynb")


@pytest.mark.skipif(
    not RUN_NOTEBOOK_TESTS,
    reason="'Error: The operation was canceled' when running this test on the package",
)
def test_ImageMosaicker_notebook() -> None:
    _run_notebook(notebook="ImageMosaicker.ipynb")


@pytest.mark.skipif(
    IS_GITHUB_RUNNER or not RUN_NOTEBOOK_TESTS,
    reason="'Error: The operation was canceled' when running this test on the package",
)
def test_imaging_notebook() -> None:
    _run_notebook(notebook="imaging.ipynb")
