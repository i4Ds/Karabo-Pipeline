import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

from karabo.util.plotting_util import Font

RUN_SLOW_TESTS = os.environ.get("RUN_SLOW_TESTS", "false").lower() == "true"
IS_GITHUB_RUNNER = os.environ.get("IS_GITHUB_RUNNER", "false").lower() == "true"


# Test preparation moved to fixture
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    path_to_notebooks = os.path.join("karabo", "examples")
    os.chdir(path_to_notebooks)


@pytest.mark.skip(reason="Not a test")
def _test_notebook(notebook):
    print(Font.BOLD + Font.BLUE + "Testing notebook " + notebook + Font.END)

    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=-1)
        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except Exception:
            assert False, f"Failed executing {notebook}"


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="IS_GITHUB_RUNNER")
def test_source_detection_notebook():
    _test_notebook(notebook="source_detection.ipynb")


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="IS_GITHUB_RUNNER")
def test_source_detection_assesment_notebook():
    _test_notebook(notebook="source_detection_assessment.ipynb")


@pytest.mark.skipif(not RUN_SLOW_TESTS, reason="SLOW_TESTS or IS_GITHUB_RUNNER")
def test_HIIM_Img_Recovery_notebook():
    _test_notebook(notebook="HIIM_Img_Recovery.ipynb")
