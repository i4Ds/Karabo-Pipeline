import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

from karabo.test.conftest import IS_GITHUB_RUNNER
from karabo.util.plotting_util import Font

# get notebook-dir not matter cwd
notebook_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def _run_notebook(notebook: str) -> None:
    notebook = os.path.join(notebook_dir, notebook)
    print(Font.BOLD + Font.BLUE + "Testing notebook " + notebook + Font.END)

    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=-1)
        cwd = os.getcwd()
        os.chdir(notebook_dir)
        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except Exception:
            pytest.fail(reason=f"Failed executing {notebook}")
        finally:
            os.chdir(cwd)


@pytest.mark.skipif(
    IS_GITHUB_RUNNER,
    reason="Error: Process completed with exit code 143",
)
def test_source_detection_notebook() -> None:
    _run_notebook(notebook="source_detection.ipynb")


@pytest.mark.skipif(
    IS_GITHUB_RUNNER,
    reason="System.IO.IOException: No space left on device",
)
def test_source_detection_assesment_notebook() -> None:
    _run_notebook(notebook="source_detection_assessment.ipynb")


@pytest.mark.skipif(
    IS_GITHUB_RUNNER,
    reason="System.IO.IOException: No space left on device",
)
def test_HIIM_Img_Recovery_notebook() -> None:
    _run_notebook(notebook="HIIM_Img_Recovery.ipynb")
