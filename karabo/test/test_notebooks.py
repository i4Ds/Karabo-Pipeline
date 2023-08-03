import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

# from karabo.test.conftest import RUN_SLOW_TESTS
from karabo.util.plotting_util import Font

# get notebook-dir not matter cwd
notebook_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def _run_notebook(notebook: str) -> None:
    notebook = os.path.join(notebook_dir, notebook)
    print(Font.BOLD + Font.BLUE + "Testing notebook " + notebook + Font.END)

    with open(notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=-1)
        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except Exception:
            pytest.fail(reason=f"Failed executing {notebook}")


# @pytest.mark.skipif(not RUN_SLOW_TESTS, reason="RUN_SLOW_TESTS")
# def test_source_detection_notebook() -> None:
#     _run_notebook(notebook="source_detection.ipynb")


# @pytest.mark.skipif(not RUN_SLOW_TESTS, reason="RUN_SLOW_TESTS")
def test_source_detection_assesment_notebook() -> None:
    _run_notebook(notebook="source_detection_assessment.ipynb")


# @pytest.mark.skipif(not RUN_SLOW_TESTS, reason="SLOW_TESTS")
# def test_HIIM_Img_Recovery_notebook() -> None:
#     _run_notebook(notebook="HIIM_Img_Recovery.ipynb")
