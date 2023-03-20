import os
import unittest

from karabo.util.plotting_util import Font

RUN_SLOW_TESTS = os.environ.get("RUN_SLOW_TESTS", "false").lower() == "true"
KERNEL_NAME = os.environ.get("KERNEL_NAME", "karabo")


class TestJupyterNotebooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os

        path_to_notebooks = os.path.join("karabo", "examples")
        os.chdir(path_to_notebooks)

    def _test_notebook(self, notebook):
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        print(Font.BOLD + Font.BLUE + "Testing notebook " + notebook + Font.END)

        with open(notebook) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=-1, kernel_name=KERNEL_NAME)
            try:
                assert (
                    ep.preprocess(nb) is not None
                ), f"Got empty notebook for {notebook}"
            except Exception:
                assert False, f"Failed executing {notebook}"

    def test_source_detection_notebook(self):
        self._test_notebook(notebook="source_detection.ipynb")

    def test_source_detection_assesment_notebook(self):
        self._test_notebook(notebook="source_detection_assessment.ipynb")

    @unittest.skipIf(not RUN_SLOW_TESTS, "Not running slow tests")
    def test_HIIM_Img_Recovery_notebook(self):
        self._test_notebook(notebook="HIIM_Img_Recovery.ipynb")
