import os
import sys
import unittest

RUN_SLOW_TESTS = os.environ.get("RUN_SLOW_TESTS", "false").lower() == "true"
KERNEL_NAME = os.environ.get("KERNEL_NAME", "karabo")
KARABO_PATH = os.environ.get("KARABO_PATH")
if KARABO_PATH is not None:
    sys.path.insert(0, KARABO_PATH)


class TestJupyterNotebooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os

        path_to_notebooks = os.path.join("karabo", "examples")
        os.chdir(path_to_notebooks)

    def _test_notebook(self, notebook):
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        with open(notebook) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600, kernel_name=KERNEL_NAME)
            try:
                assert (
                    ep.preprocess(nb) is not None
                ), f"Got empty notebook for {notebook}"
            except Exception:
                assert False, f"Failed executing {notebook}"

    @unittest.skip("Test needs a new dependency and a access code")
    def test_meerKAT_data_access_notebook(self):
        self._test_notebook("MeerKAT_data_access.ipynb")

    @unittest.skipIf(not RUN_SLOW_TESTS, "Not running slow tests")
    def test_source_detection_notebook(self):
        self._test_notebook("source_detection.ipynb")

    @unittest.skipIf(not RUN_SLOW_TESTS, "Not running slow tests")
    def test_source_detection_assesment_notebook(self):
        self._test_notebook("source_detection_assessment.ipynb")
