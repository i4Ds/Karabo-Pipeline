import unittest

from karabo.util.data_util import get_module_absolute_path


def run_tests(verbosity=0, pattern="test*.py", *args, **kwargs):
    loader = unittest.TestLoader()

    # Get location of karabo package
    start_dir = get_module_absolute_path()
    suite = loader.discover(start_dir, pattern=pattern)

    runner = unittest.TextTestRunner(verbosity=verbosity, *args, **kwargs)
    test_result = runner.run(suite)
    # Assert that the tests passed
    assert test_result.wasSuccessful()
