import unittest
from typing import Any

from karabo.util.data_util import get_module_absolute_path


def run_tests(
    verbosity: int = 0,
    pattern: str = "test*.py",
    *args: Any,
    **kwargs: Any,
) -> None:
    loader = unittest.TestLoader()

    # Get location of karabo package
    start_dir = get_module_absolute_path()
    suite = loader.discover(start_dir, pattern=pattern)

    runner = unittest.TextTestRunner(*args, **kwargs)
    test_result = runner.run(suite)
    # Assert that the tests passed
    assert test_result.wasSuccessful()
