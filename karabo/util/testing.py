import os
from types import TracebackType
from typing import Optional

import pytest


class ChangeWorkingDir:
    """Changes temporarily working dir for test-discovery."""

    def __init__(self) -> None:
        self.cwd = os.getcwd()

    def __enter__(self) -> None:
        os.chdir(os.path.dirname(os.path.dirname(__file__)))

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        os.chdir(self.cwd)


def run_tests(
    pytest_args: Optional[str] = None,
) -> None:
    """Launches pytest.

    Args:
        args: pytest cli-args, e.g. "-k test_my_favorite"
    """
    if pytest_args is not None:
        args = pytest_args.split(" ")
    else:
        args = None
    with ChangeWorkingDir():
        pytest.main(args=args)
