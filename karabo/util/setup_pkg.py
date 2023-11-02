"""This .py file is ONLY for setup-specific util-functions.
Thus, ONLY deps at building-time are allowed here.
If you don't know what that means, don't touch anything here.
"""
import os
from sysconfig import get_path


def set_rascil_data_directory_env() -> None:
    """
    Sets specific environment variables
    that the jupyter kernel is not loading by default.

    This function is idempotent (running it more than once brings no side effects).

    """
    lib_dir = os.path.dirname(os.path.dirname(os.path.dirname(get_path("platlib"))))
    data_folder = os.path.join(lib_dir, "data")
    os.environ["RASCIL_DATA"] = data_folder
