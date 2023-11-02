"""This file is executed during build-time and when karabo gets imported.
Hence, you ONLY have deps available here which are available during build-time and
in karabo. If you don't know what that means, don't touch anything here.
"""
import os
import platform
import sys

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

if "WSL" in platform.release() and (
    os.environ.get("LD_LIBRARY_PATH") is None
    or "wsl" not in os.environ["LD_LIBRARY_PATH"]
):
    wsl_ld_path = "/usr/lib/wsl/lib"
    if os.environ.get("LD_LIBRARY_PATH") is None:
        os.environ["LD_LIBRARY_PATH"] = wsl_ld_path
    else:
        os.environ["LD_LIBRARY_PATH"] = (
            os.environ["LD_LIBRARY_PATH"] + ":" + wsl_ld_path
        )
    # Restart Python Interpreter
    # https://stackoverflow.com/questions/6543847/setting-ld-library-path-from-inside-python
    os.execv(sys.executable, ["python"] + sys.argv)

# Setup dask for slurm
if "SLURM_JOB_ID" in os.environ:
    # ugly workaraound to not import stuff not available at build-time, but on import.
    from karabo.util.dask import prepare_slurm_nodes_for_dask

    prepare_slurm_nodes_for_dask()

# set rascil data directory environment variable
# see https://ska-telescope.gitlab.io/external/rascil/RASCIL_install.html
from karabo.util.setup_pkg import set_rascil_data_directory_env  # noqa: E402

set_rascil_data_directory_env()
