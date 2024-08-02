"""This file is executed during build-time and when karabo gets imported.
Hence, you ONLY have deps available here which are available during build-time and
in karabo. If you don't know what that means, don't touch anything here.
"""

import logging
import os
import platform
import sys
from logging import LogRecord
from sysconfig import get_path

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

if "SLURM_JOB_ID" in os.environ:
    # if-statement is an ugly workaround to not import pkgs not available at
    # build/install-time. This is something which is happening if you install the
    # dependencies of Karabo through pip. Then, `versioneer`` determines the current
    # version of Karabo automatically, which is done through this root-init-file.
    # But because this is happening at build/install-time, the dependencies of Karabo
    # are not yet available in the venv, and therefore the installation of the
    # dependencies will fail.
    from karabo.util.dask import DaskHandlerSlurm

    DaskHandlerSlurm._prepare_slurm_nodes_for_dask()


# Avoid the following RASCIL warning:
# The RASCIL data directory is not available - continuing but any simulations will fail
# ...which pops up because we don't download the RASCIL data directory.
# To the best of our knowledge, we don't need the data directory. (31.07.2024)
# We can therefore ignore this warning and avoid unnecessarily alerting users with it.
def filter_message(record: LogRecord) -> bool:
    if (
        record.getMessage()
        == "The RASCIL data directory is not available - continuing but any simulations will fail"  # noqa: E501
    ):
        return False
    else:
        return True


# Install filter on the RASCIL sub-module logger that logs the warning.
# This logger is instantiated with __file__ as its name.
logger_name = os.path.join(
    get_path("platlib"),
    "rascil",
    "processing_components",
    "util",
    "installation_checks.py",
)
logger = logging.getLogger(logger_name)
logger.addFilter(filter_message)
