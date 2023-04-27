# set shared library if WSL to detect GPU drivers
import os
import platform
import sys

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
from karabo.util.dask import prepare_slurm_nodes_for_dask

prepare_slurm_nodes_for_dask()

from karabo.util.data_util import get_module_absolute_path  # noqa: E402

# set rascil data directory environment variable
# see https://ska-telescope.gitlab.io/external/rascil/RASCIL_install.html
from karabo.util.jupyter import set_rascil_data_directory_env  # noqa: E402

set_rascil_data_directory_env()

# Set version. If dev version, use _version.txt, otherwise use package_version.txt
if os.path.exists(get_module_absolute_path() + "/_package_version.txt"):
    with open(get_module_absolute_path() + "/_package_version.txt", "r") as f:
        version = f.read()
else:
    with open(get_module_absolute_path() + "/_version.txt", "r") as f:
        version = f.read()

__version__ = version.strip()


if __name__ == "__main__":
    import karabo

    print(karabo.__version__)
