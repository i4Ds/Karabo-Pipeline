# set shared library if WSL to detect GPU drivers
import os, platform, sys

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
    os.execv(sys.argv[0], sys.argv)


# set rascil data directory environment variable (see https://ska-telescope.gitlab.io/external/rascil/RASCIL_install.html)
from karabo.util.jupyter import set_rascil_data_directory_env
from karabo.util.data_util import get_module_absolute_path
import sys

set_rascil_data_directory_env()

# Set version. If dev version, use _version.txt, otherwise use package_version.txt
if os.path.exists(get_module_absolute_path()+'/package_version.txt'):
    with open(get_module_absolute_path()+'/package_version.txt', 'r') as f:
        version = f.read()
else:
    with open(get_module_absolute_path()+'/_version.txt', 'r') as f:
        version = f.read()
        
__version__ = version.strip()


if __name__ == '__main__':
    import karabo
    print(karabo.__version__)
    