import os
from distutils.core import setup

__version__ = {}
with open(os.path.join("karabo", "_version.txt")) as version_file:
    __version__ = version_file.read().strip()
    version_file.close()

if os.getenv("NIGHTLY_BUILD", "false") == "true":
    if "dev" in __version__:
        # Increment the dev version number
        dev_version = int(__version__.split("dev")[-1])
        __version__ = __version__.split("dev")[0] + "dev" + str(dev_version + 1)
    else:
        __version__ = __version__ + ".dev0"

if not os.path.exists(os.path.join("karabo", "_package_version.txt")):
    with open(os.path.join("karabo", "_package_version.txt"), "w") as version_file:
        version_file.write(__version__)
        version_file.close()

# implicitly takes config from setup.cfg (used by conda build load_setup_py_data())
setup(
    version=__version__,
)
