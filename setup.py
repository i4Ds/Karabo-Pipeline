from distutils.core import setup
from karabo.version import __version__
# implicitly takes config from setup.cfg (used by conda build load_setup_py_data())
setup(
    version=__version__,
)
