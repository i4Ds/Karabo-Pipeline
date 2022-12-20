import unittest
from karabo.util.data_util import get_module_absolute_path

def run_tests(verbosity=0, *args, **kwargs):
    loader = unittest.TestLoader()
    
    # Get location of karabo package
    start_dir = get_module_absolute_path()
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner(verbosity=verbosity, *args, **kwargs)
    runner.run(suite)
        