import unittest

def run_tests(verbosity=0, *args, **kwargs):
    loader = unittest.TestLoader()
    start_dir = 'test'
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner(verbosity=verbosity, *args, **kwargs)
    runner.run(suite)
        
if __name__ == '__main__':
    run_tests()