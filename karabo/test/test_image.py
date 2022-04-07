import unittest

from karabo.util.jupyter import setup_jupyter_env


class TestImage(unittest.TestCase):
    def testJupyterSetupEnv(self):
        setup_jupyter_env()
        from karabo.Imaging.imager import Imager
        print(Imager)
