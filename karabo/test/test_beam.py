import unittest

from karabo.simulation.telescope import Telescope
from karabo.simulation.beam import BeamPattern, PolType
from karabo.test import data_path


class MyTestCase(unittest.TestCase):
    def test_fit_element(self):
        tel = Telescope.get_MEERKAT_Telescope()
        beam = BeamPattern(f"{data_path}/run5.cst")
        beam.fit_elements(tel, freq_hz=1.0e+08, avg_frac_error=0.5)


if __name__ == '__main__':
    unittest.main()
