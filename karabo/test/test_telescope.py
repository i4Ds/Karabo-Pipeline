import os
import unittest
import karabo.simulation.telescope as telescope


class TestTelescope(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists('result/tel'):
            os.makedirs('result/tel')

    def test_plot(self):
        tel = telescope.Telescope(0, -50)
        tel.plot_telescope('result/tel/simple_tel.png')

    def test_read_tm_file(self):
        tel = telescope.read_OSKAR_tm_file("./karabo/data/telescope.tm")
        tel.plot_telescope('result/tel/oskar_tel.png')
        self.assertEqual(len(tel.stations), 30)

    def test_convert_to_oskar(self):
        tel = telescope.read_OSKAR_tm_file("./karabo/data/telescope.tm")
        oskar_tel = tel.get_OSKAR_telescope()
        self.assertEqual(oskar_tel.get_num_stations(), 30)

    def test_read_alma_file(self):
        tel = telescope.get_ALMA_Telescope()
        tel.plot_telescope('result/tel/alma_tel.png')
        self.assertEqual(len(tel.stations), 43)

    def test_read_meerkat_file(self):
        tel = telescope.get_MEERKAT_Telescope()
        tel.plot_telescope('result/tel/meerkat_tel.png')
        self.assertEqual(len(tel.stations), 64)
