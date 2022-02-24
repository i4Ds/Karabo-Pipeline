import unittest
import karabo.simulation.telescope as telescope


class TestTelescope(unittest.TestCase):

    def test_plot(self):
        tel = telescope.Telescope(0, -50)
        tel.plot_telescope()

    def test_read_tm_file(self):
        tel = telescope.read_OSKAR_tm_file("../karabo/data/telescope.tm")
        tel.plot_telescope()

    def test_convert_to_oskar(self):
        tel = telescope.read_OSKAR_tm_file("../karabo/data/telescope.tm")
        oskar_tel = tel.get_OSKAR_telescope()
        print(oskar_tel)
