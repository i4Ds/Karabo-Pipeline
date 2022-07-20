import os
import unittest
from karabo.simulation.telescope import Telescope
from karabo.simulation.telescope_versions import ALMAVersions, ACAVersions, CARMAVersions, NGVLAVersions, \
    PDBIVersions, SMAVersions, VLAVersions


class TestTelescope(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists('result/tel'):
            os.makedirs('result/tel')

    def test_plot(self):
        tel = Telescope(0, -50)
        tel.plot_telescope('result/tel/simple_tel.png')

    def test_read_tm_file(self):
        tel = Telescope.get_OSKAR_Example_Telescope()
        tel.plot_telescope('result/tel/oskar_tel.png')
        self.assertEqual(len(tel.stations), 30)

    def test_convert_to_oskar(self):
        tel = Telescope.get_OSKAR_Example_Telescope()
        oskar_tel = tel.get_OSKAR_telescope()
        self.assertEqual(oskar_tel.get_num_stations(), 30)

    def test_read_alma_file(self):
        tel = Telescope.get_ALMA_Telescope(ALMAVersions.CYCLE_1_1)
        tel.plot_telescope('result/tel/alma_tel.png')
        self.assertEqual(len(tel.stations), 32)

    def test_read_meerkat_file(self):
        tel = Telescope.get_MEERKAT_Telescope()
        tel.plot_telescope('result/tel/meerkat_tel.png')
        self.assertEqual(len(tel.stations), 64)

    def test_read_all_ALMA_versions(self):
        for version in ALMAVersions:
            tel = Telescope.get_ALMA_Telescope(version)
            tel.plot_telescope(f'result/tel/{version}.png')

    def test_read_all_ACA_versions(self):
        for version in ACAVersions:
            tel = Telescope.get_ACA_Telescope(version)
            tel.plot_telescope(f'result/tel/{version}.png')

    def test_read_all_CARMA_versions(self):
        for version in CARMAVersions:
            tel = Telescope.get_CARMA_Telescope(version)
            tel.plot_telescope(f'result/tel/{version}.png')

    def test_read_all_NG_VLA_versions(self):
        for version in NGVLAVersions:
            tel = Telescope.get_NG_VLA_Telescope(version)
            tel.plot_telescope(f'result/tel/{version}.png')

    def test_read_all_PDBI_versions(self):
        for version in PDBIVersions:
            tel = Telescope.get_PDBI_Telescope(version)
            tel.plot_telescope(f'result/tel/{version}.png')

    def test_read_all_SMA_versions(self):
        for version in SMAVersions:
            tel = Telescope.get_SMA_Telescope(version)
            tel.plot_telescope(f'result/tel/{version}.png')

    def rest_read_all_VLA_versions(self):
        for version in VLAVersions:
            tel = Telescope.get_VLA_Telescope(version)
            tel.plot_telescope(f'result/tel/{version}.png')

    def test_read_SKA_LOW(self):
        tel = Telescope.get_SKA1_LOW_Telescope()
        tel.plot_telescope('result/tel/ska_low.png')

    def test_read_SKA_MID(self):
        tel = Telescope.get_SKA1_LOW_Telescope()
        tel.plot_telescope('result/tel/ska_mid.png')

    def test_read_VLBA(self):
        tel = Telescope.get_VLBA_Telescope()
        tel.plot_telescope('result/tel/vlba.png')

    def test_read_WSRT(self):
        tel = Telescope.get_WSRT_Telescope()
        tel.plot_telescope('result/tel/wsrt.png')

    def test_read_LOFAR(self):
        tel = Telescope.get_LOFAR_Telescope()
        tel.plot_telescope('result/tel/lofar.png')

    def test_read_MKATPLUS(self):
        tel = Telescope.get_MKATPLUS_Telescope()
        tel.plot_telescope('result/tel/mkatplus.png')

    def test_read_ASKAP(self):
        tel = Telescope.get_ASKAP_Telescope()
        tel.plot_telescope('result/tel/ASKAP.png')

    def test_tm_file_creation(self):
        tel = Telescope.get_OSKAR_Example_Telescope()
        tel.get_OSKAR_telescope()
