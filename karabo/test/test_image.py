import unittest
from karabo.util.jupyter import setup_jupyter_env

setup_jupyter_env()
from karabo.Imaging.imager import Imager


class TestImage(unittest.TestCase):
    def testJupyterSetupEnv(self):
        setup_jupyter_env()
        from karabo.Imaging.imager import Imager
        print(Imager)

    def test_dirty_image(self):
        setup_jupyter_env()
        imager = Imager(ingest_msname="./result/sim/test_result.ms", imaging_npixel=2048,
                        imaging_cellsize=3.878509448876288e-05)

        dirty = imager.get_dirty_image()
        dirty.plot_image()

    # removed t from tests to force it to not run on test cases, as this test case takes too long
    def tes_clean(self):
        imager = Imager(ingest_msname='./result/sim/visibilities_gleam.ms',
                        ingest_dd=[0],
                        ingest_vis_nchan=16,
                        ingest_chan_per_blockvis=1,
                        ingest_average_blockvis=True,
                        imaging_npixel=2048,
                        imaging_cellsize=3.878509448876288e-05,
                        imaging_weighting='robust',
                        imaging_robustness=-.5,
                        clean_nmajor=0,
                        clean_algorithm='mmclean',
                        clean_scales=[0, 6, 10, 30, 60],
                        clean_fractional_threshold=.3,
                        clean_threshold=.12e-3,
                        clean_nmoment=5,
                        clean_psf_support=640,
                        clean_restored_output='integrated')
        result = imager.imaging_rascil()
        print(result)
