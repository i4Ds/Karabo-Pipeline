import unittest

from karabo.util.jupyter import setup_jupyter_env


class TestImage(unittest.TestCase):
    def testImaging(self):
        pass
        # setup_jupyter_env()
        # from karabo.Imaging.imager import Imager
        # imager = Imager(ingest_msname='./data/visibilities_gleam.ms',
        #                 ingest_dd=[0],
        #                 ingest_vis_nchan=16,
        #                 ingest_chan_per_blockvis=1,
        #                 ingest_average_blockvis=True,
        #                 imaging_npixel=2048,
        #                 imaging_cellsize=3.878509448876288e-05,
        #                 imaging_weighting='robust',
        #                 imaging_robustness=-.5,
        #                 clean_nmajor=0,
        #                 clean_algorithm='mmclean',
        #                 clean_scales=[0, 6, 10, 30, 60],
        #                 clean_fractional_threshold=.3,
        #                 clean_threshold=.12e-3,
        #                 clean_nmoment=5,
        #                 clean_psf_support=640,
        #                 clean_restored_output='integrated')
        #
        # imager.imaging_rascil()
