import os
import unittest

# from karabo.Imaging.source_detection import  use_dao_star_finder
from karabo.simulation.Visibility import Visibility
from karabo.Imaging.imager import Imager
from karabo.simulation.sky_model import get_GLEAM_Sky
from karabo.test import data_path


class TestImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists('result/'):
            os.makedirs('result/')

    def testJupyterSetupEnv(self):
        from karabo.Imaging.imager import Imager
        print(Imager)

    def test_dirty_image(self):
        vis = Visibility()
        vis.load_ms_file(f"{data_path}/visibilities_gleam.ms")
        imager = Imager(vis, imaging_npixel=2048,
                        imaging_cellsize=3.878509448876288e-05)

        dirty = imager.get_dirty_image()
        dirty.save_as_fits("result/dirty.fits")
        dirty.plot()

    def test_explore_sky(self):
        sky = get_GLEAM_Sky()
        sky.explore_sky([250, -80])

    # TODO: move these on to CSCS Test Infrastructure once we have it.
    # # removed t from tests to force it to not run on test cases, as this test case takes too long
    # def test_clean(self):
    #     visibility = Visibility()
    #     visibility.load_ms_file("./data/visibilities_gleam.ms")
    #     imager = Imager(visibility,
    #                     ingest_dd=[0],
    #                     ingest_vis_nchan=16,
    #                     ingest_chan_per_blockvis=1,
    #                     ingest_average_blockvis=True,
    #                     imaging_npixel=2048,
    #                     imaging_cellsize=3.878509448876288e-05,
    #                     imaging_weighting='robust',
    #                     imaging_robustness=-.5)
    #     result = imager.imaging_rascil()
    #     print(result)

    # def test_source_detection(self):
    #     restored = open_fits_image("karabo/test/data/restored.fits")
    #     detection_result = detect_sources_in_image(restored)
    #     residual = detection_result.get_gaussian_residual_image()
    #     residual.save_as_fits("result/gaus_residual.fits")
    #     residual.plot()
    #     detection_result.detection.show_fit()

    # def test_source_detection_on_residual(self):
    #     residual = open_fits_image("./data/residual.fits")
    #     sources = detect_sources_in_image(residual, beam=(0.06, 0.02, 13.3))
    #     print(sources.sources)

    # def test_source_detection_on_dirty_image(self):
    #     dirty = open_fits_image("./data/dirty.fits")
    #     dirty.plot()
    #     detection = detect_sources_in_image(dirty, beam=(0.06, 0.02, 13.3))
    #     detection.get_island_mask().plot()

    # def test_show_deconvolved(self):
    #     deconv = open_fits_image("./data/deconvolved.fits")
    #     deconv.plot()

    # def test_try_star_finder(self):
    #     deconvolved = open_fits_image("./data/deconvolved.fits")
    #     use_dao_star_finder(deconvolved)
    #
    # def test_source_detection_on_deconvolved(self):
    #     deconvolved = open_fits_image("./data/deconvolved.fits")
    #     sources = detect_sources_in_image(deconvolved,
    #                                       # beam=(0.05425680530067645, 0.05047422298502832, -205.3215315790252))
    #                                       beam=(0.06, 0.02, 13.3))
    #     print(sources.sources)