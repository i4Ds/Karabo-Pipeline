import unittest

from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.simulation.sky_model import SkyModel


class TestData(unittest.TestCase):
    def test_download_gleam(self):
        survey = GLEAMSurveyDownloadObject()
        survey.get()

    def test_download_gleam_and_make_sky_model(self):
        sky = SkyModel.get_GLEAM_Sky()
        sky.explore_sky([250, -30], s=0.1)
