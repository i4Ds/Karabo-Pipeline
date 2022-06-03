import unittest

from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.simulation.sky_model import get_GLEAM_Sky


class TestData(unittest.TestCase):

    def test_download_gleam(self):
        survey = GLEAMSurveyDownloadObject()
        survey.get()

    def test_download_gleam_and_make_sky_model(self):
        sky = get_GLEAM_Sky()
        sky.plot_sky([250, -30])
