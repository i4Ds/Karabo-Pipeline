import pytest

from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.simulation.sky_model import SkyModel


@pytest.mark.test
def test_download_gleam():
    survey = GLEAMSurveyDownloadObject()
    survey.get()


@pytest.mark.test
def test_download_gleam_and_make_sky_model():
    sky = SkyModel.get_GLEAM_Sky([76])
    sky.explore_sky([250, -30], s=0.1)
    assert sky.num_sources > 0
    assert sky.to_array().shape == (sky.num_sources, 12)  # No source ID
    assert sky.shape == (sky.num_sources, 13)  # With source ID
