from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.simulation.sky_model import SkyModel


def test_download_gleam():
    survey = GLEAMSurveyDownloadObject()
    survey.get()


def test_download_gleam_and_make_sky_model():
    sky = SkyModel.get_GLEAM_Sky([76])
    assert sky.num_sources > 0
    assert sky.to_np_array().shape == (sky.num_sources, 12)  # No source ID
    assert sky.source_ids["dim_0"].shape[0] == sky.shape[0]  # checking source-ids
