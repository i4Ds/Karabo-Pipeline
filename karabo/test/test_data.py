from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.simulation.sky_model import SkyModel, SkyPrefixMapping


def test_download_gleam():
    survey = GLEAMSurveyDownloadObject()
    survey.get()


def test_download_gleam_and_make_sky_model():
    sky = SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=80e6)
    sample_prefix_mapping = SkyPrefixMapping([], [], [])
    number_of_sky_attributes = len(sample_prefix_mapping.__dict__)

    assert sky.num_sources > 0

    # -1 since we do not return the source ID
    assert sky.to_np_array().shape == (sky.num_sources, number_of_sky_attributes - 1)
    assert sky.source_ids["dim_0"].shape[0] == sky.shape[0]  # checking source-ids
