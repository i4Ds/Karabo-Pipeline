import unittest

import pandas as pd

from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.simulation.sky_model import SkyModel


class TestData(unittest.TestCase):
    def test_download_gleam(self):
        survey = GLEAMSurveyDownloadObject()
        survey.get()

    def test_download_gleam_and_make_sky_model(self):
        sky = SkyModel.get_GLEAM_Sky([76])
        sky.explore_sky([250, -30], s=0.1)
        assert sky.num_sources > 0
        assert len(sky.sources.source_name) == sky.num_sources

    def test_transform_dataframe_to_xarray(self):
        df = pd.DataFrame()
        df[0] = [1.0, 2.0, 3.0, 4.0, 5.0]
        df[1] = [1, 2, 3, 4, 5]
        df[2] = [1, 2, 3, 4, 5]
        df[3] = [1, 2, 3, 4, 5]
        df[4] = [1, 2, 3, 4, 5]
        df[5] = [1, 2, 3, 4, 5]
        df[6] = [1, 2, 3, 4, 5]
        df[7] = [1, 2, 3, 4, 5]
        df[8] = [1, 2, 3, 4, 5]
        df[9] = [1, 2, 3, 4, 5]
        df[10] = [1, 2, 3, 4, 5]
        df[11] = [1, 2, 3, 4, 5]
        df[12] = ["source1", "source2", "source3", "source4", "source5"]

        sky = SkyModel(df)
        assert sky.num_sources > 0
        assert sky.to_array().shape == (sky.num_sources, 12)  # No source ID
        assert len(sky.sources.source_name) == sky.num_sources
        assert all(sky.sources.source_name == df[12].values)

    def test_transform_numpy_to_xarray(self):
        df = pd.DataFrame()
        df[0] = [1.0, 2.0, 3.0, 4.0, 5.0]
        df[1] = [1, 2, 3, 4, 5]
        df[2] = [1, 2, 3, 4, 5]
        df[3] = [1, 2, 3, 4, 5]
        df[4] = [1, 2, 3, 4, 5]
        df[5] = [1, 2, 3, 4, 5]
        df[6] = [1, 2, 3, 4, 5]
        df[7] = [1, 2, 3, 4, 5]
        df[8] = [1, 2, 3, 4, 5]
        df[9] = [1, 2, 3, 4, 5]
        df[10] = [1, 2, 3, 4, 5]
        df[11] = [1, 2, 3, 4, 5]
        df[12] = ["source1", "source2", "source3", "source4", "source5"]

        sky = SkyModel(df.to_numpy())
        assert sky.num_sources > 0
        assert sky.to_array().shape == (sky.num_sources, 12)  # No source ID
        assert len(sky.sources.source_name) == sky.num_sources
        assert all(sky.sources.source_name == df[12].values)
