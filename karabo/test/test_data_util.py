import unittest

from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.simulation.sky_model import SkyModel
from karabo.util.data_util import parse_size


class TestData(unittest.TestCase):
    def test_download_gleam(self):
        survey = GLEAMSurveyDownloadObject()
        survey.get()

    def test_download_gleam_and_make_sky_model(self):
        sky = SkyModel.get_GLEAM_Sky([76])
        sky.explore_sky([250, -30], s=0.1)
        assert sky.num_sources > 0
        assert sky.to_array().shape == (sky.num_sources, 12)  # No source ID
        assert sky.shape == (sky.num_sources, 13)  # With source ID


class TestParseSize(unittest.TestCase):
    def test_valid_sizes(self):
        test_cases = [
            ("1B", 1),
            ("10B", 10),
            ("1 KB", 1000),
            ("1.5 KB", 1500),
            ("2MB", 2 * 10**6),
            ("2 GB", 2 * 10**9),
            ("1.25 TB", 1.25 * 10**12),
            ("10.5GB", 10.5 * 10**9),
        ]
        test_cases = [("1.5 KB", 1500)]

        for size_str, expected in test_cases:
            with self.subTest(size_str=size_str, expected=expected):
                self.assertEqual(parse_size(size_str), expected)

    def test_invalid_sizes(self):
        invalid_sizes = [
            "10",  # no unit
            "1.5 XB",  # invalid unit
            "1 MB K",  # extra character
            "1.2.3 GB",  # invalid number format
        ]

        for size_str in invalid_sizes:
            with self.subTest(size_str=size_str):
                self.assertRaises(ValueError, parse_size, size_str)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
