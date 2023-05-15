import unittest

import numpy as np
import xarray as xr

from karabo.data.external_data import GLEAMSurveyDownloadObject
from karabo.util.data_util import (
    calculate_chunk_size_from_max_chunk_size_in_memory,
    parse_size,
)


class TestData(unittest.TestCase):
    def test_download_gleam(self):
        survey = GLEAMSurveyDownloadObject()
        survey.get()


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


class TestCalculateChunkSize(unittest.TestCase):
    def test_calculate_chunk_size_from_max_chunk_size_in_memory(self):
        # Create test data array
        data_array = xr.DataArray(np.random.rand(1000, 1000), dims=("x", "y"))

        # 1: max_chunk_memory_size is larger than the total size of the data_array
        max_chunk_memory_size = "10MB"
        expected_chunk_size = 1000
        calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
            max_chunk_memory_size, data_array
        )
        self.assertEqual(expected_chunk_size, calculated_chunk_size)

        # 2: max_chunk_memory_size is smaller than the total size of the data_array
        max_chunk_memory_size = "1MB"
        expected_chunk_size = 125
        calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
            max_chunk_memory_size, data_array
        )
        self.assertEqual(expected_chunk_size, calculated_chunk_size)

        # 3: max_chunk_memory_size equals the total size of the data_array
        max_chunk_memory_size = "8MB"
        expected_chunk_size = 1000
        calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
            max_chunk_memory_size, data_array
        )
        self.assertEqual(expected_chunk_size, calculated_chunk_size)

        # 4: max_chunk_memory_size is smaller than one row of the data_array
        max_chunk_memory_size = "1KB"
        expected_chunk_size = 1
        calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
            max_chunk_memory_size, data_array
        )
        self.assertEqual(expected_chunk_size, calculated_chunk_size)

        # 5: max_chunk_memory_size is larger than one row of the list of data_arrays
        data_arrays = [data_array, data_array]
        max_chunk_memory_size = "10MB"
        expected_chunk_size = 500
        calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
            max_chunk_memory_size, data_arrays
        )
        self.assertEqual(expected_chunk_size, calculated_chunk_size)

        # 6: Only one row of data
        data_array = xr.DataArray(np.random.rand(1, 1000), dims=("x", "y"))
        max_chunk_memory_size = "10MB"
        expected_chunk_size = 1
        calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
            max_chunk_memory_size, data_array
        )
        self.assertEqual(expected_chunk_size, calculated_chunk_size)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
