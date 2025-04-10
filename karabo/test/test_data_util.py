import numpy as np
import pytest
import xarray as xr

from karabo.util.data_util import (
    calculate_chunk_size_from_max_chunk_size_in_memory,
    parse_size,
)


@pytest.mark.parametrize(
    ("size_str", "expected"),
    [
        ("1B", 1),
        ("10B", 10),
        ("1 KB", 1000),
        ("1.5 KB", 1500),
        ("2MB", 2 * 10**6),
        ("2 GB", 2 * 10**9),
        ("1.25 TB", 1.25 * 10**12),
        ("10.5GB", 10.5 * 10**9),
    ],
)
def test_valid_sizes(size_str: str, expected: float):
    assert parse_size(size_str) == int(expected)


@pytest.mark.parametrize(
    ("size_str",),
    [
        ("10",),
        ("1.5 XB",),
        ("1 MB K",),
        ("1.2.3 GB",),
    ],
)
def test_invalid_sizes(size_str):
    with pytest.raises(ValueError) as ve:  # noqa: F841
        parse_size(size_str)


def test_calculate_chunk_size_from_max_chunk_size_in_memory():
    # Create test data array
    data_array = xr.DataArray(np.random.rand(1000, 1000), dims=("x", "y"))

    # 1: max_chunk_memory_size is larger than the total size of the data_array
    max_chunk_memory_size = "10MB"
    expected_chunk_size = 1000
    calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
        max_chunk_memory_size, data_array
    )
    assert expected_chunk_size == calculated_chunk_size

    # 2: max_chunk_memory_size is smaller than the total size of the data_array
    max_chunk_memory_size = "1MB"
    expected_chunk_size = 125
    calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
        max_chunk_memory_size, data_array
    )
    assert expected_chunk_size == calculated_chunk_size

    # 3: max_chunk_memory_size equals the total size of the data_array
    max_chunk_memory_size = "8MB"
    expected_chunk_size = 1000
    calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
        max_chunk_memory_size, data_array
    )
    assert expected_chunk_size == calculated_chunk_size

    # 4: max_chunk_memory_size is smaller than one row of the data_array
    max_chunk_memory_size = "1KB"
    expected_chunk_size = 1
    calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
        max_chunk_memory_size, data_array
    )
    assert expected_chunk_size == calculated_chunk_size

    # 5: max_chunk_memory_size is larger than one row of the list of data_arrays
    data_arrays = [data_array, data_array]
    max_chunk_memory_size = "10MB"
    expected_chunk_size = 500
    calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
        max_chunk_memory_size, data_arrays
    )
    assert expected_chunk_size == calculated_chunk_size

    # 6: Only one row of data
    data_array = xr.DataArray(np.random.rand(1, 1000), dims=("x", "y"))
    max_chunk_memory_size = "10MB"
    expected_chunk_size = 1
    calculated_chunk_size = calculate_chunk_size_from_max_chunk_size_in_memory(
        max_chunk_memory_size, data_array
    )
    assert expected_chunk_size == calculated_chunk_size
