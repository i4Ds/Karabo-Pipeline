import os
import re
from types import ModuleType
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.special import wofz

import karabo
from karabo.util._types import NPFloatInpBroadType, NPFloatOutBroadType, NPIntFloat


def get_module_absolute_path() -> str:
    path_elements = os.path.abspath(karabo.__file__).split(os.path.sep)
    path_elements.pop()
    return os.path.sep.join(path_elements)


def get_module_path_of_module(module: ModuleType) -> str:
    module_file = cast(str, module.__file__)
    path_elements = os.path.abspath(module_file).split(os.path.sep)
    path_elements.pop()
    return os.path.sep.join(path_elements)

def extract_digit_from_string(string: str) -> int:
    digit = ""
    for char in string:
        if char.isdigit():
            digit += char
    return int(digit)

def extract_chars_from_string(string: str) -> str:
    letters = ""
    for char in string:
        if char.isalpha():
            letters += char
    return letters


def parse_size(size_str: str) -> int:
    size_str = size_str.strip().upper()
    size_units = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}

    pattern = r"^(\d+(?:\.\d+)?)\s*(" + "|".join(size_units.keys()) + ")$"
    match = re.search(pattern, size_str)

    if match:
        value, unit = float(match.group(1)), match.group(2)
        return int(value * size_units[unit])

    raise ValueError(f"Invalid size format: '{size_str}'")


def calculate_required_number_of_chunks(
    max_chunk_size_in_memory: str,
    data_array: List[xr.DataArray],
) -> int:
    max_chunk_size_bytes = parse_size(max_chunk_size_in_memory)
    data_arrays_size = sum([x.nbytes for x in data_array])
    n_chunks = int(np.ceil(data_arrays_size / max_chunk_size_bytes))
    return n_chunks


def calculate_chunk_size_from_max_chunk_size_in_memory(
    max_chunk_memory_size: str, data_array: Union[xr.DataArray, List[xr.DataArray]]
) -> int:
    if not isinstance(data_array, list):
        data_array = [data_array]
    n_chunks = calculate_required_number_of_chunks(max_chunk_memory_size, data_array)
    chunk_size = max(int(data_array[0].shape[0] / n_chunks), 1)
    return chunk_size


def read_CSV_to_ndarray(file: str) -> NDArray[np.float64]:
    import csv

    sources = []
    with open(file, newline="") as sourcefile:
        spamreader = csv.reader(sourcefile, delimiter=",", quotechar="|")
        for row in spamreader:
            if len(row) == 0:
                continue
            if row[0].startswith("#"):
                continue
            else:
                n_row = []
                for cell in row:
                    try:
                        value = float(cell)
                        n_row.append(value)
                    except ValueError:
                        pass
                sources.append(n_row)
    return np.array(sources, dtype=float)


def full_setter(self: object, state: Dict[str, Any]) -> None:
    self.__dict__ = state


def full_getter(self: object) -> Dict[str, Any]:
    state = self.__dict__
    return state


def Gauss(
    x: NPFloatInpBroadType,
    x0: NPFloatInpBroadType,
    y0: NPFloatInpBroadType,
    a: NPFloatInpBroadType,
    sigma: NPFloatInpBroadType,
) -> NPFloatOutBroadType:
    gauss = y0 + a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    return cast(NPFloatOutBroadType, gauss)


def Voigt(
    x: NPFloatInpBroadType,
    x0: NPFloatInpBroadType,
    y0: NPFloatInpBroadType,
    a: NPFloatInpBroadType,
    sigma: NPFloatInpBroadType,
    gamma: NPFloatInpBroadType,
) -> NPFloatOutBroadType:
    # sigma = alpha / np.sqrt(2 * np.log(2))
    voigt = y0 + a * np.real(
        wofz((x - x0 + 1j * gamma) / sigma / np.sqrt(2))
    ) / sigma / np.sqrt(2 * np.pi)
    return cast(NPFloatOutBroadType, voigt)


def get_spectral_sky_data(
    ra: NDArray[np.float_],
    dec: NDArray[np.float_],
    freq0: NDArray[np.float_],
    nfreq: int,
) -> NDArray[np.float_]:
    dfreq_arr = np.linspace(-0.1, 0.1, 100)
    y_voigt = cast(NDArray[NPIntFloat], Voigt(dfreq_arr, 0, 0, 1, 0.01, 0.01))
    # y_gauss = Gauss(dfreq_arr, 0, 0, 1, 0.01)
    dfreq_sample = dfreq_arr[::nfreq]
    flux_sample = y_voigt[::nfreq]
    freq_sample = freq0 + dfreq_sample * freq0
    sky_data = np.zeros((nfreq, 12))
    sky_data[:, 0] = ra
    sky_data[:, 1] = dec
    sky_data[:, 2] = flux_sample
    sky_data[:, 6] = freq_sample
    sky_data[:, 7] = -200
    return sky_data


def resample_spectral_lines(
    npoints: int,
    dfreq: NDArray[np.float_],
    spec_line: NDArray[np.float_],
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    m = int(len(dfreq) / npoints)
    dfreq_sampled = dfreq[::m]
    line_sampled = spec_line[::m]
    return dfreq_sampled, line_sampled


def input_wrapper(
    msg: str,
    ret: str = "y",
) -> str:
    """
    Wrapper of standard `input` to define what return `ret` it will get during
    Unit-tests, since the test just stops oterwise.
    The environment variable 'SKIP_INPUT' or 'UNIT_TEST' must be set
    with an arbitrary value to return `ret`.

    :param msg: input message
    :param ret: return value if 'SKIP_INPUT' or 'UNIT_TEST' is set, default='y'
    """
    if (
        os.environ.get("SKIP_INPUT") is not None
        or os.environ.get("UNIT_TEST") is not None
    ):
        return ret
    else:
        return input(msg)
