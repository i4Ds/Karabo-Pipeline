import os
from types import ModuleType
from typing import Any, Dict, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import wofz

import karabo
from karabo.util.my_types import NPBroadcType


def get_module_absolute_path() -> str:
    path_elements = os.path.abspath(karabo.__file__).split(os.path.sep)
    path_elements.pop()
    return os.path.sep.join(path_elements)


def get_module_path_of_module(module: ModuleType) -> str:
    module_file = cast(str, module.__file__)
    path_elements = os.path.abspath(module_file).split(os.path.sep)
    path_elements.pop()
    return os.path.sep.join(path_elements)


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
    x: NPBroadcType,
    x0: NPBroadcType,
    y0: NPBroadcType,
    a: NPBroadcType,
    sigma: NPBroadcType,
) -> NPBroadcType:
    gauss = y0 + a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))  # type: ignore
    return cast(NPBroadcType, gauss)


def Voigt(
    x: NPBroadcType,
    x0: NPBroadcType,
    y0: NPBroadcType,
    a: NPBroadcType,
    sigma: NPBroadcType,
    gamma: NPBroadcType,
) -> NPBroadcType:
    # sigma = alpha / np.sqrt(2 * np.log(2))
    voigt = y0 + a * np.real(
        wofz((x - x0 + 1j * gamma) / sigma / np.sqrt(2))
    ) / sigma / np.sqrt(2 * np.pi)
    return cast(NPBroadcType, voigt)


def get_spectral_sky_data(
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    freq0: NDArray[np.float64],
    nfreq: int,
) -> NDArray[np.float64]:
    dfreq_arr = np.linspace(-0.1, 0.1, 100)
    y_voigt = Voigt(dfreq_arr, 0, 0, 1, 0.01, 0.01)
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
    dfreq: NDArray[np.float64],
    spec_line: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
