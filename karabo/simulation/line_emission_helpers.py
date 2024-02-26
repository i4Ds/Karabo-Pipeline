from typing import TypeVar, cast

import numpy as np
import xarray as xr
from astropy.constants import c
from numpy.typing import NDArray

from karabo.util._types import IntFloat

_T = TypeVar("_T", NDArray[np.float_], xr.DataArray, IntFloat)


def convert_z_to_frequency(z: _T) -> _T:
    """Turn given redshift into corresponding frequency (Hz) for 21cm emission.

    :param z: Redshift values to be converted into frequencies.

    :return: Frequencies corresponding to input redshifts.
    """

    return cast(_T, c.value / (0.21 * (1 + z)))


def convert_frequency_to_z(freq: _T) -> _T:
    """Turn given frequency (Hz) into corresponding redshift for 21cm emission.

    :param freq: Frequency values to be converted into redshifts.

    :return: Redshifts corresponding to input frequencies.
    """

    return cast(_T, (c.value / (0.21 * freq)) - 1)
