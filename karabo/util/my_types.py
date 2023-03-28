from typing import Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

NPInp: TypeAlias = Union[
    int,
    float,
    NDArray[np.float64],
    NDArray[np.int64],
]

NPOutp: TypeAlias = Union[
    NDArray[np.float64],
    NDArray[np.int64],
]
