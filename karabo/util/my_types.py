from typing import Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

NPUnk: TypeAlias = Union[
    float,
    int,
    np.float64,
    np.int64,
    NDArray[np.float64],
    NDArray[np.int64],
]
