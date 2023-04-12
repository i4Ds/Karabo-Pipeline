from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import TypeAlias

# numpy dtypes
NPBoolLike = Union[bool, np.bool_]
NPUIntLike = Union[
    bool, np.unsignedinteger
]  # not BoolLike because "-" doesn't support np.bool_
NPIntLike = Union[
    bool, int, np.integer
]  # not BoolLike because "-" doesn't support np.bool_
NPFloatLike = Union[NPIntLike, float, np.floating]
NPComplexLike = Union[NPFloatLike, complex, np.complexfloating]
NPTD64Like = Union[NPIntLike, np.timedelta64]
NPNumberLike = Union[int, float, complex, np.number, np.bool_]
NPScalarLike = Union[
    int,
    float,
    complex,
    str,
    bytes,
    np.generic,
]
NPBroadcType: TypeAlias = Union[
    NPFloatLike,
    ArrayLike,
]

# similar to `numpy dtypes` but without numpy
IntLike = Union[bool, int]
FloatLike = Union[IntLike, float]
ComplexLike = Union[FloatLike, complex]
NumberLike = ComplexLike
ScalarLike = Union[
    int,
    float,
    complex,
    str,
    bytes,
]
