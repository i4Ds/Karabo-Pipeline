from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

# numpy dtypes
NPBoolLike: TypeAlias = Union[bool, np.bool_]
NPUIntLike: TypeAlias = Union[
    bool, np.unsignedinteger
]  # not BoolLike because "-" doesn't support np.bool_
NPIntLike: TypeAlias = Union[
    bool, int, np.integer
]  # not BoolLike because "-" doesn't support np.bool_
NPFloatLike: TypeAlias = Union[NPIntLike, float, np.floating]
NPIntFloat: TypeAlias = Union[np.int_, np.float_]
NPIntFloatCompl: TypeAlias = Union[NPIntFloat, np.complex_]
NPComplexLike: TypeAlias = Union[NPFloatLike, complex, np.complexfloating]
NPTD64Like: TypeAlias = Union[NPIntLike, np.timedelta64]
NPNumberLike: TypeAlias = Union[int, float, complex, np.number, np.bool_]
NPScalarLike: TypeAlias = Union[
    int,
    float,
    complex,
    str,
    bytes,
    np.generic,
]
NPFloatInpBroadType: TypeAlias = Union[
    NPFloatLike,
    NDArray[NPIntFloat],
]
NPFloatOutBroadType: TypeAlias = Union[
    NPIntFloat,
    NDArray[NPIntFloat],
]
NPComplInpBroadType: TypeAlias = Union[
    NPComplexLike,
    NDArray[NPIntFloatCompl],
]
NPComplOutBroadType: TypeAlias = Union[
    NPIntFloatCompl,
    NDArray[NPIntFloatCompl],
]

# similar to `numpy dtypes` but without numpy
IntLike: TypeAlias = Union[bool, int]
FloatLike: TypeAlias = Union[IntLike, float]
IntFloat: TypeAlias = Union[int, float]
ComplexLike: TypeAlias = Union[FloatLike, complex]
IntFloatCompl: TypeAlias = Union[IntFloat, complex]
NumberLike: TypeAlias = ComplexLike
ScalarLike: TypeAlias = Union[
    int,
    float,
    complex,
    str,
    bytes,
]

NPIntLikeStrict = Union[np.int_, int]
NPFloatLikeStrict = Union[NPIntFloat, IntFloat]

IntFloatList: TypeAlias = Union[List[int], List[float]]
PrecisionType: TypeAlias = Literal["single", "double"]

OskarSettingsTreeType: TypeAlias = Dict[str, Dict[str, Any]]

# File handling types

# Used for directory paths, to which one can append a file name
DirPathType: TypeAlias = Union[Path, str]
TDirPathType = TypeVar("TDirPathType", bound=DirPathType)
# Used for file paths
FilePathType: TypeAlias = Union[Path, str]
TFilePathType = TypeVar("TFilePathType", bound=FilePathType)


class MissingType:
    ...


MISSING = MissingType()


class BeamType(TypedDict):
    bmaj: float  # major-axis in arcsec
    bmin: float  # minor-axis in arcsec
    bpa: float  # position-angle in deg
