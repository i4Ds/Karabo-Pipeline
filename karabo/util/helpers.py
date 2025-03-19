"""Module for helper utils which don't belong to any other modules."""

from __future__ import annotations

import os
import random
import string
from typing import Literal, Optional, TypeVar, Union, overload

from karabo.util._types import MISSING, MissingType

_EnvType = TypeVar("_EnvType", bound=Union[str, int, float, bool])


def get_rnd_str(k: int, seed: Optional[Union[str, int, float, bytes]] = None) -> str:
    """Creates a random ascii+digits string with length=`k`.

    Most tmp-file tools are using a string-length of 10.

    Args:
        k: Length of random string.
        seed: Seed.

    Returns:
        Random generated string.
    """
    random.seed(seed)
    return "".join(random.choices(string.ascii_letters + string.digits, k=k))


class Environment:
    """Environment variable utility class."""

    @overload
    @classmethod
    def get(
        cls,
        name: str,
        type_: type[_EnvType],
        default: MissingType = MISSING,
        *,
        required: Literal[True] = True,
        allow_none_input: Literal[False] = False,
    ) -> _EnvType:
        ...

    @overload
    @classmethod
    def get(
        cls,
        name: str,
        type_: type[_EnvType],
        default: MissingType = MISSING,
        *,
        required: Literal[True] = True,
        allow_none_input: Literal[True],
    ) -> Optional[_EnvType]:
        ...

    @overload
    @classmethod
    def get(
        cls,
        name: str,
        type_: type[_EnvType],
        default: Optional[MissingType] = MISSING,
        *,
        required: Literal[False],
        allow_none_input: bool = False,
    ) -> Optional[_EnvType]:
        ...

    @overload
    @classmethod
    def get(
        cls,
        name: str,
        type_: type[_EnvType],
        default: _EnvType,
        *,
        required: Literal[False],
        allow_none_input: bool = False,
    ) -> _EnvType:
        ...

    @classmethod
    def get(
        cls,
        name: str,
        type_: type[_EnvType],
        default: Optional[Union[_EnvType, MissingType]] = MISSING,
        *,
        required: bool = True,
        allow_none_input: bool = False,
    ) -> Optional[_EnvType]:  # generics & unions not supported atm
        """Env-var parsing & handling.

        Args:
            name: Name of env-var.
            type_: Type to parse into.
            default: Default value in case env-var is not set. This makes only sense
                if `required=False`. None default values are supported.
            required: Specification if env-var must be set. If it's not available,
                a default value will be returned. If `default` is not specified, `None`
                is the return value.
            allow_none_input: Allow None value parsing.

        Returns:
            Parsed env-var.
        """
        env = os.environ.get(name)
        assert not (required and default is not MISSING)  # signature violation
        if env is None:
            if required:
                if default is MISSING:
                    err_msg = (
                        f"Environment variable {name} is required but not defined."
                    )
                    raise KeyError(err_msg)
                else:
                    err_msg = "Setting a default only makes sense if `required=False`."
                    raise RuntimeError(err_msg)  # overload violation here
            if default is MISSING:
                return None  # type: ignore[return-value]
            else:
                return default  # type: ignore[return-value]
        env_lower = env.lower()
        if allow_none_input and env_lower == "none":
            return None
        if type_ is bool:
            if env_lower == "true":
                return True  # type: ignore[return-value]
            elif env_lower == "false":
                return False  # type: ignore[return-value]
            else:
                err_msg = f"could not convert string to bool: '{env}'"
                raise ValueError(err_msg)
        return type_(env)  # type: ignore[return-value]  # can throw `ValueError`
