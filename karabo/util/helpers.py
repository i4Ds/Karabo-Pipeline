"""Module for helper utils which don't belong to any other modules."""

from __future__ import annotations

import os
import random
import string
from typing import Literal, Optional, TypeVar, Union, overload

from karabo.util._types import MISSING, MissingType

_TEnv = TypeVar("_TEnv", bound=Union[str, int, float, bool])


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
        type_: type[_TEnv],
        default: Union[MissingType, _TEnv] = MISSING,
        *,
        allow_none_parsing: Literal[False] = False,
    ) -> _TEnv:
        ...

    @overload
    @classmethod
    def get(
        cls,
        name: str,
        type_: type[_TEnv],
        default: Optional[Union[_TEnv, MissingType]] = ...,
        *,
        allow_none_parsing: Literal[True],
    ) -> Optional[_TEnv]:
        ...

    @overload
    @classmethod
    def get(
        cls,
        name: str,
        type_: type[_TEnv],
        default: Literal[None],
        *,
        allow_none_parsing: bool = ...,
    ) -> Optional[_TEnv]:
        ...

    @classmethod
    def get(
        cls,
        name: str,
        type_: type[_TEnv],
        default: Optional[Union[_TEnv, MissingType]] = MISSING,
        *,
        allow_none_parsing: bool = False,
    ) -> Optional[_TEnv]:  # generics & unions not supported atm
        """Env-var parsing & handling.

        Args:
            name: Name of env-var.
            type_: Type to parse into.
            default: Default value in case env-var is not set
                (NoneType allowed as default).
            allow_none_parsing: Allow None-type parsing?

        Returns:
            Parsed env-var.
        """
        env = os.environ.get(name)
        if env is None:
            if default is MISSING:
                err_msg = f"Environment variable '{name}' not found!"
                raise KeyError(err_msg)
            else:
                return default  # type: ignore[return-value]
        env_lower = env.lower()
        if allow_none_parsing and env_lower == "none":
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
