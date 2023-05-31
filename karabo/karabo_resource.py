import os
import sys
from types import TracebackType
from typing import Any, Literal, Optional

import numpy as np


class KaraboResource:
    def write_to_file(self, path: str) -> None:
        """
        Save the specified resource to disk (in format specified by resource itself)
        """
        raise NotImplementedError()

    @staticmethod
    def read_from_file(path: str) -> Any:
        """
        Read the specified resource from disk into Karabo.
        (format specified by resource itself)
        """
        raise NotImplementedError()


ErrKind = Literal["ignore", "warn", "raise", "call", "print", "log"]


class NumpyHandleError:
    def __init__(
        self,
        all: Optional[ErrKind] = None,
        divide: Optional[ErrKind] = None,
        over: Optional[ErrKind] = None,
        under: Optional[ErrKind] = None,
        invalid: Optional[ErrKind] = None,
    ) -> None:
        self.all = all
        self.divide = divide
        self.over = over
        self.under = under
        self.invalid = invalid

    def __enter__(self) -> None:
        self._old_settings = np.seterr(
            all=self.all,
            divide=self.divide,
            over=self.over,
            under=self.under,
            invalid=self.invalid,
        )

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        np.seterr(**self._old_settings)


class HiddenPrints:
    def __init__(
        self,
        stdout: bool = True,
        stderr: bool = True,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr

    def __enter__(self) -> None:
        if self.stdout:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        if self.stderr:
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self.stdout:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        if self.stderr:
            sys.stderr.close()
            sys.stderr = self._original_stderr
