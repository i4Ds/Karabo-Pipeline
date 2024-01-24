from __future__ import annotations

import os
import sys
from types import TracebackType
from typing import Any, Literal, Optional, TextIO

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
    """Captures numpy-errors."""

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
    """Captures `sys.stdout` and/or `sys.stderr` to silent ouput."""

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


class CaptureSpam:
    """Captures spam of `print` to `sys.stdout` or `sys.stderr`.

    Captures exact-match spam-messages of an external library.
    It checks each new line (not an entire print-message with
     multiple multiple newlines), if it has already been printed once.
    Don't use CaptureSpam if the external library function
     provides a way to suppress their output.

    Example usage:
        ```
        with CaptureSpam():
            library_spam_fun()
        ```
    """

    def __init__(self, stream: TextIO = sys.stdout):
        self._buf = ""
        self._stream = stream
        self._captured: list[str] = []

    def write(self, buf: str) -> None:
        while buf:
            try:
                newline_index = buf.index("\n")
            except ValueError:
                # no newline, buffer for next call
                self._buf += buf
                break
            # get data up to next newline and combine with any buffered data
            self._buf = self._buf + buf[: newline_index + 1]
            buf = buf[newline_index + 1 :]

            self.flush()

    def flush(self) -> None:
        if self._buf not in self._captured:
            self._captured.append(self._buf)
            self._stream.write(self._buf)
        self._buf = ""

    def __enter__(self) -> CaptureSpam:
        if self._stream == sys.stdout:
            sys.stdout = self  # type: ignore[assignment]
            self._std = "stdout"
        elif self._stream == sys.stderr:
            sys.stderr = self  # type: ignore[assignment]
            self._std = "stderr"
        else:
            raise ValueError(
                "CaptureSpam only supports `sys.stdout` and `sys.stderr`, "
                f"but got {self._stream=}."
            )
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._std == "stdout":
            sys.stdout = self._stream
        elif self._std == "stderr":
            sys.stderr = self._stream
