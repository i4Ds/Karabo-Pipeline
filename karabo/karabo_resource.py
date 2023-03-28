from types import TracebackType
from typing import Any, Optional

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


class NumpyAssertionsDisabled:
    def __enter__(self) -> None:
        self.old_settings = np.seterr(all="ignore")

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        np.seterr(**self.old_settings)
