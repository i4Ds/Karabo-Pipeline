from typing import Any

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
    def __enter__(self):
        self.old_settings = np.seterr(all="ignore")

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.seterr(**self.old_settings)
