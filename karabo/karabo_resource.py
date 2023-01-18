from typing import Any

class KaraboResource:

    def copy_image_file_to(self, path: str) -> None:
        """
        Save the specified resource to disk (in format specified by resource itself)
        """
        raise NotImplementedError()

    @staticmethod
    def read_from_file(path: str) -> Any:
        """
        Read the specified resource from disk into Karabo. (format specified by resource itself)
        """
        raise NotImplementedError()
