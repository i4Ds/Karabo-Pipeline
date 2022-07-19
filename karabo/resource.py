

class KaraboResource:

    def save_to_file(self, path: str) -> None:
        """
        Save the specified resource to disk (in format specified by resource itself)
        """
        pass

    @staticmethod
    def open_from_file(path: str) -> any:
        """
        Read the specified resource from disk into Karabo. (format specified by resource itself)
        """
        pass
