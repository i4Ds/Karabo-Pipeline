import shutil

from karabo.util.FileHandle import FileHandle


class Visibility:

    def __init__(self):
        self.file = FileHandle(is_dir=True)
        self.path = self.file.path + ".MS"

    def load_ms_file(self, filepath: str) -> None:
        self.path = filepath
        self.file = FileHandle(self.path)

    def save_to_ms(self, directory_path: str) -> None:
        if self.path is not None:
            shutil.copytree(self.path, directory_path)