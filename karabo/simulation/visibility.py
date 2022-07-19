import shutil

from karabo.resource import KaraboResource
from karabo.util.FileHandle import FileHandle


class Visibility(KaraboResource):

    def __init__(self):
        self.file = FileHandle(is_dir=True, suffix=".ms")

    def save_to_file(self, path: str) -> None:
        shutil.copytree(self.file.path, path)

    @staticmethod
    def open_from_file(path: str) -> any:
        file = FileHandle(path)
        vis = Visibility()
        vis.file = file
        return vis



