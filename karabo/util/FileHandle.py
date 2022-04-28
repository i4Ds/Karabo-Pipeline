import os
import tempfile


class FileHandle:

    def __init__(self, existing_file_path: str = None, is_dir: bool = False, auto_clean: bool = True):
        self.auto_clean = auto_clean
        if is_dir:
            self.file = tempfile.TemporaryDirectory()
            self.path = self.file.name

        if existing_file_path is not None:
            self.file = open(existing_file_path)
            self.path = existing_file_path
        else:
            self.file = tempfile.NamedTemporaryFile()
            self.path = self.file.name

    def __del__(self):
        if os.path.exists(self.path) and self.auto_clean:
            os.remove(self.path)
