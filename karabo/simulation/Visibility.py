import shutil


class Visibility:

    def __init__(self):
        self.path = ""

    def use_ms_file(self, filepath: str) -> None:
        self.path = filepath

    def save_to_ms(self, filepath: str) -> None:
        shutil.copyfile(self.path, filepath)
