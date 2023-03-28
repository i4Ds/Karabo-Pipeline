import os
import shutil
import uuid
from typing import Optional


class FileHandle:
    path: str
    __temp_path = os.path.join(os.getcwd(), "karabo_folder" + os.path.sep)

    def __init__(
        self,
        existing_file_path: Optional[str] = None,
        is_dir: bool = False,
        suffix: str = "",
    ) -> None:
        tmp_path = os.path.abspath(self.__temp_path + str(uuid.uuid4()) + suffix)

        # make temp folder if not present
        if not os.path.exists(self.__temp_path):
            os.mkdir(self.__temp_path)

        if existing_file_path:
            # existing
            if is_dir:
                # is a directory
                self.path = tmp_path
                shutil.copytree(existing_file_path, tmp_path)
            else:
                # is a file
                open(tmp_path, "x")
                shutil.copyfile(existing_file_path, tmp_path)
                self.path = tmp_path

        else:
            # not existing
            if is_dir:
                # is a directory
                self.path = tmp_path
                os.mkdir(tmp_path)
            else:
                # is a file
                open(tmp_path, "x")
                self.path = tmp_path

    def clean_up(self) -> None:
        # remove temp folder and everything inside it
        shutil.rmtree(self.__temp_path)

    def remove_file(self, file_path: str) -> None:
        os.remove(file_path)


def check_ending(path: str, ending: str) -> None:
    if not path.endswith(ending):
        fname = path.split(os.path.sep)[-1]
        raise ValueError(
            f"Invalid file-ending, file {fname} must have {ending} extension!"
        )
