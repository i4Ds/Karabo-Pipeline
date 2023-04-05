import os
import shutil
import uuid
from typing import Optional


class FileHandle:
    def __init__(
        self,
        dir: Optional[str] = None,
        file_name: Optional[str] = None,
        suffix: str = "",
    ) -> None:
        if dir:
            base_path = os.path.abspath(dir)
        else:
            base_path = os.path.join(os.getcwd(), "karabo_folder")

        # Make the base path if it does not exist
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        # If the folder is a visibility or measurement set, use it as the base path
        if FileHandle.__folder_is_vis_or_ms(base_path):
            self.path = base_path
        else:
            if file_name:
                self.path = os.path.join(
                    base_path, str(uuid.uuid4()), file_name + suffix
                )
            else:
                self.path = os.path.join(base_path, str(uuid.uuid4()) + suffix)

        # Make sure everything, except the file, exists
        if file_name:
            print(self.path)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        else:
            os.makedirs(self.path, exist_ok=True)

        # Create the file if it does not exist
        if file_name:
            print(f"Creating file {self.path}")
            open(self.path, "a").close()

    @staticmethod
    def __folder_is_vis_or_ms(folder: str) -> bool:
        return folder.endswith(".ms") or folder.endswith(".vis")

    def clean_up(self) -> None:
        # remove temp folder and everything inside it
        shutil.rmtree(self.path)

    def remove_file(self, file_path: str) -> None:
        os.remove(file_path)

    def copy_file(self, file_path: str) -> None:
        file_name = file_path.split(os.path.sep)[-1]
        shutil.copy(file_path, self.path + os.path.sep + file_name)

    def save_file(self, file_path: str, file_name: Optional[str] = None) -> None:
        if file_name is None:
            file_name = file_path.split(os.path.sep)[-1]
        shutil.copy(file_path, self.path + os.path.sep + file_name)


def check_ending(path: str, ending: str) -> None:
    if not path.endswith(ending):
        fname = path.split(os.path.sep)[-1]
        raise ValueError(
            f"Invalid file-ending, file {fname} must have {ending} extension!"
        )
