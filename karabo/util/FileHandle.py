import os
import shutil
import uuid
from typing import Optional


class FileHandle:
    """
    Utility class for handling file paths and creating temporary
    files in a given directory.

    Parameters
    ----------
    dir : str, optional
        Directory path where the resulting files will be stored.
        If not provided, a default directory named 'karabo_folder'
        will be created in the current working directory.
    file_name : str, optional
        Name of the output file. If not provided, a unique UUID will
        be used as the filename.
    create_file : bool, optional
        Whether to create the file or not. Default is False. If True,
        the file will be created in the directory specified by dir by touching.
    suffix : str, optional
        Suffix to add to the filename. Default is an empty string.

    Attributes
    ----------
    path : str
        Absolute path to the file or directory.

    Methods
    -------
    clean_up() -> None:
        Remove the directory and its contents.
    remove_file(file_path: str) -> None:
        Remove a file given its absolute path.
    copy_file(file_path: str) -> None:
        Copy a file to the directory.
    save_file(file_path: str, file_name: str) -> None:
        Save a file to the directory with a given name.

    """

    def __init__(
        self,
        dir: Optional[str] = None,
        file_name: Optional[str] = None,
        create_file: bool = False,
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
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        else:
            os.makedirs(self.path, exist_ok=True)

        # Create the file if it does not exist
        if create_file and file_name:
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
