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
    path : str, optional
        Directory path where the resulting files will be stored or
        a file path.
        If not provided, a default directory named 'karabo_folder'
        will be created in the current working directory.
    create_additional_folder_in_dir : bool, optional
        Whether to create a new folder inside the directory specified
        by dir. Default is False. If True, a unique UUID will be used
        as the name of the folder.
    file_name : str, optional
        Name of the output file. If not provided, a unique UUID will
        be used as the filename.
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
        path: Optional[str] = None,
        create_additional_folder_in_dir: bool = False,
        file_name: Optional[str] = None,
        suffix: str = "",
    ) -> None:
        filehandle_root_path: str = os.getcwd()
        use_scratch_folder_if_exist: bool = True

        if "SCRATCH" in os.environ and use_scratch_folder_if_exist:
            filehandle_root_path = os.environ["SCRATCH"]

        # Check if the passed path is a path to a file
        if path and os.path.isfile(path):
            file_name = os.path.basename(path)
            path = os.path.dirname(path)
            suffix = ""

        # If a directory is provided, use it as the base path
        if path:
            base_path = os.path.abspath(path)
        else:
            base_path = os.path.join(filehandle_root_path, "karabo_folder")
            # generate unique id, either use e.g. a JOBID or generate a UUID
            if "SLURM_JOB_ID" in os.environ:
                unique_id = str(os.environ["SLURM_JOB_ID"])
            else:
                unique_id = str(uuid.uuid4())
            if suffix.lower() == ".ms":
                base_path = os.path.join(base_path, unique_id + ".MS")
            else:
                base_path = os.path.join(base_path, unique_id)

        # If a new folder to host the data should be created inside the base_path
        if create_additional_folder_in_dir:
            base_path = os.path.join(base_path, str(uuid.uuid4()))

        # Make the base path if it does not exist
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        # If a file name is provided, use it as the path
        if file_name:
            self.path = os.path.join(base_path, file_name + suffix)
        else:
            self.path = base_path

        self.dir = base_path
        self.file_name = file_name
        self.suffix = suffix

    def clean_up(self) -> None:
        shutil.rmtree(self.dir)

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
