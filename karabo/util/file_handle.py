import os
import shutil
import uuid
from typing import Optional


def _get_default_root_dir() -> str:
    karabo_folder = "karabo_folder"
    if os.environ.get("SCRATCH") is not None:
        return os.path.join(os.environ.get("SCRATCH"), karabo_folder)
    else:
        return os.path.join(os.getcwd(), karabo_folder)


class FileHandler:
    """Utility file-handler for unspecified directories.

    Provides directory-management functionality in case no dir-path was specified.
    `FileHandler.root` is a static root-directory where each subdir is located.
    Subdirs are `prefix`_{uuid4[:8]} in case `prefix` is defined, otherwise uuid4[:8].

    Args:
        prefix: Prefix of dir-path where dir-path is `prefix`_{uuid4[:8]}
    """

    root: str = _get_default_root_dir()
    fh_dir_identifier = "fhdir"  # additional security to protect against dir-removal

    def __init__(
        self,
        prefix: Optional[str] = None,
    ) -> None:
        self.subdir = f"{FileHandler.fh_dir_identifier}_{str(uuid.uuid4())[:8]}"
        if prefix is not None:
            self.subdir = f"{prefix}_{self.subdir}"
        os.makedirs(self.subdir, exist_ok=True)

    def clean_up(self) -> None:
        if os.path.exists(self.subdir):
            shutil.rmtree(self.subdir)  # make dir-removal protected

    @staticmethod
    def clean_up_root(force: bool = False) -> None:
        if os.path.exists(FileHandler.root):
            shutil.rmtree(FileHandler.root)  # make dir-removal protected


class FileHandle:
    """
    Utility class for handling file paths and creating temporary
    files in a given directory.

    Parameters
    ----------
    path : str, optional
        File path where the resulting file will be stored.
        If not provided, a uuid will be created.
        Can also be a folder but then it needs to be a
        .ms file.
    dir : str, optional
        Directory where the resulting file will be stored.
        If not provided, it either will be saved in the current
        working directory or in a scratch folder if it exists.
    create_additional_folder_in_dir : bool, optional
        Whether to create a new folder inside the directory specified
        by dir. Default is False. If True, a unique UUID will be used
        as the name of the folder.
    file_name : str, optional
        Name of the output file. If not provided, a unique UUID will
        be used as the filename.
    file_is_dir : bool
        If the file_name is a folder instead of a file.
        Example: for measurement sets the "file" is a folder.
    suffix : str, optional
        Suffix to add to the filename. Default is an empty string.
        Example: ".MS"

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
        dir: Optional[str] = None,
        create_additional_folder_in_dir: bool = False,
        file_name: Optional[str] = None,
        file_is_dir: bool = False,
        suffix: str = "",
    ) -> None:
        use_scratch_folder_if_exist: bool = True
        use_slurm_job_id_if_exist: bool = True

        # Some logic
        if not dir:
            if path:
                dir = os.path.split(path)[0]
            elif "SCRATCH" in os.environ and use_scratch_folder_if_exist:
                dir = os.path.join(os.environ["SCRATCH"], "karabo_folder")
            else:
                dir = os.path.join(os.getcwd(), "karabo_folder")
            if "SLURM_JOB_ID" in os.environ and use_slurm_job_id_if_exist:
                dir = os.path.join(dir, os.environ["SLURM_JOB_ID"])

        if create_additional_folder_in_dir:
            dir = os.path.join(dir, str(uuid.uuid4()))

        if not file_name:
            if path:
                file_name = path.split(os.path.sep)[-1]
            else:
                file_name = str(uuid.uuid4()) + suffix

        if not path:
            path = os.path.join(dir, file_name)

        # Add some logic to file_is_dir
        if path.lower().endswith(".ms") or file_name.lower().endswith(".ms"):
            file_is_dir = True

        if file_is_dir:
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(dir, exist_ok=True)

        # Set variables
        self.path = path
        self.dir = dir
        self.file_name = file_name

    def clean_up(self) -> None:
        if os.path.exists(self.path):
            if os.path.isfile(self.path):
                self.remove_file(self.path)
            else:
                self.remove_dir(self.path)

    def remove_file(self, file_path: str) -> None:
        os.remove(file_path)

    def remove_dir(self, dir_path: str) -> None:
        shutil.rmtree(dir_path)

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
