from __future__ import annotations

import glob
import os
import re
import shutil
import uuid
from types import TracebackType
from typing import Optional

from karabo.util.plotting_util import Font


def _get_default_root_dir() -> str:
    karabo_folder = "karabo_folder"
    scratch = os.environ.get("SCRATCH")
    if scratch is not None:
        root_parent = scratch
    else:
        root_parent = os.getcwd()
    root_dir = os.path.join(root_parent, karabo_folder)
    return os.path.abspath(root_dir)


class FileHandler:
    """Utility file-handler for unspecified directories.

    Provides directory-management functionality in case no dir-path was specified.
    `FileHandler.root` is a static root-directory where each subdir is located.
    Subdirs are usually {prefix}_{fh_dir_identifier}_{uuid4[:8]} in case `prefix`
     is defined, otherwise just {fh_dir_identifier}_{uuid4[:8]}.
    This class provides an additional security layer for the removal of subdirs
     in case a root is specified where other files and directories live.
    FileHanlder can be used the same way as `tempfile.TemporaryDirectory` using with.

    Args:
        prefix: Prefix of dir-path where dir-path is `prefix`_{uuid4[:8]}
    """

    root: str = _get_default_root_dir()
    fh_dir_identifier = "fhdir"  # additional security to protect against dir-removal

    def __init__(
        self,
        prefix: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Creates `FileHandler` instance with the according sub-directory.

        Args:
            prefix: Prefix for easier identification of sub-directory.
            verbose: Subdir creation and removal verbose?
        """
        self.verbose = verbose
        subdir_name = str(uuid.uuid4())[:8]
        if (
            FileHandler.fh_dir_identifier is not None
            and len(FileHandler.fh_dir_identifier) > 0
        ):
            subdir_name = f"{FileHandler.fh_dir_identifier}_{subdir_name}"
        if prefix is not None and len(prefix) > 0:
            subdir_name = f"{prefix}_{subdir_name}"
        self.subdir = os.path.join(FileHandler.root, subdir_name)
        if self.verbose:
            print(
                f"Creating {Font.BLUE}{Font.BOLD}{self.subdir}{Font.END} "
                "directory for data object storage."
            )
        os.makedirs(self.subdir, exist_ok=False)

    def clean_up(self) -> None:
        """Removes instance-bound `self.subdir`."""
        if os.path.exists(self.subdir):
            if self.verbose:
                print(f"Removing {self.subdir}")
            shutil.rmtree(self.subdir)
            if len(os.listdir(FileHandler.root)) == 0:
                shutil.rmtree(FileHandler.root)

    @staticmethod
    def clean_up_fh_root(force: bool = False, verbose: bool = True) -> None:
        """Removes the from `FileHandler` created directories.

        Args:
            force: Remove `FileHandler.root` entirely regardless of content?
            verbose: Verbose removal?
        """
        if os.path.exists(FileHandler.root):
            if force:  # force remove fh-root
                if verbose:
                    print(f"Force remove {FileHandler.root}")
                shutil.rmtree(FileHandler.root)
            elif (  # check if fh-dir-identifier is properly set for safe removal
                FileHandler.fh_dir_identifier is None
                or len(FileHandler.fh_dir_identifier) < 1
            ):
                print(
                    "`clean_up_fh_root` can't remove anything because "
                    f"{FileHandler.fh_dir_identifier=}. Set `fh_dir_identifier` "
                    f"correctly or use `force` to remove {FileHandler.root} regardless."
                )
            else:
                if verbose:
                    print(
                        f"Remove {FileHandler.root} in case all subdirs match "
                        f"{FileHandler.fh_dir_identifier=}"
                    )
                paths = glob.glob(os.path.join(FileHandler.root, "*"))
                for path in paths:
                    if (
                        os.path.isdir(path)
                        and re.match(
                            FileHandler.fh_dir_identifier, os.path.split(path)[-1]
                        )
                        is not None
                    ):  # safe removal of subdir because it has the fh-dir-identifier
                        shutil.rmtree(path=path)
                if len(os.listdir(FileHandler.root)) > 0:
                    if verbose:
                        print(
                            f"`clean_up_fh_root` is not able safely remove "
                            f"{FileHandler.root} because there are directories which "
                            f"don't match {FileHandler.fh_dir_identifier=} or files."
                        )
                else:  # remove fh-root if dir is empty
                    shutil.rmtree(FileHandler.root)

    def __enter__(self) -> str:
        return self.subdir

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.clean_up()


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
