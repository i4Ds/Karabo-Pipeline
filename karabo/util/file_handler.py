from __future__ import annotations

import glob
import os
import random
import shutil
import string
import uuid
from types import TracebackType
from typing import Optional, Union

from karabo.util._types import DirPathType, FilePathType
from karabo.util.plotting_util import Font


def _get_tmp_dir() -> str:
    """Gets the according tmpdir.

    Defined env-var-dir > scratch-dir > tmp-dir

    Honors TMPDIR, TEMP and TMP environment variable(s).
    The only thing not allowed is a collision between the mentioned env-vars.

    Returns:
        path of tmpdir
    """
    # first guess is just /tmp (low prio)
    tmpdir = f"{os.path.sep}tmp"
    # second guess is if scratch is available (mid prio)
    scratch = os.environ.get("SCRATCH")
    if scratch is not None and os.path.exists(scratch):
        tmpdir = scratch
    # third guess is to honor the env-variables mentioned (high prio)
    env_check: Optional[str] = None  # variable to check previous environment variables
    environment_varname = ""
    if (TMPDIR := os.environ.get("TMPDIR")) is not None:
        tmpdir = os.path.abspath(TMPDIR)
        env_check = TMPDIR
        environment_varname = "TMPDIR"
    if (TEMP := os.environ.get("TEMP")) is not None:
        if env_check is not None:
            if TEMP != env_check:
                raise RuntimeError(
                    f"Environment variables collision: TEMP={TEMP} != "
                    + f"{environment_varname}={env_check}"
                )
        else:
            tmpdir = os.path.abspath(TEMP)
            env_check = TEMP
            environment_varname = "TEMP"
    if (TMP := os.environ.get("TMP")) is not None:
        if env_check is not None:
            if TMP != env_check:
                raise RuntimeError(
                    f"Environment variables collision: TMP={TMP} != "
                    + f"{environment_varname}={env_check}"
                )
        else:
            tmpdir = os.path.abspath(TMP)
            env_check = TEMP
            environment_varname = "TEMP"
    return tmpdir


def _get_cache_dir() -> str:
    """Gets a default cache-dir.

    dir-name: karabo-($USER-)<10-rnd-asci-letters-and-digits>

    Returns:
        path of cache-dir
    """
    tmpdir = _get_tmp_dir()
    delimiter = "-"
    prefix = "karabo"
    user = os.environ.get("USER")
    if user is not None:
        prefix = delimiter.join((prefix, user))
    suffix = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    cache_dir_name = delimiter.join((prefix, suffix))
    cache_dir = os.path.join(tmpdir, cache_dir_name)
    return cache_dir


class FileHandler:
    """Utility file-handler for unspecified directories.

    Provides chache-management functionality.
    `FileHandler.root` is a static root-directory where each cache-dir is located.
    In case you want to extract something specific from the cache, the path is usually
    printed blue & bold in stdout.

    Set `FileHandler.root` to change the directory where files and dirs will be saved.
    Otherwise, we provide $TMP, $TMPDIR & $TEMP with a following /karabo-cache as root.
    Subdirs are usually {prefix}_{fh_dir_identifier}_{uuid4} in case `prefix`
    is defined, otherwise just {fh_dir_identifier}_{uuid4}.
    This class provides an additional security layer for the removal of subdirs
    in case a root is specified where other files and directories live.
    FileHanlder can be used the same way as `tempfile.TemporaryDirectory` using `with`.
    """

    root: str = _get_cache_dir()
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
        subdir_name = str(uuid.uuid4())
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
    def remove_empty_dirs(consider_fh_dir_identifier: bool = True) -> None:
        """Removes emtpy directories in `FileHandler.root`.

        Just manual use recommended since it doesn't consider directories which
         are currently in use and therefore it could interrupt running code.

        Args:
            consider_fh_dir_identifier: Consider `fh_dir_identifier` for dir matching?
        """
        paths = glob.glob(os.path.join(FileHandler.root, "*"), recursive=False)
        for path in paths:
            if os.path.isdir(path) and len(os.listdir(path=path)) == 0:
                if consider_fh_dir_identifier:
                    if FileHandler.fh_dir_identifier in os.path.split(path)[-1]:
                        shutil.rmtree(path=path)
                else:
                    shutil.rmtree(path=path)

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
                        and FileHandler.fh_dir_identifier in os.path.split(path)[-1]
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

    @staticmethod
    def get_file_handler(
        obj: object,
        prefix: Optional[str] = None,
        verbose: bool = True,
    ) -> FileHandler:
        """Utility function to always get unique `FileHandler` bound to `obj`.

        `FileHandler` args have just an effect while the first instance is created.

        Args:
            obj: Any object which should have an unique `FileHandler` assigned.
            prefix: See `FileHandler.__init__`
            verbose: See `FileHandler.__init__`

        Returns:
            The `FileHandler` bound to `obj`.
        """
        for attr_name in obj.__dict__:
            attr = getattr(obj, attr_name)
            if isinstance(attr, FileHandler):
                return attr
        fh = FileHandler(prefix=prefix, verbose=verbose)
        setattr(obj, "file_handler", fh)
        return fh

    def __enter__(self) -> str:
        return self.subdir

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.clean_up()


def check_ending(path: Union[str, FilePathType, DirPathType], ending: str) -> None:
    path_ = str(path)
    if not path_.endswith(ending):
        fname = path_.split(os.path.sep)[-1]
        raise ValueError(
            f"Invalid file-ending, file {fname} must have {ending} extension!"
        )
