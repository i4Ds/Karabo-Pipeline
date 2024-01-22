from __future__ import annotations

import glob
import os
import random
import shutil
import string
from copy import copy
from types import TracebackType
from typing import Literal, Optional, Union, overload

from typing_extensions import assert_never

from karabo.util._types import DirPathType, FilePathType
from karabo.util.plotting_util import Font

_LongShortTermType = Literal["long", "short"]


def _get_tmp_dir() -> str:
    """Gets the according tmpdir.

    Defined env-var-dir > scratch-dir > tmp-dir

    Honors TMPDIR and TMP environment variable(s).
    The only thing not allowed is a collision between the mentioned env-vars.

    Returns:
        path of tmpdir
    """
    # first guess is just /tmp (low prio)
    tmpdir = f"{os.path.sep}tmp"
    # second guess is if scratch is available (mid prio)
    if (scratch := os.environ.get("SCRATCH")) is not None and os.path.exists(scratch):
        tmpdir = scratch
    # third guess is to honor the env-variables mentioned (high prio)
    env_check: Optional[str] = None  # variable to check previous environment variables
    environment_varname = ""
    if (TMPDIR := os.environ.get("TMPDIR")) is not None:
        tmpdir = os.path.abspath(TMPDIR)
        env_check = TMPDIR
        environment_varname = "TMPDIR"
    if (TMP := os.environ.get("TMP")) is not None:
        if env_check is not None:
            if TMP != env_check:
                raise RuntimeError(
                    f"Environment variables collision: TMP={TMP} != "
                    + f"{environment_varname}={env_check}"
                )
        else:
            tmpdir = os.path.abspath(TMP)
            env_check = TMP
            environment_varname = "TMP"
    return tmpdir


def _get_rnd_str(k: int, seed: str | int | float | bytes | None = None) -> str:
    random.seed(seed)
    return "".join(random.choices(string.ascii_letters + string.digits, k=k))


def _get_cache_dir(term: _LongShortTermType) -> str:
    """Creates cache-dir-name.

    dir-name: karabo-<LTM|STM>-($USER-)<10-rnd-asci-letters-and-digits>

    Returns:
        cache-dir-name
    """
    delimiter = "-"
    prefix = "karabo"
    if term == "long":
        prefix = delimiter.join((prefix, "LTM"))
    elif term == "short":
        prefix = delimiter.join((prefix, "STM"))
    else:
        assert_never(term)
    user = os.environ.get("USER")
    if user is not None:
        prefix = delimiter.join((prefix, user))
        seed = user + term
    else:
        seed = "42" + term
    suffix = _get_rnd_str(k=10, seed=seed)
    cache_dir_name = delimiter.join((prefix, suffix))
    return cache_dir_name


class FileHandler:
    """Utility file-handler for unspecified directories.

    Provides chache-management functionality.
    `FileHandler.root` is a static root-directory where each cache-dir is located.
    In case you want to extract something specific from the cache, the path is usually
    printed blue & bold in stdout.

    The root STM and LTM should be unique per user (seeded rnd chars+digits), thus just
    having two disk-cache directories per user.

    Set `FileHandler.root` to change the directory where files and dirs will be saved.
    The dir-structure is as follows where "tmp" is `FileHandler.root`:

    tmp
    ├── karabo-LTM-<user>-<10 rnd chars+digits>
    │   ├── <prefix><10 rnd chars+digits>
    |   |    ├── <sbudir>
    |   |    └── <file>
    │   └── <prefix><10 rnd chars+digits>
    |        ├── <sbudir>
    |        └── <file>
    └── karabo-STM-<user>-<10 rnd chars+digits>
        ├── <prefix><10 rnd chars+digits>
        |    ├── <sbudir>
        |    └── <file>
        └── <prefix><10 rnd chars+digits>
             ├── <sbudir>
             └── <file>

    LTM stand for long-term-memory (FileHandler.ltm) and STM for short-term-memory
    (FileHandler.stm). The data-products usually get into in the STM directory.

    FileHanlder can be used the same way as `tempfile.TemporaryDirectory` using `with`.
    """

    root: str = _get_tmp_dir()
    ltm = os.path.join(root, _get_cache_dir(term="long"))
    stm = os.path.join(root, _get_cache_dir(term="short"))

    def __init__(
        self,
    ) -> None:
        """Creates `FileHandler` instance."""
        # tmps is an instance bound dirs and/or files registry for STM
        self.tmps: list[str] = list()

    def _get_term_dir(self, term: _LongShortTermType) -> str:
        if term == "short":
            dir_ = FileHandler.stm
        elif term == "long":
            dir_ = FileHandler.ltm
        else:
            assert_never(term)
        return dir_

    @overload
    def get_tmp_dir(
        self,
        prefix: Union[str, None] = None,
        term: Literal["short"] = "short",
        purpose: Union[str, None] = None,
        unique: object = None,
        subdir: Union[DirPathType, None] = None,
        mkdir: bool = True,
    ) -> str:
        ...

    @overload
    def get_tmp_dir(
        self,
        prefix: str,
        term: Literal["long"],
        purpose: Union[str, None] = None,
        unique: object = None,
        subdir: Union[DirPathType, None] = None,
        mkdir: bool = True,
    ) -> str:
        ...

    def get_tmp_dir(
        self,
        prefix: Union[str, None] = None,
        term: _LongShortTermType = "short",
        purpose: Union[str, None] = None,
        unique: object = None,
        subdir: Union[DirPathType, None] = None,
        mkdir: bool = True,
    ) -> str:
        """Gets a tmp-dir path.

        This is the to-go function to get a tmp-dir in the according directory.

        Args:
            prefix: Dir-name prefix for STM (optional) and dir-name for LTM (required).
            term: "short" for STM or "long" for LTM.
            purpose: Creates a verbose print-msg with it's purpose if set.
            unique: If an object which has attributes is provided, then you get
                the same tmp-dir for the unique instance.
            subdir: If set, it directly creates & returns <tmp_dir>/subdir
            mkdir: Make-dir directly?

        Returns:
            tmp-dir path
        """
        obj_tmp_dir_short_name = "_karabo_tmp_dir_short"
        tmp_dir: Union[str, None] = None  # without subdir
        if unique is not None:
            if term != "short":
                raise RuntimeError(
                    "`unique` not None is just supported for short-term tmp-dirs."
                )
            try:
                unique.__dict__  # just to test try-except AttributeError
                if hasattr(unique, obj_tmp_dir_short_name):
                    tmp_dir = getattr(unique, obj_tmp_dir_short_name)
            except AttributeError:
                raise AttributeError(
                    "`unique` must be an object with attributes, "
                    + f"but is of type {type(unique)} instead."
                )

        if tmp_dir is not None:
            dir_path = tmp_dir
            if subdir is not None:
                dir_path = os.path.join(dir_path, subdir)
            exist_ok = True
        elif term == "short":
            dir_path = self._get_term_dir(term=term)
            dir_name = _get_rnd_str(k=10, seed=None)
            if prefix is not None:
                dir_name = "".join((prefix, dir_name))
            dir_path = os.path.join(dir_path, dir_name)
            if unique is not None:
                setattr(unique, obj_tmp_dir_short_name, dir_path)
            self.tmps.append(dir_path)
            if subdir is not None:
                dir_path = os.path.join(dir_path, subdir)
            exist_ok = False
        elif term == "long":
            dir_path = self._get_term_dir(term=term)
            if prefix is None:
                raise RuntimeError(
                    "For long-term-memory, `prefix` must be set to have unique dirs."
                )
            dir_name = _get_rnd_str(k=10, seed=prefix)
            dir_name = "".join((prefix, dir_name))
            dir_path = os.path.join(dir_path, dir_name)
            if subdir is not None:
                dir_path = os.path.join(dir_path, subdir)
            exist_ok = True
        else:
            assert_never(term)
        if not exist_ok and os.path.exists(dir_path):
            raise FileExistsError(f"{dir_path} already exists")
        if mkdir and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=exist_ok)
            if purpose and len(purpose) > 0:
                purpose = f" for {purpose}"
                print(f"Creating {Font.BLUE}{Font.BOLD}{dir_path}{Font.END}{purpose}")
        return dir_path

    def clean_instance(self) -> None:
        """Cleans instance-bound tmp-dirs of `self.tmps` from disk."""
        tmps = copy(self.tmps)
        for tmp in tmps:
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            self.tmps.remove(tmp)

    def clean(
        self,
        term: _LongShortTermType = "short",
    ) -> None:
        """Removes the entire directory specified by `term`.

        Be careful with cleaning, to not mess up dirs of other processes.

        Args:
            term: "long" or "short" term memory
        """
        dir_ = self._get_term_dir(term=term)
        if os.path.exists(dir_):
            shutil.rmtree(dir_)

    @staticmethod
    def is_dir_empty(dirname: DirPathType) -> bool:
        """Checks if `dirname` is empty assuming `dirname` exists.

        Args:
            dirname: Directory to check.

        Raises:
            NotADirectoryError: If `dirname` is not an existing directory.

        Returns:
            True if dir is empty, else False
        """
        if not os.path.isdir(dirname):
            raise NotADirectoryError(f"{dirname} is not an existing directory.")
        is_empty = len(os.listdir(path=dirname)) == 0
        return is_empty

    @staticmethod
    def remove_empty_dirs(term: _LongShortTermType = "short") -> None:
        """Removes emtpy directories in the chosen cache-dir.

        Args:
            term: "long" or "short" term memory
        """
        dir_ = _get_cache_dir(term=term)
        paths = glob.glob(os.path.join(dir_, "*"), recursive=False)
        for path in paths:
            if os.path.isdir(path) and len(os.listdir(path=path)) == 0:
                shutil.rmtree(path=path)

    @staticmethod
    def empty_dir(dir_path: DirPathType) -> None:
        """Deletes all contents of `dir_path`, but not the directory itself.

        This function assumes that all filed and directories are owned by
        the function-user.

        Args:
            dir_path: Directory to empty.
        """
        shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=False)

    def __enter__(self) -> str:
        return self.get_tmp_dir(prefix=None, term="short")

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.clean_instance()


def check_ending(path: Union[str, FilePathType, DirPathType], ending: str) -> None:
    """Utility function to check if the ending of `path` is `ending`.

    Args:
        path: Path to check.
        ending: Ending match.

    Raises:
        ValueError: When the ending of `path` doesn't match `ending`.
    """
    path_ = str(path)
    if not path_.endswith(ending):
        fname = path_.split(os.path.sep)[-1]
        raise ValueError(
            f"Invalid file-ending, file {fname} must have {ending} extension!"
        )
