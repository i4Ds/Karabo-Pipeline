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

    Set `FileHandler.root` to change the directory where files and dirs will be saved.
    The dir-structure is as follows where "tmp" is `FileHandler.root`:

    tmp
    ├── karabo-LTM-<user>-<10 rnd chars+digits>
    │   ├── a-dir
    │   └── another-dir
    └── karabo-STM-<user>-<10 rnd chars+digits>
        ├── a-dir
        └── another-dir

    LTM stand for long-term-memory (self.ltm) and STM for short-term-memory (self.stm).
    The data-products usually get into in the STM directory.

    FileHanlder can be used the same way as `tempfile.TemporaryDirectory` using `with`.
    """

    root: str = _get_tmp_dir()

    def __init__(
        self,
    ) -> None:
        """Creates `FileHandler` instance."""
        self._ltm_dir_name = _get_cache_dir(term="long")
        self._stm_dir_name = _get_cache_dir(term="short")
        # tmps is an instance bound dirs and/or files registry for STM
        self.tmps: list[str] = list()

    @property
    def ltm(self) -> str:
        ltm_path = os.path.join(FileHandler.root, self._ltm_dir_name)
        os.makedirs(ltm_path, exist_ok=True)
        return ltm_path

    @property
    def stm(self) -> str:
        stm_path = os.path.join(FileHandler.root, self._stm_dir_name)
        os.makedirs(stm_path, exist_ok=True)
        return stm_path

    def _get_term_dir(self, term: _LongShortTermType) -> str:
        if term == "short":
            dir_ = self.stm
        elif term == "long":
            dir_ = self.ltm
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
    ) -> str:
        ...

    @overload
    def get_tmp_dir(
        self,
        prefix: str,
        term: Literal["long"],
        purpose: Union[str, None] = None,
        unique: object = None,
    ) -> str:
        ...

    def get_tmp_dir(
        self,
        prefix: Union[str, None] = None,
        term: _LongShortTermType = "short",
        purpose: Union[str, None] = None,
        unique: object = None,
    ) -> str:
        """Gets a tmp-dir path.

        This is the to-go function to get a tmp-dir in the according directory.

        Args:
            prefix: Dir-name prefix for STM (optional) and dir-name for LTM (required).
            term: "short" for STM or "long" for LTM.
            purpose: Creates a verbose print-msg with it's purpose if set.
            unique: If an object which has attributes is provided, then you get
                the same tmp-dir for the unique instance.

        Returns:
            tmp-dir path
        """
        set_unique = False
        obj_tmp_dir_name = "_karabo_tmp_dir"
        if unique is not None:
            if term != "short":
                raise RuntimeError(
                    "`unique` not None is just supported for short-term tmp-dirs."
                )
            try:
                unique.__dict__  # just to test try-except
                if hasattr(unique, obj_tmp_dir_name):
                    return getattr(unique, obj_tmp_dir_name)
                else:
                    set_unique = True
            except AttributeError:
                raise AttributeError(
                    "`unique` must be an object with attributes, "
                    + f"but is of type {type(unique)} instead."
                )

        dir_path = self._get_term_dir(term=term)
        if term == "short":
            dir_name = _get_rnd_str(k=10, seed=None)
            if prefix is not None:
                dir_name = "".join((prefix, dir_name))
            dir_path = os.path.join(dir_path, dir_name)
            os.makedirs(dir_path, exist_ok=False)
            self.tmps.append(dir_path)
        elif term == "long":
            if prefix is None:
                raise RuntimeError(
                    "For long-term-memory, `prefix` must be set to have unique dirs."
                )
            dir_name = prefix
            dir_path = os.path.join(dir_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)
        else:
            assert_never(term)
        if set_unique:
            setattr(unique, obj_tmp_dir_name, dir_path)
        if purpose:
            if len(purpose) > 0:
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
