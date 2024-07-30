import json
import os
import tempfile

import pytest

from karabo.util.file_handler import FileHandler, write_dir


def test_file_handler():
    """Test global FileHandler functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        FileHandler.root_ltm = tmpdir
        FileHandler.root_stm = tmpdir
        assert FileHandler.is_dir_empty(dirname=tmpdir)
        assert len(os.listdir(tmpdir)) == 0
        tmpdir_fh1 = FileHandler().get_tmp_dir(
            prefix="dummy-",
            purpose="test-file-handler-global disk-cache",
        )
        assert len(os.listdir(tmpdir)) == 1
        assert not FileHandler.is_dir_empty(dirname=tmpdir)
        assert len(os.listdir(FileHandler.stm())) == 1
        json_path = os.path.join(tmpdir_fh1, "my_json.json")
        with open(json_path, "w") as outfile1:
            json.dump({"A": "B"}, outfile1)
        assert os.path.exists(json_path)
        fh_instance = FileHandler()
        _ = fh_instance.get_tmp_dir(
            prefix="dummy-",
        )
        assert len(os.listdir(FileHandler.stm())) == 2
        _ = fh_instance.get_tmp_dir(
            mkdir=False,
        )
        assert len(os.listdir(FileHandler.stm())) == 2
        with pytest.raises(RuntimeError):
            _ = FileHandler().get_tmp_dir(
                term="long",
            )
        _ = FileHandler().get_tmp_dir(
            term="long",
            prefix="dummy-ltm-name",
        )
        assert len(os.listdir(tmpdir)) == 2
        assert len(os.listdir(FileHandler.ltm())) == 1
        assert len(os.listdir(FileHandler.stm())) == 2

        fh_instance.clean_instance()
        assert len(os.listdir(FileHandler.stm())) == 1

        _ = FileHandler().get_tmp_dir()
        fh_empty = FileHandler()
        empty_path = fh_empty.get_tmp_dir()
        assert len(os.listdir(FileHandler.stm())) == 3
        assert len(os.listdir(empty_path)) == 0
        json_empty_path = os.path.join(empty_path, "my_json.json")
        with open(json_empty_path, "w") as outfile2:
            json.dump({"A": "B"}, outfile2)
        assert len(os.listdir(empty_path)) == 1
        FileHandler.empty_dir(dir_path=empty_path)
        assert len(os.listdir(empty_path)) == 0
        assert len(os.listdir(FileHandler.stm())) == 3
        fh_empty.clean_instance()
        assert len(os.listdir(FileHandler.stm())) == 2

        FileHandler.clean()
        assert not os.path.exists(FileHandler.stm())


def test_object_bound_file_handler():
    """Test obj unique FileHandler creation."""

    class MyClass:
        ...

    with tempfile.TemporaryDirectory() as tmpdir:
        FileHandler.root_stm = tmpdir
        FileHandler.root_ltm = tmpdir
        my_obj = MyClass()
        assert not os.path.exists(FileHandler.stm())
        tmpdir_fh1 = FileHandler().get_tmp_dir(unique=my_obj)
        assert len(os.listdir(FileHandler.stm())) == 1
        tmpdir_fh2 = FileHandler().get_tmp_dir(unique=my_obj)
        assert len(os.listdir(FileHandler.stm())) == 1
        assert tmpdir_fh1 == tmpdir_fh2


def test_write_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        # test successful subdir creation
        dir1 = os.path.join(tmpdir, "dir1")
        with write_dir(dir=dir1, overwrite=False) as wd:
            os.makedirs(os.path.join(wd, "subdir1"))
        assert os.path.exists(os.path.join(dir1, "subdir1"))

        # test overwrite=False
        with pytest.raises(FileExistsError):
            with write_dir(dir=dir1, overwrite=False) as wd:
                pass

        # test overwrite=True
        with write_dir(dir=dir1, overwrite=True) as wd:
            os.makedirs(os.path.join(wd, "subdir2"))
        assert not os.path.exists(os.path.join(dir1, "subdir1"))
        assert os.path.exists(os.path.join(dir1, "subdir2"))

        # test interrupt
        try:
            with write_dir(dir=dir1, overwrite=True) as wd:
                os.makedirs(os.path.join(wd, "subdir3"))
                raise RuntimeError()
        except RuntimeError:
            pass
        assert not os.path.exists(os.path.join(dir1, "subdir1"))
        assert os.path.exists(os.path.join(dir1, "subdir2"))
        assert not os.path.exists(os.path.join(dir1, "subdir3"))

        # test multiple subdir removal
        try:
            with write_dir(dir=dir1, overwrite=True) as wd:
                subdirs = os.path.join(wd, "subdir4", "subdir4", "subdir4")
                os.makedirs(subdirs)
                assert os.path.exists(subdirs)
                raise RuntimeError()
        except RuntimeError:
            pass
        assert os.path.exists(os.path.join(dir1, "subdir2"))
        assert not os.path.exists(os.path.join(dir1, "subdir4"))

        # test successful root-dir removal
        dir2 = os.path.join(tmpdir, "dir2")
        try:
            with write_dir(dir=dir2, overwrite=False) as wd:
                os.makedirs(os.path.join(wd, "subdir1"))
                raise RuntimeError()
        except RuntimeError:
            pass
        assert not os.path.exists(dir2)
