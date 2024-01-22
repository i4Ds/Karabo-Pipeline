import json
import os
import tempfile

import pytest

from karabo.util.file_handler import FileHandler


def test_file_handler():
    """Test global FileHanlder functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        FileHandler.root = tmpdir
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
            subdir="dummy-dir",
        )
        assert len(os.listdir(FileHandler.stm())) == 2
        _ = fh_instance.get_tmp_dir(
            mkdir=False,
        )
        assert len(os.listdir(FileHandler.stm())) == 2
        with pytest.raises(RuntimeError):
            _ = FileHandler().get_tmp_dir(
                term="long",
                subdir="dummy-dir",
            )
        _ = FileHandler().get_tmp_dir(
            term="long",
            prefix="dummy-ltm-name",
            subdir="dummy-dir",
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
        FileHandler.root = tmpdir
        my_obj = MyClass()
        assert not os.path.exists(FileHandler.stm())
        tmpdir_fh1 = FileHandler().get_tmp_dir(unique=my_obj)
        assert len(os.listdir(FileHandler.stm())) == 1
        tmpdir_fh2 = FileHandler().get_tmp_dir(unique=my_obj)
        assert len(os.listdir(FileHandler.stm())) == 1
        assert tmpdir_fh1 == tmpdir_fh2
