import json
import os
import tempfile

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
        assert len(os.listdir(FileHandler.stm)) == 1
        json_path = os.path.join(tmpdir_fh1, "my_json.json")
        with open(json_path, "w") as outfile1:
            json.dump({"A": "B"}, outfile1)
        assert os.path.exists(json_path)
        fh_instance = FileHandler()
        _ = fh_instance.get_tmp_dir(
            prefix="dummy-",  # same name as fh1 is intentional
            subdir="dummy-dir",
        )
        assert len(os.listdir(FileHandler.stm)) == 2
        _ = fh_instance.get_tmp_dir(
            mkdir=False,
        )
        assert len(os.listdir(FileHandler.stm)) == 2
        _ = FileHandler().get_tmp_dir(
            term="long",
            subdir="dummy-dir",
        )
        assert len(os.listdir(tmpdir)) == 2
        assert len(os.listdir(FileHandler.ltm)) == 1
        assert len(os.listdir(FileHandler.stm)) == 2

        fh_instance.clean_instance()
        assert len(os.listdir(FileHandler.stm)) == 1

        empty_path = FileHandler.get_tmp_dir()
        _ = FileHandler.get_tmp_dir()
        assert len(os.listdir(FileHandler.stm)) == 3
        FileHandler.empty_dir(dir_path=empty_path)
        assert len(os.listdir(FileHandler.stm)) == 2

        FileHandler.clean()
        assert len(os.listdir(FileHandler.stm)) == 0


def test_object_bound_file_handler():
    """Test obj unique FileHandler creation."""

    class MyClass:
        ...

    with tempfile.TemporaryDirectory() as tmpdir:
        FileHandler.root = tmpdir
        my_obj = MyClass()
        assert len(os.listdir(FileHandler.stm)) == 0
        tmpdir_fh1 = FileHandler().get_tmp_dir(unique=my_obj)
        assert len(os.listdir(FileHandler.stm)) == 1
        tmpdir_fh2 = FileHandler().get_tmp_dir(unique=my_obj)
        assert len(os.listdir(FileHandler.stm)) == 1
        assert tmpdir_fh1 == tmpdir_fh2
