import json
import os
import tempfile

from karabo.util.file_handler import FileHandler


def test_file_handler_global():
    """Test global FileHanlder functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # test root
        assert FileHandler.root == os.path.join(os.getcwd(), "karabo_folder")
        FileHandler.root = tmpdir

        # add 2 dirs created through FileHanlder with and without random content
        fh1 = FileHandler(prefix="my_domain", verbose=True)
        with open(os.path.join(fh1.subdir, "my_json.json"), "w") as outfile1:
            json.dump({"A": "B"}, outfile1)
        _ = FileHandler(prefix="my_other_domain", verbose=False)
        assert len(os.listdir(tmpdir)) == 2

        # create 2 additional random other dirs and files with and without content
        os.mkdir(path=os.path.join(tmpdir, "my_dir1"))
        with open(os.path.join(tmpdir, "my_dir1", "my_json.json"), "w") as outfile2:
            json.dump({"A": "B"}, outfile2)
        os.mkdir(path=os.path.join(tmpdir, "my_dir2"))
        with open(os.path.join(tmpdir, "my_root_json.json"), "w") as outfile3:
            json.dump({"A": "B"}, outfile3)
        assert len(os.listdir(tmpdir)) == 5

        # test removal of dirs not created from FileHandler
        FileHandler.clean_up_fh_root(force=False, verbose=True)
        assert len(os.listdir(tmpdir)) == 3
        # test removal of emtpy (remaining) dirs
        FileHandler.remove_empty_dirs(consider_fh_dir_identifier=False)
        assert len(os.listdir(tmpdir)) == 2  # 1 file & 1 non-empty dir
        # test removal of FileHandler root
        FileHandler.clean_up_fh_root(force=True, verbose=False)
        assert not os.path.exists(tmpdir)


def test_file_handler_instances():
    """Test instance bound dir creation and removal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        FileHandler.root = tmpdir
        fh1 = FileHandler(prefix="my_domain", verbose=True)
        assert len(os.listdir(tmpdir)) == 1
        fh2 = FileHandler()
        assert len(os.listdir(tmpdir)) == 2
        fh1.clean_up()
        assert len(os.listdir(tmpdir)) == 1
        fh2.clean_up()
        assert not os.path.exists(tmpdir)
        with FileHandler() as fhdir:
            assert os.path.exists(fhdir)
            assert len(os.listdir(tmpdir)) == 1
        assert not os.path.exists(tmpdir)


def test_get_file_handler():
    """Test obj unique FileHandler creation."""
    prefix = "my_domain"

    class MyClass:
        ...

    with tempfile.TemporaryDirectory() as tmpdir:
        FileHandler.root = tmpdir
        my_obj1 = MyClass()
        my_obj2 = MyClass()
        _ = FileHandler.get_file_handler(obj=my_obj1, prefix=prefix, verbose=True)
        assert len(os.listdir(tmpdir)) == 1
        _ = FileHandler.get_file_handler(obj=my_obj1, prefix=prefix, verbose=False)
        len(os.listdir(tmpdir))
        assert len(os.listdir(tmpdir)) == 1
        _ = FileHandler.get_file_handler(obj=my_obj2, prefix=prefix, verbose=False)
        len(os.listdir(tmpdir))
        assert len(os.listdir(tmpdir)) == 2
