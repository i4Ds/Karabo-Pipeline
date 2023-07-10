import os
import tempfile

import pytest

from karabo.util.file_handle import FileHandle


def setup_handle(
    file_name=None,
    suffix="",
    path=None,
    create_additional_folder_in_dir=False,
    dir=None,
) -> FileHandle:
    return FileHandle(
        file_name=file_name,
        dir=dir,
        suffix=suffix,
        path=path,
        create_additional_folder_in_dir=create_additional_folder_in_dir,
    )


@pytest.mark.parametrize("file_name, suffix", [("file.file", ""), ("file", ".file")])
def test_create_file(file_name, suffix):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_name = os.path.join(tmpdir, file_name)
        handle = setup_handle(file_name=file_name, suffix=suffix)
        dir_path = os.path.dirname(handle.path)
        assert os.path.exists(dir_path)
        handle.clean_up()
        assert not os.path.exists(dir_path)


def test_create_folder_and_cleanup():
    with tempfile.TemporaryDirectory() as tmpdir:
        handle = setup_handle(dir=tmpdir)
        assert os.path.exists(handle.path)
        handle.clean_up()
        assert not os.path.exists(handle.path)


def test_existing_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        handle = setup_handle(dir=tmpdir)
        path = handle.path
        assert os.path.exists(path)

        handle = setup_handle(path=path)
        assert os.path.exists(handle.path)
        handle.clean_up()
        assert not os.path.exists(handle.path)


def test_folder_creation_in_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        handle = setup_handle(path=tmpdir, create_additional_folder_in_dir=True)
        assert tmpdir == os.path.split(handle.path)[0]
        handle.clean_up()
        assert os.path.exists(tmpdir)
