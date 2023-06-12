import os
import tempfile
from typing import Generator

import pytest

from karabo.test import data_path
from karabo.util.file_handle import FileHandle


def setup_handle(
    file_name=None,
    suffix="",
    path=None,
    create_additional_folder_in_dir=False,
) -> FileHandle:
    return FileHandle(
        file_name=file_name,
        suffix=suffix,
        path=path,
        create_additional_folder_in_dir=create_additional_folder_in_dir,
    )


@pytest.fixture
def handle() -> Generator[FileHandle, None, None]:
    handle = setup_handle()
    yield handle  # yield the setup


@pytest.mark.parametrize("file_name, suffix", [("file.file", ""), ("file", ".file")])
def test_create_file(file_name, suffix):
    handle = setup_handle(file_name=file_name, suffix=suffix)
    dir_path = os.path.dirname(handle.path)
    assert os.path.exists(dir_path)
    handle.clean_up()
    assert not os.path.exists(dir_path)


def test_create_folder(handle: FileHandle):
    assert os.path.exists(handle.path)
    handle.clean_up()
    assert not os.path.exists(handle.path)


def test_existing_folder(handle: FileHandle):
    path = handle.path
    assert os.path.exists(path)

    handle = setup_handle(path=path)
    assert os.path.exists(handle.path)
    handle.clean_up()
    assert not os.path.exists(handle.path)


def test_existing_file():
    handle = setup_handle(path=data_path, file_name="detection.csv")
    assert os.path.exists(handle.path)


def test_cleanup(handle: FileHandle):
    assert os.path.exists(handle.path)
    handle.clean_up()
    assert not os.path.exists(handle.path)


def test_folder_creation_in_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        handle = setup_handle(path=tmpdir, create_additional_folder_in_dir=True)
        assert tmpdir == os.path.split(handle.path)[0]
        handle.clean_up()
        assert os.path.exists(tmpdir)
