import os.path

import pytest

from karabo.test import data_path
from karabo.util.file_handle import FileHandle


def setup_handle(
    file_name=None, suffix="", path=None, create_additional_folder_in_dir=False
):
    return FileHandle(
        file_name=file_name,
        suffix=suffix,
        path=path,
        create_additional_folder_in_dir=create_additional_folder_in_dir,
    )


@pytest.fixture
def handle():
    handle = setup_handle()
    yield handle  # yield the setup


@pytest.mark.parametrize("file_name, suffix", [("file.file", ""), ("file", ".file")])
def test_create_file(file_name, suffix):
    handle = setup_handle(file_name=file_name, suffix=suffix)
    dir_path = os.path.dirname(handle.path)
    assert os.path.exists(dir_path)
    handle.clean_up()
    assert not os.path.exists(dir_path)


def test_create_folder(handle):
    assert os.path.exists(handle.path)
    handle.clean_up()
    assert not os.path.exists(handle.path)


def test_existing_folder(handle):
    path = handle.path
    assert os.path.exists(path)

    handle = setup_handle(path=path)
    assert os.path.exists(handle.path)
    handle.clean_up()
    assert not os.path.exists(handle.path)


def test_existing_file():
    handle = setup_handle(path=data_path, file_name="detection.csv")
    assert os.path.exists(handle.path)


def test_cleanup(handle):
    assert os.path.exists(handle.path)
    handle.clean_up()
    assert not os.path.exists(handle.path)


@pytest.mark.parametrize(
    "path, file_name, expected_path, expected_dir",
    [
        (
            os.path.join(data_path, "test_123.ms"),
            None,
            os.path.join(data_path, "test_123.ms"),
            os.path.join(data_path, "test_123.ms"),
        ),
        (data_path, "test_123.ms", os.path.join(data_path, "test_123.ms"), data_path),
    ],
)
def test_correct_path_and_file_location(path, file_name, expected_path, expected_dir):
    handle = setup_handle(path=path, file_name=file_name)
    assert handle.path == expected_path
    assert handle.dir == expected_dir


def test_folder_creation_in_folder():
    path = os.path.join(data_path, "test_123")
    if not os.path.exists(path):
        os.mkdir(path)
    handle = setup_handle(path=path, create_additional_folder_in_dir=True)
    assert path == os.path.split(handle.path)[0]
    handle.clean_up()
    assert os.path.exists(path)
