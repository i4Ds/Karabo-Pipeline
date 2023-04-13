import os.path
import unittest

from karabo.test import data_path
from karabo.util.FileHandle import FileHandle


class TestFileHandle(unittest.TestCase):
    def test_create(self):
        handle = FileHandle()
        path = handle.path
        self.assertTrue(os.path.exists(path))

    def test_folder(self):
        handle = FileHandle()
        path = handle.path
        self.assertTrue(os.path.exists(path))

    def test_existing_file(self):
        handle = FileHandle(dir=data_path, file_name="detection.csv")
        path = handle.path
        self.assertTrue(os.path.exists(path))

    def test_existing_folder(self):
        handle = FileHandle(
            dir=data_path,
            file_name="detection.csv",
        )
        path = handle.path
        self.assertTrue(os.path.exists(path))
        del handle

    def test_cleanup(self):
        handle = FileHandle()
        path = handle.path
        self.assertTrue(os.path.exists(path))
        handle.clean_up()
        self.assertFalse(os.path.exists(path))
