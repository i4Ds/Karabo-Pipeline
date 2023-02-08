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
        handle = FileHandle(is_dir=True)
        path = handle.path
        self.assertTrue(os.path.exists(path))

    def test_existing_file(self):
        handle = FileHandle(existing_file_path=f"{data_path}/detection.csv")
        path = handle.path
        self.assertTrue(os.path.exists(path))

    def test_existing_folder(self):
        handle = FileHandle(
            existing_file_path=f"{data_path}/poisson_vis.ms", is_dir=True
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
