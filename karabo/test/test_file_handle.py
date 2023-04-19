import os.path
import unittest

from karabo.test import data_path
from karabo.util.FileHandle import FileHandle


class TestFileHandle(unittest.TestCase):
    def test_create_file(self):
        handle = FileHandle(file_name="file.file")
        path = handle.path
        dir_path = os.path.dirname(path)
        self.assertTrue(os.path.exists(dir_path))
        handle.clean_up()
        self.assertFalse(os.path.exists(dir_path))

    def test_create_file_w_suffix(self):
        handle = FileHandle(file_name="file", suffix=".file")
        path = handle.path
        dir_path = os.path.dirname(path)
        self.assertTrue(os.path.exists(dir_path))
        handle.clean_up()
        self.assertFalse(os.path.exists(dir_path))

    def test_create_folder(self):
        handle = FileHandle()
        path = handle.path
        self.assertTrue(os.path.exists(path))
        handle.clean_up()
        self.assertFalse(os.path.exists(path))

    def test_existing_folder(self):
        handle = FileHandle()
        path = handle.path
        self.assertTrue(os.path.exists(path))

        handle = FileHandle(path=path)
        path = handle.path
        self.assertTrue(os.path.exists(path))
        handle.clean_up()
        self.assertFalse(os.path.exists(path))

    def test_existing_file(self):
        handle = FileHandle(path=data_path, file_name="detection.csv")
        path = handle.path
        self.assertTrue(os.path.exists(path))

    def test_cleanup(self):
        handle = FileHandle()
        path = handle.path
        self.assertTrue(os.path.exists(path))
        handle.clean_up()
        self.assertFalse(os.path.exists(path))

    def test_correct_path_creation(self):
        path = os.path.join(data_path, "test_123.ms")
        handle = FileHandle(path=path)
        self.assertEqual(handle.path, path)
        self.assertEqual(handle.dir, path)

    def test_correct_file_location(self):
        path = os.path.join(data_path)
        handle = FileHandle(path=path, file_name="test_123.ms")
        self.assertEqual(handle.path, os.path.join(path, "test_123.ms"))
        self.assertEqual(handle.dir, data_path)

    def test_folder_creation_in_folder(self):
        path = os.path.join(data_path, "test_123")
        if not os.path.exists(path):
            os.mkdir(path)
        handle = FileHandle(path=path, create_additional_folder_in_dir=True)
        self.assertEqual(path, os.path.split(handle.path)[0])
        handle.clean_up()
        self.assertTrue(os.path.exists(path))
