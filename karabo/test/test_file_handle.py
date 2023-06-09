import os.path
import unittest

from karabo.test import data_path
from karabo.util.file_handle import FileHandle


class TestFileHandle(unittest.TestCase):
    def test_create_folder(self):
        handle = FileHandle()
        dir = handle.dir
        self.assertTrue(os.path.exists(dir))
        handle.clean_up()
        self.assertFalse(os.path.exists(handle.path))

    def test_existing_folder(self):
        handle = FileHandle(file_is_dir=True)
        path = handle.path
        self.assertTrue(os.path.exists(path))

        handle = FileHandle(dir=path)
        dir = handle.dir
        self.assertTrue(os.path.exists(dir))

    def test_existing_file(self):
        handle = FileHandle(path=data_path, file_name="detection.csv")
        path = handle.path
        self.assertTrue(os.path.exists(path))

    def test_cleanup(self):
        handle = FileHandle(create_additional_folder_in_dir=True)
        dir = handle.dir
        self.assertTrue(os.path.exists(dir))
        handle.clean_up()
        self.assertFalse(os.path.exists(handle.path))

    def test_correct_path_creation(self):
        path = os.path.join(data_path, "test_123.ms")
        handle = FileHandle(path=path)
        self.assertEqual(handle.path, path)
        self.assertEqual(handle.dir, data_path)

    def test_correct_file_location(self):
        dir = os.path.join(data_path)
        handle = FileHandle(dir=dir, file_name="test_123.ms")
        self.assertEqual(handle.path, os.path.join(dir, "test_123.ms"))
        self.assertEqual(handle.dir, data_path)

    def test_folder_creation_in_folder(self):
        dir = os.path.join(data_path, "test_123")
        if not os.path.exists(dir):
            os.mkdir(dir)
        handle = FileHandle(dir=dir, create_additional_folder_in_dir=True)
        self.assertEqual(dir, os.path.split(handle.dir)[0])
        handle.clean_up()
        self.assertTrue(os.path.exists(dir))

    def test_pass_ms_set_path(self):
        dir = os.path.join(data_path)
        handle = FileHandle(dir=dir, suffix=".ms")
        self.assertEqual(handle.dir, dir)
        # Check that in path there is a .ms file
        self.assertTrue(".ms" in handle.path)
        # Check that that path exist
        self.assertTrue(os.path.exists(handle.path))
