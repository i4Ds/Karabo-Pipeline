import os
import tempfile
import uuid

from distributed import Client

from karabo.util.dask import get_global_client


class FileHandle:
    # def __init__(self, existing_file_path: str = None, is_dir: bool = False):
    #     if is_dir:
    #         self.file = tempfile.TemporaryDirectory()
    #         self.path = self.file.name
    #     else:
    #         if existing_file_path is not None:
    #             if os.path.isdir(existing_file_path):
    #                 self.file = None
    #                 self.path = existing_file_path
    #             else:
    #                 self.file = open(existing_file_path)
    #                 self.path = existing_file_path
    #         else:
    #             self.file = tempfile.NamedTemporaryFile()
    #             self.path = self.file.name
    path: str
    existing: bool = False
    is_dir = False

    __temp_path = './.tmp/'

    def __init__(self, existing_file_path: str = None, is_dir: bool = False):
        unique_path = self.__temp_path + str(uuid.uuid4())
        if existing_file_path:
            unique_path = existing_file_path
            self.existing = True

        # make temp folder if not present
        if not os.path.exists(self.__temp_path):
            os.mkdir(self.__temp_path)

        if is_dir and not self.existing:
            os.mkdir(unique_path)
            self.file = None
            self.path = unique_path
            self.is_dir = True

        else:
            self.file = open(unique_path, "x")
            self.path = unique_path

    # def __del__(self):
    #     if self.existing:
    #         return
    #
    #     if self.is_dir:
    #         os.rmdir(self.path)
    #     if not self.is_dir:
    #

    def copy_to_dask_worker(self):
        client = get_global_client()
        client.wait_for_workers(client, 3)
        client.upload_file(self.path)
