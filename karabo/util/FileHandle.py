import os
import shutil
import tempfile
import uuid

from distributed import Client

from karabo.util.dask import get_global_client


class FileHandle:
    path: str
    __temp_path = './.tmp/'

    def __init__(self,
                 existing_file_path: str = None,
                 is_dir: bool = False,
                 mode="rt",
                 suffix=""):

        tmp_path = self.__temp_path + str(uuid.uuid4()) + suffix

        # make temp folder if not present
        if not os.path.exists(self.__temp_path):
            os.mkdir(self.__temp_path)

        if existing_file_path:
            # existing
            if is_dir:
                # is a directory
                self.path = tmp_path
                os.mkdir(tmp_path)
                shutil.copytree(existing_file_path, tmp_path)
            else:
                # is a file
                open(tmp_path, mode)
                self.path = tmp_path
        else:
            # not existing
            if is_dir:
                # is a directory
                self.path = tmp_path
                os.mkdir(tmp_path)
            else:
                # is a file
                open(tmp_path, "x")
                self.path = tmp_path

    # def __del__(self):
    #     if self.existing:
    #         return
    #
    #     if self.is_dir:
    #         os.rmdir(self.path)
    #     if not self.is_dir:
    #
