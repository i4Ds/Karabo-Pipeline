import os
import shutil
import uuid

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
                shutil.copytree(existing_file_path, tmp_path)
            else:
                # is a file
                open(tmp_path, 'x')
                shutil.copyfile(existing_file_path, tmp_path)
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

    def __del__(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        else:
            os.remove(self.path)
        # remove temp dir if it was the last temporary file
        if len(os.listdir(self.__temp_path)) == 0:
            os.rmdir(self.__temp_path)