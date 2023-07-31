import os
import shutil
from pathlib import Path

from karabo.util._types import FilePathType

# from MontagePy.archive import *
# from MontagePy.main import *


def mosaic_directories(output_directory_path: FilePathType) -> None:
    """
    Creating a directory structure which can be used for coadding several fits files
    with MontagePy.

    :param output_directory_path: The new folder (path) which is created and in which
                                  all subfolders used for the mosaic are created.
    """
    output_directory_path = Path(output_directory_path)

    if os.path.exists(output_directory_path):
        shutil.rmtree(output_directory_path)

    os.makedirs(output_directory_path)

    os.makedirs(output_directory_path / "raw")
    os.makedirs(output_directory_path / "projected")
    os.makedirs(output_directory_path / "unused_output")
