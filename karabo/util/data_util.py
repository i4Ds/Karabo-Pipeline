import os

import karabo


def __get_module_absolute_path() -> str:
    path_elements = os.path.abspath(karabo.__file__).split('/')
    path_elements.pop()
    return '/'.join(path_elements)
