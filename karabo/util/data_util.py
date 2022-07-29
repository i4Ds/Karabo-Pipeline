import os

import karabo


def get_module_absolute_path() -> str:
    path_elements = os.path.abspath(karabo.__file__).split("/")
    path_elements.pop()
    return "/".join(path_elements)


def get_module_path_of_module(module) -> str:
    path_elements = os.path.abspath(module.__file__).split("/")
    path_elements.pop()
    return "/".join(path_elements)
