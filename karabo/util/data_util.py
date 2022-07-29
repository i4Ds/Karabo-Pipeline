import os

import numpy as np

import karabo


def get_module_absolute_path() -> str:
    path_elements = os.path.abspath(karabo.__file__).split("/")
    path_elements.pop()
    return "/".join(path_elements)


def get_module_path_of_module(module) -> str:
    path_elements = os.path.abspath(module.__file__).split("/")
    path_elements.pop()
    return "/".join(path_elements)


def read_CSV_to_ndarray(file: str) -> np.ndarray:
    import csv
    sources = []
    with open(file, newline='') as sourcefile:
        spamreader = csv.reader(sourcefile, delimiter=',', quotechar='|')
        for row in spamreader:
            if len(row) == 0:
                continue
            if row[0].startswith("#"):
                continue
            else:
                n_row = []
                for cell in row:
                    try:
                        value = float(cell)
                        n_row.append(value)
                    except ValueError:
                        pass
                sources.append(n_row)
    return np.array(sources, dtype=float)


def full_setter(self, state):
    self.__dict__ = state


def full_getter(self):
    state = self.__dict__
    return state
