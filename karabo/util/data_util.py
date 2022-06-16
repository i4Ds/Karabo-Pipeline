import os

import numpy as np

import karabo


def __get_module_absolute_path() -> str:
    path_elements = os.path.abspath(karabo.__file__).split('/')
    path_elements.pop()
    return '/'.join(path_elements)


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
                    except:
                        value = cell
                    n_row.append(value)
                sources.append(n_row)
    return np.array(sources)
