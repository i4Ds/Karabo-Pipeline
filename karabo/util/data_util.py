import os

import numpy as np

import karabo

from astropy.io import fits

from numpy.typing import NDArray

from typing import List, Dict, Any, Tuple

def get_module_absolute_path() -> str:
    path_elements = os.path.abspath(karabo.__file__).split("/")
    path_elements.pop()
    return "/".join(path_elements)


def get_module_path_of_module(module) -> str:
    path_elements = os.path.abspath(module.__file__).split("/")
    path_elements.pop()
    return "/".join(path_elements)

def image_header_has_parameters(image, parameters):
    fitsfile=fits.open(image.file.path)
    header=fitsfile[0].header
    for parameter in parameters:
        if parameter not in header:
            return False
    return True

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


def Gauss(x, x0, y0, a,  sigma):
    return y0 + a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def Voigt(x, x0, y0, a, sigma, gamma):
    # sigma = alpha / np.sqrt(2 * np.log(2))
    return y0 + a * np.real(wofz((x - x0 + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)

def get_spectral_sky_data(
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    freq0: NDArray[np.float64],
    nfreq: int,
) -> NDArray[np.float64]:
    dfreq_arr = np.linspace(-0.1, 0.1, 100)
    y_voigt = Voigt(dfreq_arr, 0, 0, 1, 0.01, 0.01)
    y_gauss = Gauss(dfreq_arr, 0, 0, 1, 0.01)
    dfreq_sample = dfreq_arr[::nfreq]
    flux_sample = y_voigt[::nfreq]
    freq_sample = freq0+dfreq_sample * freq0
    sky_data = np.zeros((nfreq,12))
    sky_data[:,0] = ra
    sky_data[:,1] = dec
    sky_data[:,2] = flux_sample
    sky_data[:,6] = freq_sample
    sky_data[:,7] = -200
    return sky_data

def resample_spectral_lines(
    npoints: int,
    dfreq: NDArray[np.float64],
    spec_line: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    m = int(len(dfreq) / npoints)
    dfreq_sampled = dfreq[::m]
    line_sampled = spec_line[::m]
    return dfreq_sampled, line_sampled


def full_setter(self, state):
    self.__dict__ = state


def full_getter(self):
    state = self.__dict__
    return state
