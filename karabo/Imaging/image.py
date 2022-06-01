import shutil

import numpy
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from karabo.util.FileHandle import FileHandle


class Image:

    def __init__(self):
        """
        Proxy Object Class for Images. Dirty, Cleaned or any other type of image in a fits format
        """
        self.data = None
        self.header = None
        self.file = FileHandle()

    # overwrite getter to make sure it always contains the data
    @property
    def data(self):
        if self._data is None:
            self.__read_fits_data()
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def get_squeezed_data(self):
        return numpy.squeeze(self.data[:1, :1, :, :])

    def save_as_fits(self, path_with_name: str):
        if not path_with_name.endswith(".fits"):
            raise EnvironmentError("The passed path and name of file must end with .fits")

        shutil.copy(self.file.path, path_with_name)

    def plot(self):
        import matplotlib.pyplot as plt
        wcs = WCS(self.header)
        print(wcs)

        plt.subplot(projection=wcs, slices=('y', 'x'))
        squeezed = numpy.squeeze(self.data[:1, :1, :, :])  # remove any (1) size dimensions
        plt.imshow(squeezed, cmap="jet", origin='lower')
        plt.colorbar()
        plt.show()

    def __read_fits_data(self):
        self.data, self.header = fits.getdata(self.file.path, ext=0, header=True)

    def get_dimensions_of_image(self) -> []:
        """
        Get the sizes of the dimensions of this Image in an array.
        :return: list with the dimensions.
        """
        result = []
        dimensions = self.header["NAXIS"]
        for dim in np.arange(0, dimensions, 1):
            result.append(f'NAXIS${dim}')
        return result


def open_fits_image(fits_path: str) -> Image:
    image = Image()
    image.file = FileHandle(existing_file_path=fits_path, auto_clean=False)
    return image
