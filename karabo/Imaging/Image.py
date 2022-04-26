import shutil
import tempfile

import matplotlib
import numpy
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits


class Image:

    def __init__(self):
        """
        Proxy Object Class for Images. Dirty, Cleaned or any other type of image in a fits format
        """
        self.image_data = None
        self.position = tempfile.NamedTemporaryFile()

    def save_as_fits(self, path_with_name: str):
        if not path_with_name.endswith(".fits"):
            raise EnvironmentError("The passed path and name of file must end with .fits")

        shutil.copy(self.position.name, path_with_name)

    def plot_image(self):
        import matplotlib.pyplot as plt
        if self.image_data is None:
            self.__read_fits_data()

        plt.figure()
        squeezed = numpy.squeeze(self.image_data[:1, :1, :, :])  # remove any (1) size dimensions
        # squeezed = squeezed[:1, :, :]
        plt.imshow(squeezed, cmap="rainbow", origin='lower')
        plt.colorbar()
        plt.show()

    def __read_fits_data(self):
        image_file = f"{self.position.name}"
        self.image_data = fits.getdata(image_file, ext=0)
