import shutil

import numpy
from astropy.io import fits
from karabo.util.FileHandle import FileHandle


class Image:

    def __init__(self):
        """
        Proxy Object Class for Images. Dirty, Cleaned or any other type of image in a fits format
        """
        self.data = None
        self.header = None
        self.file = FileHandle()

    def get_squeezed_data(self):
        if self.data is None:
            self.__read_fits_data()
        return numpy.squeeze(self.data[:1, :1, :, :])

    def save_as_fits(self, path_with_name: str):
        if not path_with_name.endswith(".fits"):
            raise EnvironmentError("The passed path and name of file must end with .fits")

        shutil.copy(self.file.path, path_with_name)

    def plot(self):
        import matplotlib.pyplot as plt
        if self.data is None:
            self.__read_fits_data()

        plt.figure()
        squeezed = numpy.squeeze(self.data[:1, :1, :, :])  # remove any (1) size dimensions
        # squeezed = squeezed[:1, :, :]
        plt.imshow(squeezed, cmap="rainbow", origin='lower')
        plt.colorbar()
        plt.show()

    def __read_fits_data(self):
        image_file = f"{self.file.path}"
        self.data, self.header = fits.getdata(image_file, ext=0, header=True)




def open_fits_image(fits_path: str) -> Image:
    image = Image()
    image.file = FileHandle(existing_file_path=fits_path, auto_clean=False)
    return image
