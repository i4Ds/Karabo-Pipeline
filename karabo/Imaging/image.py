import logging
import shutil
import uuid

import matplotlib
import numpy
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt

from karabo.resource import KaraboResource
from karabo.util.FileHandle import FileHandle

# store and restore the previously set matplotlib backend, because rascil sets it to Agg (non-GUI)
previous_backend = matplotlib.get_backend()
from rascil.apps.imaging_qa.imaging_qa_diagnostics import power_spectrum

matplotlib.use(previous_backend)


class Image(KaraboResource):

    def __init__(self, name=None):
        """
        Proxy Object Class for Images. Dirty, Cleaned or any other type of image in a fits format
        """
        self.header = None
        self.data = None
        self.name = name
        self.file = FileHandle()
        # self.power_spectrum_profile = None
        # self.power_spectrum_theta_axis = None

    def save_to_file(self, path: str) -> None:
        if not path.endswith(".fits"):
            raise EnvironmentError("The passed path and name of file must end with .fits")

        shutil.copy(self.file.path, path)

    @staticmethod
    def open_from_file(path: str) -> 'Image':
        image = Image()
        image.file = FileHandle(existing_file_path=path, mode='r')
        return image

    # overwrite getter to make sure it always contains the data
    @property
    def data(self) -> npt.NDArray:
        if self._data is None:
            self.__read_fits_data()
        return self._data

    @data.setter
    def data(self, value: npt.NDArray):
        self._data = value

    @property
    def header(self) -> dict:
        if self._header is None:
            self.__read_fits_data()
        return self._header

    @header.setter
    def header(self, value: dict) -> None:
        self._header = value

    def get_squeezed_data(self) -> npt.NDArray:
        return numpy.squeeze(self.data[:1, :1, :, :])

    def plot(self) -> None:
        import matplotlib.pyplot as plt
        wcs = WCS(self.header)
        print(wcs)

        slices = []
        for i in range(wcs.pixel_n_dim):
            if i == 0:
                slices.append('x')
            elif i == 1:
                slices.append('y')
            else:
                slices.append(0)

        plt.subplot(projection=wcs, slices=slices)
        plt.imshow(self.data[0][0], cmap="jet", origin='lower')
        plt.colorbar()
        plt.show()

    def __read_fits_data(self) -> None:
        self.data, self.header = fits.getdata(self.file.path, ext=0, header=True)

    def get_dimensions_of_image(self) -> []:
        """
        Get the sizes of the dimensions of this Image in an array.
        :return: list with the dimensions.
        """
        result = []
        dimensions = self.header["NAXIS"]
        for dim in np.arange(0, dimensions, 1):
            result.append(self.header[f'NAXIS{dim + 1}'])
        return result

    def get_quality_metric(self) -> dict:
        """
        Get image statistics.
        Statistics include :

        - Shape of Image --> 'shape'
        - Max Value --> 'max'
        - Min Value --> 'min'
        - Max Value absolute --> 'max-abs'
        - Root mean square (RMS) --> 'rms'
        - Sum of values --> 'sum'
        - Median absolute --> 'median-abs'
        - Median absolute deviation median --> 'median-abs-dev-median'
        - Median --> 'median'
        - Mean --> 'mean'

        :return: Dictionary holding all image statistics
        """
        # same implementation as RASCIL
        image_stats = {
            "shape": str(self.data.shape),
            "max": np.max(self.data),
            "min": np.min(self.data),
            "max-abs": np.max(np.abs(self.data)),
            "rms": np.std(self.data),
            "sum": np.sum(self.data),
            "median-abs": np.median(np.abs(self.data)),
            "median-abs-dev-median": np.median(np.abs(self.data - np.median(self.data))),
            "median": np.median(self.data),
            "mean": np.mean(self.data),
        }

        return image_stats

    def get_power_spectrum(self, resolution=5.0e-4, signal_channel=None) -> (npt.NDArray, npt.NDArray):
        """
        Calculate the power spectrum of this image.

        :param resolution: Resolution in radians needed for conversion from Jy to Kelvin
        :param signal_channel: channel containing both signal and noise (arr of same shape as nchan of Image), optional
        :return (profile, theta_axis)
            profile: Brightness temperature for each angular scale in Kelvin
            theta_axis: Angular scale data in degrees
        """
        # use RASCIL for power spectrum
        profile, theta = power_spectrum(self.file.path, resolution, signal_channel)
        return profile, theta

    def plot_power_spectrum(self, resolution=5.0e-4, signal_channel=None, save_png=False) -> None:
        """
        Plot the power spectrum of this image.

        :param resolution: Resolution in radians needed for conversion from Jy to Kelvin
        :param signal_channel: channel containing both signal and noise (arr of same shape as nchan of Image), optional
        :param save_png: True if result should be saved, default = False
        """
        profile, theta = self.get_power_spectrum(resolution, signal_channel)
        plt.clf()

        plt.plot(theta, profile)
        plt.gca().set_title(f"Power spectrum of {self.name if self.name is not None else ''} image")
        plt.gca().set_xlabel("Angular scale [degrees]")
        plt.gca().set_ylabel("Brightness temperature [K]")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.gca().set_ylim(1e-6 * numpy.max(profile), 2.0 * numpy.max(profile))
        plt.tight_layout()

        if save_png:
            plt.savefig(f"./power_spectrum_{self.name if self.name is not None else uuid.uuid4()}")
        plt.show()

    def get_cellsize(self):
        cdelt1 = self.header["CDELT1"]
        cdelt2 = self.header["CDELT2"]
        if abs(cdelt1) != abs(cdelt2):
            logging.warning("The Images's cdelt1 and cdelt2 are not the same in absolute value. Continuing with cdelt1")
        return np.deg2rad(np.abs(cdelt1))

    def get_wcs(self):
        return WCS(self.header)

    def project_sky_to_image(self,
                             sky: 'SkyModel',
                             filter_outliers=False) -> (npt.NDArray, npt.NDArray, npt.NDArray):
        """
        Calculates the pixel coordinates of the given sky sources, based on the dimensions passed for a certain image.
        The WCS of this image will be used to transform the sky coordinates.

        :param sky: Sky of which the sources will be projected onto the image plane.
        :param filter_outliers: Exclude source
        :return: pixel-coordinates x-axis, pixel-coordinates y-axis, sky sources indices
        """
        return sky.project_sky_to_image(self, filter_outliers)



    # def plot_histogram(self):
    #
    #     counts, bins, _ = plt.hist(
    #         self.data.flatten(), bins=10, density=False, zorder=5, histtype="step"
    #     )
    #     # # "bins" are the bin edge points, so need the mid points.
    #     # mid_points = bins[:-1] + 0.5 * abs(bins[1:] - bins[:-1])
    #
    #     # # initial guess of the parameters of the fitted gaussian curve
    #     # initial_params = [counts.max(), bdsf_image.raw_mean, bdsf_image.raw_rms]
    #     # optimized_fit_params, _ = optimize.curve_fit(
    #     #     gaussian, mid_points, counts, p0=initial_params
    #     # )
    #     plt.show()
