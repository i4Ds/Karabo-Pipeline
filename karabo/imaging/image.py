from __future__ import annotations
import logging, os, shutil, uuid
from typing import Tuple, Dict, List, Any, Optional

import matplotlib
import numpy
import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

from karabo.karabo_resource import KaraboResource
from karabo.util.FileHandle import FileHandle, check_ending

# store and restore the previously set matplotlib backend, because rascil sets it to Agg (non-GUI)
previous_backend = matplotlib.get_backend()
from rascil.apps.imaging_qa.imaging_qa_diagnostics import power_spectrum

matplotlib.use(previous_backend)


class Image(KaraboResource):

    def __init__(
        self,
        path: str,
        **kwargs,
    ) -> None:
        """
        Proxy Object Class for Images. Dirty, Cleaned or any other type of image in a fits format
        """
        self.__name = path.split(os.path.sep)[-1]
        self.file = FileHandle(existing_file_path=path, mode='r')
        self.data, self.header = fits.getdata(self.file.path, ext=0, header=True, **kwargs)

    @staticmethod
    def read_from_file(path: str) -> Image:
        return Image(path=path)

    def copy_image_file_to(self, path: str) -> None:
        """
        Makes a copy the .fits file to `path`.
        Pay attention, this doesn't copy the current state of `Image` to `path` if the `Image` was altered,
            it just creates a copy of the current .fits file of this `Image` to `path`.
        """
        check_ending(path=path, ending='.fits')
        shutil.copy(self.file.path, path)

    def export_image_to(
        self,
        path: str,
        overwrite: bool = False,
    ) -> None:
        """Write an `Image` to `path`  as .fits"""
        check_ending(path=path, ending='.fits')
        fits.writeto(
            filename=path,
            data=self.data,
            header=self.header,
            overwrite=overwrite,
        )

    def header_has_parameters(
        self,
        parameters: List[str],
    ) -> bool:
        for parameter in parameters:
            if parameter not in self.header:
                return False
        return True

    def get_squeezed_data(self) -> NDArray[np.float64]:
        return numpy.squeeze(self.data[:1, :1, :, :])

    def plot(
        self,
        title: str,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        plot_title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        cmap: Optional[str] = "jet",
        origin: Optional[str] = 'lower',
        wcs_enabled: bool = True,
        invert_xaxis: bool = True,
        filename: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Plots the image

        :param title: the title of the colormap
        :param xlim: RA-limit of plot
        :param ylim: DEC-limit of plot
        :param figsize: figsize as tuple
        :param plot_title: plot title
        :param xlabel: xlabel
        :param ylabel: ylabel
        :param cmap: matplotlib color map
        :param origin: place the [0, 0] index of the array in the upper left or lower left corner of the Axes
        :param wcs_enabled: Use wcs transformation?
        :param invert_xaxis: Do you want to invert the xaxis?
        :param filename: Set to path/fname to save figure (set extension to fname to overwrite .png default)
        :param kwargs: matplotlib kwargs for scatter & Collections, e.g. customize `s`, `vmin` or `vmax`
        """
        if wcs_enabled:
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

            # create dummy xlim or ylim if only one is set for conversion
            xlim_reset, ylim_reset = False, False
            if xlim is None and ylim is not None:
                xlim = (-1,1)
                xlim_reset = True
            elif xlim is not None and ylim is None:
                ylim = (-1,1)
                ylim_reset = True
            if xlim is not None and ylim is not None:
                xlim, ylim = wcs.wcs_world2pix(xlim, ylim, 0)
            if xlim_reset: xlim = None
            if ylim_reset: ylim = None

        if wcs_enabled:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=wcs, slices=slices))
        else:
            fig, ax = plt.subplots(figsize=figsize)

        plt.imshow(self.data[0][0], cmap=cmap, origin=origin, **kwargs)
        plt.colorbar(label=title)
        if plot_title is not None: plt.title(plot_title)
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if invert_xaxis: ax.invert_xaxis()
        if filename is not None: plt.savefig(filename)
        plt.show(block=False)
        plt.pause(1)

    def get_dimensions_of_image(self) -> List[int]:
        """
        Get the sizes of the dimensions of this Image in an array.
        :return: list with the dimensions.
        """
        result = []
        dimensions = self.header["NAXIS"]
        for dim in np.arange(0, dimensions, 1):
            result.append(self.header[f'NAXIS{dim + 1}'])
        return result

    def get_phase_center(self) -> Tuple[float, float]:
        return float(self.header["CRVAL1"]), float(self.header["CRVAL2"])

    def has_beam_parameters(self) -> bool:
        """
        Check if the image has the beam parameters in the header.
        :param image: Image to check
        :return: True if the image has the beam parameters in the header
        """
        return self.header_has_parameters(
            ["BMAJ", "BMIN", "BPA"],
        )

    def get_quality_metric(self) -> Dict[str,Any]:
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

    def get_power_spectrum(
        self,
        resolution: float = 5.0e-4,
        signal_channel: Optional[int] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.floating]]:
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

    def plot_power_spectrum(
        self,
        resolution: float = 5.0e-4,
        signal_channel: Optional[int] = None,
        save_png: bool = False,
    ) -> None:
        """
        Plot the power spectrum of this image.

        :param resolution: Resolution in radians needed for conversion from Jy to Kelvin
        :param signal_channel: channel containing both signal and noise (arr of same shape as nchan of Image), optional
        :param save_png: True if result should be saved, default = False
        """
        profile, theta = self.get_power_spectrum(resolution, signal_channel)
        plt.clf()

        plt.plot(theta, profile)
        plt.gca().set_title(f"Power spectrum of {self.__name if self.__name is not None else ''} image")
        plt.gca().set_xlabel("Angular scale [degrees]")
        plt.gca().set_ylabel("Brightness temperature [K]")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.gca().set_ylim(1e-6 * numpy.max(profile), 2.0 * numpy.max(profile))
        plt.tight_layout()

        if save_png:
            plt.savefig(f"./power_spectrum_{self.__name if self.__name is not None else uuid.uuid4()}")
        plt.show(block=False)
        plt.pause(1)

    def get_cellsize(self) -> float:
        cdelt1 = self.header["CDELT1"]
        cdelt2 = self.header["CDELT2"]
        if abs(cdelt1) != abs(cdelt2):
            logging.warning("The Images's cdelt1 and cdelt2 are not the same in absolute value. Continuing with cdelt1")
        return np.deg2rad(np.abs(cdelt1))

    def get_wcs(self) -> WCS:
        return WCS(self.header)

    def get_2d_wcs(
        self,
        invert_ra: bool = True,
    ) -> WCS:
        wcs = WCS(naxis=2)
        radian_degree = lambda rad: rad * (180 / np.pi)
        cdelt = radian_degree(self.get_cellsize())
        crpix = np.floor((self.get_dimensions_of_image()[0] / 2)) + 1
        wcs.wcs.crpix = np.array([crpix, crpix])
        ra_sign = -1 if invert_ra else 1
        wcs.wcs.cdelt = np.array([ra_sign*cdelt, cdelt])
        wcs.wcs.crval = [self.header["CRVAL1"], self.header["CRVAL2"]]
        wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]  # coordinate axis type
        return wcs
