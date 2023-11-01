from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import NDArray
from rascil.apps.imaging_qa.imaging_qa_diagnostics import power_spectrum
from scipy.interpolate import RegularGridInterpolator

from karabo.karabo_resource import KaraboResource
from karabo.util._types import FilePathType
from karabo.util.file_handler import FileHandler, check_ending
from karabo.util.plotting_util import get_slices

# store and restore the previously set matplotlib backend,
# because rascil sets it to Agg (non-GUI)
previous_backend = matplotlib.get_backend()

matplotlib.use(previous_backend)


class Image(KaraboResource):
    def __init__(
        self,
        path: Optional[str] = None,
        data: Optional[np.ndarray] = None,
        header: Optional[fits.header.Header] = None,
        **kwargs: Any,
    ) -> None:
        self._fh_prefix = "image"
        self._fh_verbose = False
        if path is not None:
            self.path = path
            self.data, self.header = fits.getdata(
                str(self.path),
                ext=0,
                header=True,
                **kwargs,
            )
        elif data is not None and header is not None:
            self.data = data
            self.header = header

            # Generate a random path for the data
            fh = FileHandler.get_file_handler(
                obj=self,
                prefix=self._fh_prefix,
                verbose=self._fh_verbose,
            )
            restored_fits_path = os.path.join(fh.subdir, "image.fits")

            # Write the FITS file
            self.write_to_file(restored_fits_path, overwrite=True)
            self.path = restored_fits_path
        else:
            raise ValueError("Either path or both data and header must be provided.")

        self._fname = os.path.split(self.path)[-1]


    @staticmethod
    def read_from_file(path: FilePathType) -> Image:
        return Image(path=path)

    @property
    def data(self) -> NDArray[np.float_]:
        return self._data

    @data.setter
    def data(self, new_data: NDArray[np.float_]) -> None:
        self._data = new_data
        if hasattr(self, "header"):
            self._update_header_after_resize()

    def write_to_file(
        self,
        path: FilePathType,
        overwrite: bool = False,
    ) -> None:
        """Write an `Image` to `path`  as .fits"""
        check_ending(path=path, ending=".fits")
        dir_name = os.path.dirname(path)
        if dir_name != "" and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fits.writeto(
            filename=str(path),
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
        return np.squeeze(self.data[:1, :1, :, :])

    def resample(
        self,
        shape: Tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        """
        Resamples the image to the given shape using SciPy's RegularGridInterpolator
        for bilinear interpolation. See:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html

        :param shape: The desired shape of the image
        :param kwargs: Keyword arguments for the interpolation function

        """
        new_data = np.empty(
            (self.data.shape[0], 1, shape[0], shape[1]), dtype=self.data.dtype
        )

        for c in range(self.data.shape[0]):
            y = np.arange(self.data.shape[2])
            x = np.arange(self.data.shape[3])
            interpolator = RegularGridInterpolator(
                (y, x),
                self.data[c, 0],
                **kwargs,
            )

            new_x = np.linspace(0, self.data.shape[3] - 1, shape[1])
            new_y = np.linspace(0, self.data.shape[2] - 1, shape[0])
            new_points = np.array(np.meshgrid(new_y, new_x)).T.reshape(-1, 2)
            new_data[c] = interpolator(new_points).reshape(shape[0], shape[1])

        self.data = new_data

    def _update_header_after_resize(self) -> None:
        """Reshape the header to the given shape"""
        old_shape = (self.header["NAXIS2"], self.header["NAXIS1"])
        new_shape = (self.data.shape[2], self.data.shape[3])
        self.header["NAXIS1"] = new_shape[1]
        self.header["NAXIS2"] = new_shape[0]

        self.header["CRPIX1"] = (new_shape[1] + 1) / 2
        self.header["CRPIX2"] = (new_shape[0] + 1) / 2

        self.header["CDELT1"] = self.header["CDELT1"] * old_shape[1] / new_shape[1]
        self.header["CDELT2"] = self.header["CDELT2"] * old_shape[0] / new_shape[0]

    def cutout(self, x_range: Tuple[int, int], y_range: Tuple[int, int]) -> None:
        """
        Cuts out a portion of the image based on the specified x and y ranges and
        returns a copy.

        :param x_range: The start and end coordinates in the x direction
        :param y_range: The start and end coordinates in the y direction
        """
        # Ensure the ranges are within the image dimensions
        x_range = (max(0, x_range[0]), min(self.data.shape[3], x_range[1]))
        y_range = (max(0, y_range[0]), min(self.data.shape[2], y_range[1]))

        # Extract the sub-image
        data = self.data[:, :, y_range[0] : y_range[1], x_range[0] : x_range[1]].copy()
        header = self.header.copy()

        ## Update the header
        # Pass information that the image is a cutout
        header["CUTOUT"] = True

        # Keep the original reference pixel coordinates
        header["OGCRPIX1"] = header["CRPIX1"]
        header["OGCRPIX2"] = header["CRPIX2"]
        header["OGCRVAL1"] = header["CRVAL1"]
        header["OGCRVAL2"] = header["CRVAL2"]

        # Adjust the reference pixel coordinates
        header["CRPIX1"] -= x_range[0]
        header["CRPIX2"] -= y_range[0]

        # Update the CRVAL values to reflect the new reference pixel coordinates
        header["CRVAL1"] = header["CRVAL1"] + (header["CRPIX1"] - header["OGCRPIX1"]) * header["CDELT1"]
        header["CRVAL2"] = header["CRVAL2"] + (header["CRPIX2"] - header["OGCRPIX2"]) * header["CDELT2"]

        # Update the NAXIS values to reflect the new shape
        header["NAXIS1"] = data.shape[3]
        header["NAXIS2"] = data.shape[2]

        # Create path for the cutout image
        cutout_image = Image(None)
        fh = FileHandler.get_file_handler(
            obj=cutout_image,
            prefix=self._fh_prefix,
            verbose=self._fh_verbose,
        )
        restored_fits_path = os.path.join(fh.subdir, "cutout.fits")

        # Save the image
        cutout_image.data = data
        cutout_image.header = header
        cutout_image.path = restored_fits_path

        # Write the updated image to file
        cutout_image.write_to_file(restored_fits_path, overwrite=True)

        return cutout_image

    @staticmethod
    def split_image(image: Image, N: int, overlap: int = 0):
        """
        Splits the image into N*N cutouts and returns a list of the cutouts with optional overlap.
        """
        _, _, x_size, y_size = image.data.shape
        x_step = x_size // N
        y_step = y_size // N

        cutouts = [[None]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                x_start = max(0, i * x_step - overlap)
                x_end = min(x_size, (i + 1) * x_step + overlap)
                y_start = max(0, j * y_step - overlap)
                y_end = min(y_size, (j + 1) * y_step + overlap)

                cut = image.cutout(x_range=[x_start, x_end], y_range=[y_start, y_end])
                cutouts[i][j] = cut
        return cutouts

    def plot(
        self,
        title: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        colobar_label: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        cmap: Optional[str] = "jet",
        origin: Optional[str] = "lower",
        wcs_enabled: bool = True,
        invert_xaxis: bool = False,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Plots the image

        :param title: the title of the colormap
        :param xlim: RA-limit of plot
        :param ylim: DEC-limit of plot
        :param figsize: figsize as tuple
        :param title: plot title
        :param xlabel: xlabel
        :param ylabel: ylabel
        :param cmap: matplotlib color map
        :param origin: place the [0, 0] index of the array in
        the upper left or lower left corner of the Axes
        :param wcs_enabled: Use wcs transformation?
        :param invert_xaxis: Do you want to invert the xaxis?
        :param filename: Set to path/fname to save figure
        (set extension to fname to overwrite .png default)
        :param kwargs: matplotlib kwargs for scatter & Collections,
        e.g. customize `s`, `vmin` or `vmax`
        """

        if wcs_enabled:
            wcs = WCS(self.header)

            slices = get_slices(wcs=wcs)

            # create dummy xlim or ylim if only one is set for conversion
            xlim_reset, ylim_reset = False, False
            if xlim is None and ylim is not None:
                xlim = (-1, 1)
                xlim_reset = True
            elif xlim is not None and ylim is None:
                ylim = (-1, 1)
                ylim_reset = True
            if xlim is not None and ylim is not None:
                xlim, ylim = wcs.wcs_world2pix(xlim, ylim, 0)
            if xlim_reset:
                xlim = None
            if ylim_reset:
                ylim = None

        if wcs_enabled:
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw=dict(projection=wcs, slices=slices)
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(self.data[0][0], cmap=cmap, origin=origin, **kwargs)
        ax.grid()
        fig.colorbar(im, label=colobar_label)

        if title is not None:
            ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if invert_xaxis:
            ax.invert_xaxis()
        if filename is not None:
            fig.savefig(filename)
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
            result.append(self.header[f"NAXIS{dim + 1}"])
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

    def get_quality_metric(self) -> Dict[str, Any]:
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
            "median-abs-dev-median": np.median(
                np.abs(self.data - np.median(self.data))
            ),
            "median": np.median(self.data),
            "mean": np.mean(self.data),
        }

        return image_stats

    def get_power_spectrum(
        self,
        resolution: float = 5.0e-4,
        signal_channel: Optional[int] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Calculate the power spectrum of this image.

        :param resolution: Resolution in radians needed for conversion from Jy to Kelvin
        :param signal_channel: channel containing both signal and noise
        (arr of same shape as nchan of Image), optional
        :return (profile, theta_axis)
            profile: Brightness temperature for each angular scale in Kelvin
            theta_axis: Angular scale data in degrees
        """
        profile, theta = power_spectrum(self.path, resolution, signal_channel)
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
        :param signal_channel: channel containing both signal and noise
        (arr of same shape as nchan of Image), optional
        :param save_png: True if result should be saved, default = False
        """
        profile, theta = self.get_power_spectrum(resolution, signal_channel)
        plt.clf()

        plt.plot(theta, profile)
        plt.gca().set_title(
            f"Power spectrum of {self._fname if self._fname is not None else ''} image"
        )
        plt.gca().set_xlabel("Angular scale [degrees]")
        plt.gca().set_ylabel("Brightness temperature [K]")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.gca().set_ylim(1e-6 * np.max(profile), 2.0 * np.max(profile))
        plt.tight_layout()

        if save_png:
            power_spectrum_name = (
                self._fname if self._fname is not None else uuid.uuid4()
            )
            plt.savefig(f"./power_spectrum_{power_spectrum_name}")
        plt.show(block=False)
        plt.pause(1)

    def get_cellsize(self) -> np.float64:
        cdelt1 = self.header["CDELT1"]
        cdelt2 = self.header["CDELT2"]
        if not isinstance(cdelt1, float) or not isinstance(cdelt2, float):
            raise ValueError(
                "CDELT1 & CDELT2 in header are expected to be of type float."
            )
        if np.abs(cdelt1) != np.abs(cdelt2):
            logging.warning(
                "Non-square pixels are not supported, continue with `cdelt1`."
            )
        cellsize = cast(np.float64, np.deg2rad(np.abs(cdelt1)))
        return cellsize

    def get_wcs(self) -> WCS:
        return WCS(self.header)

    def get_2d_wcs(
        self,
        invert_ra: bool = True,
    ) -> WCS:
        wcs = WCS(naxis=2)

        def radian_degree(rad: np.float64) -> np.float64:
            return rad * (180 / np.pi)

        cdelt = radian_degree(self.get_cellsize())
        crpix = np.floor((self.get_dimensions_of_image()[0] / 2)) + 1
        wcs.wcs.crpix = np.array([crpix, crpix])
        ra_sign = -1 if invert_ra else 1
        wcs.wcs.cdelt = np.array([ra_sign * cdelt, cdelt])
        wcs.wcs.crval = [self.header["CRVAL1"], self.header["CRVAL2"]]
        wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]  # coordinate axis type
        return wcs
