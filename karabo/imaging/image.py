from __future__ import annotations

import logging
import os
import warnings
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.io.fits.header import Header
from astropy.nddata import Cutout2D, NDData
from astropy.wcs import WCS
from numpy.typing import NDArray
from rascil.apps.imaging_qa.imaging_qa_diagnostics import power_spectrum
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
from scipy.interpolate import RegularGridInterpolator

from karabo.simulation.sky_model import SkyModel
from karabo.util._types import BeamType, FilePathType
from karabo.util.file_handler import FileHandler, assert_valid_ending
from karabo.util.plotting_util import get_slices

# store and restore the previously set matplotlib backend,
# because rascil sets it to Agg (non-GUI)
previous_backend = matplotlib.get_backend()

matplotlib.use(previous_backend)


class Image:
    @overload
    def __init__(
        self,
        *,
        path: FilePathType,
        data: Literal[None] = None,
        header: Literal[None] = None,
        **kwargs: Any,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        *,
        path: Literal[None] = None,
        data: NDArray[np.float_],
        header: Header,
        **kwargs: Any,
    ) -> None:
        ...

    def __init__(
        self,
        *,
        path: Optional[FilePathType] = None,
        data: Optional[NDArray[np.float_]] = None,
        header: Optional[Header] = None,
        **kwargs: Any,
    ) -> None:
        self.header: Header
        if path is not None and (data is None and header is None):
            self.path = path
            self.data, self.header = fits.getdata(
                str(self.path),
                ext=0,
                header=True,
                **kwargs,
            )
        elif path is None and (data is not None and header is not None):
            self.data = data
            self.header = header

            tmp_dir = FileHandler().get_tmp_dir(
                prefix="Image-",
                purpose="restored fits-path",
                unique=self,
            )

            restored_fits_path = os.path.join(tmp_dir, "image.fits")

            # Write the FITS file
            self.write_to_file(restored_fits_path)
            self.path = restored_fits_path
        else:
            raise RuntimeError("Provide either `path` or both `data` and `header`.")

        if self.data.ndim not in (2, 3, 4):
            raise ValueError(
                f"""Unexpected shape for image data:
            {self.data.shape}; expected 2D, 3D or (ideally) 4D array. Ideal image shape:
            (frequencies, polarisations, pixels_x, pixels_y)"""
            )

        if self.data.ndim == 2:
            warnings.warn(
                """Received 2D data for image object.
                Will assume the 2 axes correspond to (pixels_x, pixels_y).
                Inserting 2 additional axes for frequencies and polarisations."""
            )
            self.data = np.array([[self.data]])
        elif self.data.ndim == 3:
            warnings.warn(
                """Received 3D data for image object.
                Will assume the 3 axes correspond to
                (polarisations, pixels_x, pixels_y).
                Inserting 1 additional axis for frequencies."""
            )
            self.data = np.array([self.data])

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
        assert_valid_ending(path=path, ending=".fits")
        dir_name = os.path.abspath(os.path.dirname(path))
        os.makedirs(dir_name, exist_ok=True)
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

    def cutout(
        self, center_xy: Tuple[float, float], size_xy: Tuple[float, float]
    ) -> Image:
        """
        Cutout the image to the given size and center.

        :param center_xy: Center of the cutout in pixel coordinates
        :param size_xy: Size of the cutout in pixel coordinates
        :return: Cutout of the image
        """
        cut = Cutout2D(
            self.data[0, 0, :, :],
            center_xy,
            size_xy,
            wcs=self.get_2d_wcs(),
        )
        header = cut.wcs.to_header()
        header = self.update_header_from_image_header(header, self.header)
        return Image(data=cut.data[np.newaxis, np.newaxis, :, :], header=header)

    def circle(self) -> None:
        """For each frequency channel and polarisation, cutout the pixel values,
        only keeping data for a circle of the computed radius, centered
        at the center of the image.
        This is an in-place transformation of the data.

        :return: None (data of current Image instance is transformed in-place)
        """

        def circle_pixels(pixels: NDArray[np.float_]) -> NDArray[np.float_]:
            radius = min(pixels.shape) // 2
            y, x = np.ogrid[-radius:radius, -radius:radius]
            mask = x**2 + y**2 > radius**2
            pixels[mask] = np.nan

            return pixels

        # This assumes self.data is a 4D array, with shape corresponding to
        # (frequency, polarisations, pixels_x, pixels_y)
        for i, frequency_image in enumerate(self.data):
            for j, pixels in enumerate(frequency_image):
                self.data[i][j] = circle_pixels(pixels)

    @staticmethod
    def update_header_from_image_header(
        new_header: Header,
        old_header: Header,
        keys_to_copy: Optional[List[str]] = None,
    ) -> Header:
        if keys_to_copy is None:
            keys_to_copy = [
                "CTYPE3",
                "CRPIX3",
                "CDELT3",
                "CRVAL3",
                "CUNIT3",
                "CTYPE4",
                "CRPIX4",
                "CDELT4",
                "CRVAL4",
                "CUNIT4",
                "BMAJ",
                "BMIN",
                "BPA",
            ]
        for key in keys_to_copy:
            if key in old_header and key not in new_header:
                new_header[key] = old_header[key]
        return new_header

    def split_image(self, N: int, overlap: int = 0) -> List[Image]:
        """
        Split the image into N x N equal-sized sections with optional overlap.

        Parameters
        ----------
        N : int
            The number of sections to split the image into along one axis. The
            total number of image sections will be N^2.
            It is assumed that the image can be divided into N equal parts along
            both axes. If this is not the case (e.g., image size is not a
            multiple of N), the sections on the edges will have fewer pixels.

        overlap : int, optional
            The number of pixels by which adjacent image sections will overlap.
            Default is 0, meaning no overlap. Negative overlap means that there
            will be empty sections between the cutouts.

        Returns
        -------
        cutouts : list
            A list of cutout sections of the image. Each element in the list is
            a 2D array representing a section of the image.

        Notes
        -----
        The function calculates the step size for both the x and y dimensions
        by dividing the dimension size by N. It then iterates over N steps
        in both dimensions to generate starting and ending indices for each
        cutout section, taking the overlap into account.

        The `cutout` function (not shown) is assumed to take the center
        (x, y) coordinates and the (width, height) of the desired cutout
        section and return the corresponding 2D array from the image.

        The edge sections will be equal to or smaller than the sections in
        the center of the image if the image size is not an exact multiple of N.

        Examples
        --------
        >>> # Assuming `self.data` is a 4D array with shape (C, Z, X, Y)
        >>> # and `self.cutout` method is defined
        >>> image = Image()
        >>> cutouts = image.split_image(4, overlap=10)
        >>> len(cutouts)
        16  # because 4x4 grid
        """
        if N < 1:
            raise ValueError("N must be >= 1")
        _, _, x_size, y_size = self.data.shape
        x_step = x_size // N
        y_step = y_size // N

        cutouts = []
        for i in range(N):
            for j in range(N):
                x_start = max(0, i * x_step - overlap)
                x_end = min(x_size, (i + 1) * x_step + overlap)
                y_start = max(0, j * y_step - overlap)
                y_end = min(y_size, (j + 1) * y_step + overlap)

                center_x = (x_start + x_end) // 2
                center_y = (y_start + y_end) // 2
                size_x = x_end - x_start
                size_y = y_end - y_start

                cut = self.cutout((center_x, center_y), (size_x, size_y))
                cutouts.append(cut)
        return cutouts

    def to_NNData(self) -> NDData:
        return NDData(data=self.data, wcs=self.get_wcs())

    def to_2dNNData(self) -> NDData:
        # TODO this assumes stokesI, i.e. 0th polarisation
        # Need to modify when adding full-stokes support
        return NDData(data=self.data[0, 0, :, :], wcs=self.get_2d_wcs())

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
        block: bool = False,
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
        :param block: Whether plotting should block the remaining of the script
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
        plt.show(block=block)
        plt.pause(1)

    def overplot_with_skymodel(
        self,
        sky: SkyModel,
        filename: Optional[FilePathType] = None,
        block: bool = False,
        channel_index: int = 0,
        stokes_index: int = 0,
        vmin_image: Optional[float] = None,
        vmax_image: Optional[float] = None,
    ) -> None:
        """Create a plot with the current image data,
        as well as an overlay of sources from a given SkyModel instance.

        :param sky: a SkyModel instance, with sources to be plotted.
        :param filename: path to the file where the final plot will be saved.
            If None, the plot is not saved.
        :param block: whether plotting should block the remaining of the script.
        :param channel_index: Which frequency channel to show in the plot.
            Defaults to 0.
        :param stokes_index: Which polarisation to show in the plot.
            Defaults to 0 (stokesI).
        :param vmin_image, vmax_image: Limits for colorbar of Image plot.
        """
        # wcs.wcs_world2pix expects a FITS header with only 2 coordinates (x, y).
        # For this plot, we temporarily remove the 3rd and 4th axes from the image
        # Per suggestion from:
        # https://github.com/aplpy/aplpy/issues/423#issuecomment-848170880
        two_dimensional_header = deepcopy(self.header)
        two_dimensional_header["NAXIS"] = 2
        two_dimensional_header["WCSAXES"] = 2
        temporary_data = self.data[channel_index][stokes_index]

        for kw in ("CTYPE", "CROTA", "CRVAL", "CRPIX", "CDELT", "CUNIT", "NAXIS"):
            for n in (3, 4):
                k = f"{kw}{n}"
                if k in two_dimensional_header.keys():
                    two_dimensional_header.remove(k)

        wcs = WCS(two_dimensional_header)
        slices = get_slices(wcs)

        px, py = wcs.wcs_world2pix(sky[:, 0], sky[:, 1], 0)

        # Create the figure and axes
        plt.figure(figsize=(20, 10))
        ax = plt.subplot(111, projection=wcs, slices=slices)
        img = ax.imshow(
            temporary_data,
            cmap="YlGnBu",
            origin="lower",
            vmin=vmin_image,
            vmax=vmax_image,
        )
        ax.scatter(px, py, facecolors="none", edgecolors="r", s=20)
        plt.colorbar(img, ax=ax, label="Flux Density [Jy]")
        ax.set_xlim((0, temporary_data.shape[1]))
        ax.set_ylim((0, temporary_data.shape[0]))
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show(block=block)

    def plot_side_by_side_with_skymodel(
        self,
        sky: SkyModel,
        filename: Optional[FilePathType] = None,
        block: bool = False,
        channel_index: int = 0,
        stokes_index: int = 0,
        vmin_sky: float = 0,
        vmax_sky: float = np.inf,
        vmin_image: float = 0,
        vmax_image: float = np.inf,
    ) -> None:
        """Create a plot with two panels:
        1. the current image data, and
        2. a scatter plot of sources from a given SkyModel instance.

        :param sky: a SkyModel instance, with sources to be plotted.
        :param filename: path to the file where the final plot will be saved.
            If None, the plot is not saved.
        :param block: whether plotting should block the remaining of the script.
        :param channel_index: Which frequency channel to show in the plot.
            Defaults to 0.
        :param stokes_index: Which polarisation to show in the plot.
            Defaults to 0 (stokesI).
        :param vmin_sky, vmax_sky: Limits for colorbar of SkyModel scatter plot.
        :param vmin_image, vmax_image: Limits for colorbar of Image plot.
        """
        wcs = WCS(self.header)
        slices = get_slices(wcs)

        fig = plt.figure(figsize=(12, 6))

        # Left panel: scatter plot of SkyModel sources
        ax1 = fig.add_subplot(121)
        scatter = ax1.scatter(
            sky[:, 0],
            sky[:, 1],
            c=sky[:, 2],
            s=10,
            cmap="jet",
            vmin=vmin_sky,
            vmax=vmax_sky,
        )
        ax1.set_aspect("equal")
        plt.colorbar(scatter, ax=ax1, label="Flux [Jy]")
        ra_deg = self.header["CRVAL1"]
        dec_deg = self.header["CRVAL2"]
        img_size_ra = self.header["NAXIS1"]
        img_size_dec = self.header["NAXIS2"]
        cut_ra = -self.header["CDELT1"] * float(img_size_ra)
        cut_dec = self.header["CDELT2"] * float(img_size_dec)
        ax1.set_xlim((ra_deg - cut_ra / 2, ra_deg + cut_ra / 2))
        ax1.set_ylim((dec_deg - cut_dec / 2, dec_deg + cut_dec / 2))
        ax1.set_xlabel("RA [deg]")
        ax1.set_ylabel("DEC [deg]")
        ax1.invert_xaxis()

        # Right panel: plot of current image data
        # For the desired channel and polarisation
        ax2 = fig.add_subplot(122, projection=wcs, slices=slices)
        image = ax2.imshow(
            self.data[channel_index][stokes_index],
            cmap="YlGnBu",
            origin="lower",
            vmin=vmin_image,
            vmax=vmax_image,
        )
        plt.colorbar(image, ax=ax2, label="Flux Density [Jy]")

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show(block=block)

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

    def get_beam_parameters(self) -> BeamType:
        """Gets the beam-parameters fom the image-header.

        "bmaj": FWHM of the major axis of the elliptical Gaussian beam in arcsec
        "bmin": FWHM of the minor minor axis of the elliptical Gaussian beam in arcsec
        "bpa": position angle of the major axis of the elliptical Gaussian beam in
            degrees, counter-clock from the North direction

        Returns:
           "bmaj" (arcsec), "bmin" (arcsec), "bpa" (deg)
        """
        try:
            bmaj = float(self.header["BMAJ"])
            bmin = float(self.header["BMIN"])
            bpa = float(self.header["BPA"])
        except Exception as e:
            raise RuntimeError(
                f"No beam-parameters 'BMAJ', 'BMIN', 'BPA' found in {self.path}. "
                + "Use `has_beam_parameters` for save use of this function."
            ) from e
        beam: BeamType = {
            "bmaj": bmaj,
            "bmin": bmin,
            "bpa": bpa,
        }
        return beam

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
        path: Optional[FilePathType] = None,
        block: bool = False,
    ) -> None:
        """
        Plot the power spectrum of this image.

        :param resolution: Resolution in radians needed for conversion from Jy to Kelvin
        :param signal_channel: channel containing both signal and noise
        (arr of same shape as nchan of Image), optional
        :param save_png: True if result should be saved, default = False
        :param block: Whether plotting should block the remaining of the script
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
        max_profile = float(np.max(profile))
        plt.gca().set_ylim(1e-6 * max_profile, 2.0 * max_profile)
        plt.tight_layout()

        if path is not None:
            plt.savefig(path)
        plt.show(block=block)
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

    def get_2d_wcs(self, ra_dec_axis: Tuple[int, int] = (1, 2)) -> WCS:
        wcs = WCS(self.header)
        wcs_2d = wcs.sub(ra_dec_axis)
        return wcs_2d


class ImageMosaicker:
    """
    A class to handle the combination of multiple images into a single mosaicked image.
    See: https://reproject.readthedocs.io/en/stable/mosaicking.html

    Parameters
    More information on the parameters can be found in the documentation:
    https://reproject.readthedocs.io/en/stable/api/reproject.mosaicking.reproject_and_coadd.html # noqa: E501
    However, here the most common to tune are explained.
    ----------
    reproject_function : callable, optional
        The function to use for the reprojection.
    combine_function : {'mean', 'sum'}
        The type of function to use for combining the values into the final image.
    match_background : bool, optional
        Whether to match the backgrounds of the images.
    background_reference : None or int, optional
        If None, the background matching will make it so that the average of the
        corrections for all images is zero.
        If an integer, this specifies the index of the image to use as a reference.

    Methods
    -------
    get_optimal_wcs(images, projection='SIN', **kwargs)
        Get the optimal WCS for the given images. See:
        https://reproject.readthedocs.io/en/stable/api/reproject.mosaicking.find_optimal_celestial_wcs.html # noqa: E501
    process(
        images
        )
        Combine the provided images into a single mosaicked image.

    """

    def __init__(
        self,
        reproject_function: Callable[..., Any] = reproject_interp,
        combine_function: str = "mean",
        match_background: bool = False,
        background_reference: Optional[int] = None,
    ):
        self.reproject_function = reproject_function
        self.combine_function = combine_function
        self.match_background = match_background
        self.background_reference = background_reference

    def get_optimal_wcs(
        self,
        images: List[Image],
        projection: str = "SIN",
        **kwargs: Any,
    ) -> Tuple[WCS, tuple[int, int]]:
        """
        Set the optimal WCS for the given images.
        See: https://reproject.readthedocs.io/en/stable/api/reproject.mosaicking.find_optimal_celestial_wcs.html # noqa: E501

        Parameters
        ----------
        images : list
            A list of images to combine.
        projection : str, optional
            Three-letter code for the WCS projection, such as 'SIN' or 'TAN'.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the reprojection function.

        Returns
        -------
        WCS
            The optimal WCS for the given images.
        tuple
            The shape of the optimal WCS.

        """
        optimal_wcs = find_optimal_celestial_wcs(
            [image.to_2dNNData() for image in images]
            if isinstance(images[0], Image)
            else images,
            projection=projection,
            **kwargs,
        )
        # Somehow, a cast is needed here otherwise mypy complains
        return cast(Tuple[WCS, Tuple[int, int]], optimal_wcs)

    def mosaic(
        self,
        images: List[Image],
        wcs: Optional[Tuple[WCS, Tuple[int, int]]] = None,
        input_weights: Optional[
            List[Union[str, fits.HDUList, fits.PrimaryHDU, NDArray[np.float64]]],
        ] = None,
        hdu_in: Optional[Union[int, str]] = None,
        hdu_weights: Optional[Union[int, str]] = None,
        shape_out: Optional[Tuple[int]] = None,
        image_for_header: Optional[Image] = None,
        **kwargs: Any,
    ) -> Tuple[Image, NDArray[np.float64]]:
        """
        Combine the provided images into a single mosaicked image.

        Parameters
        ----------
        images : list
            A list of images to combine.
        wcs : tuple, optional
            The WCS to use for the mosaicking. Will be calculated with `get_optimal_wcs`
            if not passed.
        input_weights : list, optional
            If specified, an iterable with the same length as images, containing weights
            for each image.
        shape_out : tuple, optional
            The shape of the output data. If None, it will be computed from the images.
        hdu_in : int or str, optional
            If one or more items in input_data is a FITS file or an HDUList instance,
            specifies the HDU to use.
        hdu_weights : int or str, optional
            If one or more items in input_weights is a FITS file or an HDUList instance,
            specifies the HDU to use.
        image_for_header : Image, optional
            From which image the header should be used to readd the lost information
            by the mosaicking because some information is not propagated.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the reprojection function.

        Returns
        -------
        fits.PrimaryHDU
            The final mosaicked image as a FITS HDU.
        np.ndarray
            The footprint of the final mosaicked image.

        Raises
        ------
        ValueError
            If less than two images are provided.

        """

        if image_for_header is None:
            image_for_header = images[0]

        if isinstance(images[0], Image):
            images = [image.to_2dNNData() for image in images]
        if wcs is None:
            wcs = self.get_optimal_wcs(images)

        array, footprint = reproject_and_coadd(
            images,
            output_projection=wcs[0],
            shape_out=wcs[1] if shape_out is None else shape_out,
            input_weights=input_weights,
            hdu_in=hdu_in,
            reproject_function=reproject_interp,
            hdu_weights=hdu_weights,
            combine_function=self.combine_function,
            match_background=self.match_background,
            background_reference=self.background_reference,
            **kwargs,
        )
        header = wcs[0].to_header()
        header = Image.update_header_from_image_header(header, image_for_header.header)
        return (
            Image(
                data=array[np.newaxis, np.newaxis, :, :],
                header=wcs[0].to_header(),
            ),
            footprint,
        )
