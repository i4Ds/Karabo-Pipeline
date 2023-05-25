from __future__ import annotations

import copy
import enum
import logging
import math
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast
from warnings import warn

import dask.array as da
import h5py
import matplotlib.pyplot as plt
import numpy as np
import oskar
import pandas as pd
import xarray as xr
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.wcs import WCS
from numpy.typing import NDArray

from karabo.data.external_data import (
    BATTYESurveyDownloadObject,
    GLEAMSurveyDownloadObject,
    MIGHTEESurveyDownloadObject,
)
from karabo.error import KaraboSkyModelError
from karabo.util._types import (
    IntFloat,
    IntFloatList,
    NPFloatInpBroadType,
    PrecisionType,
)
from karabo.util.hdf5_util import convert_healpix_2_radec, get_healpix_image
from karabo.util.math_util import get_poisson_disk_sky
from karabo.util.plotting_util import get_slices
from karabo.warning import KaraboWarning

GLEAM_freq = Literal[
    76,
    84,
    92,
    99,
    107,
    115,
    122,
    130,
    143,
    151,
    158,
    166,
    174,
    181,
    189,
    197,
    204,
    212,
    220,
    227,
]
StokesType = Literal["Stokes I", "Stokes Q", "Stokes U", "Stokes V "]

NPSkyType = Union[NDArray[np.float_], NDArray[np.object_]]
SkySourcesType = Union[NPSkyType, xr.DataArray]


class Polarisation(enum.Enum):
    STOKES_I = (0,)
    STOKES_Q = (1,)
    STOKES_U = (2,)
    STOKES_V = 3


@dataclass
class SkyPrefixMapping:
    ra: str
    dec: str
    stokes_i: Optional[str] = None
    stokes_q: Optional[str] = None
    stokes_u: Optional[str] = None
    stokes_v: Optional[str] = None
    ref_freq: Optional[str] = None
    spectral_index: Optional[str] = None
    rm: Optional[str] = None
    major: Optional[str] = None
    minor: Optional[str] = None
    pa: Optional[str] = None
    id: Optional[str] = None


class SkyModel:
    """
    Class containing all information of the to be observed Sky.

    `SkyModel.sources` is a `xarray.DataArray`
        ( https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html ).
    `np.ndarray` are also supported as input type for all `SkyModel` functions,
        however, the values in `SkyModel.sources` are `xarray.DataArray`.

    :ivar sources:  List of all point sources in the sky.
                    A single point source consists of:

                    - right ascension (deg)
                    - declination (deg)
                    - stokes I Flux (Jy)
                    - stokes Q Flux (Jy): defaults to 0
                    - stokes U Flux (Jy): defaults to 0
                    - stokes V Flux (Jy): defaults to 0
                    - reference_frequency (Hz): defaults to 0
                    - spectral index (N/A): defaults to 0
                    - rotation measure (rad / m^2): defaults to 0
                    - major axis FWHM (arcsec): defaults to 0
                    - minor axis FWHM (arcsec): defaults to 0
                    - position angle (deg): defaults to 0
                    - source id (object): defaults to None

    """

    SOURCES_COLS = 13
    _STOKES_IDX: Dict[StokesType, int] = {
        "Stokes I": 2,
        "Stokes Q": 3,
        "Stokes U": 4,
        "Stokes V": 5,
    }
    _SOURCES_DIM1 = "source_name"
    _SOURCES_DIM2 = "data"

    def __init__(
        self,
        sources: Optional[SkySourcesType] = None,
        wcs: Optional[WCS] = None,
    ) -> None:
        """
        Initialization of a new SkyModel

        :param sources: Adds point sources
        :param wcs: world coordinate system
        """
        self.wcs = wcs
        self.sources: Optional[xr.DataArray] = None
        if sources is not None:
            if not isinstance(sources, xr.DataArray):
                sources = self._to_xarray(sources)

            self.add_point_sources(sources)

    def __get_empty_sources(self, n_sources: int) -> SkySourcesType:
        empty_sources = np.hstack(
            (
                np.zeros((n_sources, SkyModel.SOURCES_COLS - 1)),
                np.array([[np.nan] * n_sources]).reshape(-1, 1),
            )
        )
        return xr.DataArray(empty_sources)

    def _to_xarray(self, array: SkySourcesType) -> xr.DataArray:
        # if isinstance(array, xr.DataArray):
        #     return array
        if array.shape[1] == SkyModel.SOURCES_COLS:
            da = xr.DataArray(
                array[:, 0:12],
                dims=[SkyModel._SOURCES_DIM1, SkyModel._SOURCES_DIM2],
                coords={SkyModel._SOURCES_DIM1: array[:, 12]},
            )
        else:
            da = xr.DataArray(
                array, dims=[SkyModel._SOURCES_DIM1, SkyModel._SOURCES_DIM2]
            )
            # Generate source names
            da.coords[SkyModel._SOURCES_DIM1] = (
                SkyModel._SOURCES_DIM1,
                [f"source_{i}" for i in range(array.shape[0])],
            )

        da.data = da.data.astype(np.float_)
        return da

    def add_point_sources(self, sources: SkySourcesType) -> None:
        """
        Add new point sources to the sky model.

        :param sources: Array-like with shape (number of sources, 13), each row
        representing one source. The indices in the second dimension of the array
        correspond to:

            - [0] right ascension (deg)-
            - [1] declination (deg)
            - [2] stokes I Flux (Jy)
            - [3] stokes Q Flux (Jy): defaults to 0
            - [4] stokes U Flux (Jy): defaults to 0
            - [5] stokes V Flux (Jy): defaults to 0
            - [6] reference_frequency (Hz): defaults to 0
            - [7] spectral index (N/A): defaults to 0
            - [8] rotation measure (rad / m^2): defaults to 0
            - [9] major axis FWHM (arcsec): defaults to 0
            - [10] minor axis FWHM (arcsec): defaults to 0
            - [11] position angle (deg): defaults to 0
            - [12] source id (object): defaults to None

        """
        if len(sources.shape) != 2:
            raise ValueError(
                "`sources` must be 2-dimensional but "
                + f"is {len(sources.shape)}-dimensional."
            )
        if sources.shape[1] < 3:
            raise ValueError(
                "`sources` requires min 3 cols: `right_ascension`, "
                + "`declination` and `stokes I Flux`."
            )
        if sources.shape[1] < SkyModel.SOURCES_COLS - 1:
            # if some elements are missing,
            # fill them up with zeros except `source_id`
            missing_shape = SkyModel.SOURCES_COLS - sources.shape[1]
            fill = self.__get_empty_sources(sources.shape[0])
            fill[:, :-missing_shape] = sources
            sources = fill
        if self.sources is not None:
            self.sources = np.vstack((self.sources, sources))
        else:
            self.sources = sources

    def add_point_source(
        self,
        right_ascension: IntFloat,
        declination: IntFloat,
        stokes_I_flux: IntFloat,
        stokes_Q_flux: IntFloat = 0,
        stokes_U_flux: IntFloat = 0,
        stokes_V_flux: IntFloat = 0,
        reference_frequency: IntFloat = 0,
        spectral_index: IntFloat = 0,
        rotation_measure: IntFloat = 0,
        major_axis_FWHM: IntFloat = 0,
        minor_axis_FWHM: IntFloat = 0,
        position_angle: IntFloat = 0,
        source_id: Optional[object] = None,
    ) -> None:
        """
        Add a single new point source to the sky model.

        :param right_ascension:
        :param declination:
        :param stokes_I_flux:
        :param stokes_Q_flux:
        :param stokes_U_flux:
        :param stokes_V_flux:
        :param reference_frequency:
        :param spectral_index:
        :param rotation_measure:
        :param major_axis_FWHM:
        :param minor_axis_FWHM:
        :param position_angle:
        :param source_id:
        """
        new_sources = np.array(
            [
                [
                    right_ascension,
                    declination,
                    stokes_I_flux,
                    stokes_Q_flux,
                    stokes_U_flux,
                    stokes_V_flux,
                    reference_frequency,
                    spectral_index,
                    rotation_measure,
                    major_axis_FWHM,
                    minor_axis_FWHM,
                    position_angle,
                    source_id,
                ]
            ],
            dtype=np.object_,
        )
        if self.sources is not None:
            self.sources = np.vstack(self.sources, new_sources)  # type: ignore
        else:
            self.sources = new_sources

    def write_to_file(self, path: str) -> None:
        self.save_sky_model_as_csv(path)

    @staticmethod
    def read_from_file(path: str) -> SkyModel:
        """
        Read a CSV file in to create a SkyModel.
        The CSV should have the following columns

        - right ascension (deg)
        - declination (deg)
        - stokes I Flux (Jy)
        - stokes Q Flux (Jy): if no information available, set to 0
        - stokes U Flux (Jy): if no information available, set to 0
        - stokes V Flux (Jy): if no information available, set to 0
        - reference_frequency (Hz): if no information available, set to 0
        - spectral index (N/A): if no information available, set to 0
        - rotation measure (rad / m^2): if no information available, set to 0
        - major axis FWHM (arcsec): if no information available, set to 0
        - minor axis FWHM (arcsec): if no information available, set to 0
        - position angle (deg): if no information available, set to 0
        - source id (object): if no information available, set to None

        :param path: file to read in
        :return: SkyModel
        """
        dataframe = pd.read_csv(path)

        if dataframe.shape[1] < 3:
            raise KaraboSkyModelError(
                f"CSV does not have the necessary 3 basic columns (RA, DEC and "
                f"STOKES I), but only {dataframe.shape[1]} columns."
            )

        if dataframe.shape[1] < 13:
            logging.info(
                f"The CSV only holds {dataframe.shape[1]} columns."
                f" Extra {13 - dataframe.shape[1]} "
                + "columns will be filled with defaults."
            )

        if dataframe.shape[1] >= 13:
            logging.info(
                f"CSV has {dataframe.shape[1] - 13} too many rows. "
                + "The extra rows will be cut off."
            )

        sky = SkyModel(dataframe)
        return sky

    def to_array(self, with_obj_ids: bool = False) -> SkySourcesType:
        """
        Gets the sources as np.ndarray

        :param with_obj_ids: Option whether object ids should be included or not

        :return: the sources of the SkyModel as np.ndarray
        """
        if self.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before calling `to_array`."
            )
        if with_obj_ids:
            return np.hstack(
                (
                    self.sources.to_numpy(),
                    self.sources[SkyModel._SOURCES_DIM1].values.reshape(-1, 1),
                )
            )
        else:
            return self.sources.to_numpy()

    def rechunk_array_based_on_self(self, array: xr.DataArray):
        if self.sources.chunks is not None:
            chunk_size = max(self.sources.chunks[0][0], 1)
            array = array.chunk({SkyModel._SOURCES_DIM1: chunk_size})
        else:
            pass
        return array

    def filter_by_radius(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: bool = False,
    ) -> Union[SkyModel, Tuple[SkyModel, NDArray[np.int_]]]:
        """
        Filters the sky according to an inner and outer circle from the phase center

        :param inner_radius_deg: Inner radius in degrees
        :param outer_radius_deg: Outer radius in degrees
        :param ra0_deg: Phase center right ascension
        :param dec0_deg: Phase center declination
        :param indices: Optional parameter, if set to True,
        we also return the indices of the filtered sky copy
        :return sky: Filtered copy of the sky
        """
        copied_sky = copy.deepcopy(self)
        if copied_sky.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before calling `filter_by_radius`."
            )
        inner_circle = SphericalCircle(
            (ra0_deg * u.deg, dec0_deg * u.deg),
            inner_radius_deg * u.deg,  # pyright: ignore
        )
        outer_circle = SphericalCircle(
            (ra0_deg * u.deg, dec0_deg * u.deg),
            outer_radius_deg * u.deg,  # pyright: ignore
        )
        outer_sources = outer_circle.contains_points(copied_sky[:, 0:2])
        inner_sources = inner_circle.contains_points(copied_sky[:, 0:2])
        filtered_sources = np.logical_and(outer_sources, np.logical_not(inner_sources))
        filtered_sources_idxs = np.where(filtered_sources == True)[0]  # noqa
        copied_sky.sources = copied_sky.sources[filtered_sources_idxs]

        # Rechunk the array to the original chunk size
        copied_sky.sources = self.rechunk_array_based_on_self(copied_sky.sources)

        if indices is True:
            return copied_sky, filtered_sources_idxs
        else:
            return copied_sky

    def filter_by_radius_euclidean_flat_approximation(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: bool = False,
    ) -> Union[SkyModel, Tuple[SkyModel, NPSkyType]]:
        copied_sky = copy.deepcopy(self)
        if copied_sky.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before calling `filter_by_radius`."
            )

        # Calculate distances to phase center using flat Euclidean approximation
        x = (copied_sky[:, 0] - ra0_deg) * np.cos(np.radians(dec0_deg))
        y = copied_sky[:, 1] - dec0_deg
        distances_sq = np.add(np.square(x), np.square(y))

        # Filter sources based on inner and outer radius
        filter_mask = (distances_sq >= np.square(inner_radius_deg)) & (
            distances_sq <= np.square(outer_radius_deg)
        )

        copied_sky.sources = copied_sky.sources[filter_mask]

        copied_sky.sources = self.rechunk_array_based_on_self(copied_sky.sources)

        if indices:
            filtered_indices = np.where(filter_mask)[0]
            return copied_sky, filtered_indices
        else:
            return copied_sky

    def filter_by_flux(
        self,
        min_flux_jy: IntFloat,
        max_flux_jy: IntFloat,
    ) -> SkyModel:
        """
        Filters the sky using the Stokes-I-flux
        Values outside the range are removed

        :param min_flux_jy: Minimum flux in Jy
        :param max_flux_jy: Maximum flux in Jy
        :return sky: Filtered copy of the sky
        """
        copied_sky = copy.deepcopy(self)
        if copied_sky.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before calling `filter_by_flux`."
            )

        # Create mask
        filter_mask = (copied_sky[:, 2] >= min_flux_jy) & (
            copied_sky[:, 2] <= max_flux_jy
        )
        filter_mask = self.rechunk_array_based_on_self(filter_mask)

        # Apply the filter mask and drop the unmatched rows
        copied_sky.sources = copied_sky.sources.where(filter_mask, drop=True)

        return copied_sky

    def filter_by_frequency(
        self,
        min_freq: float,
        max_freq: float,
    ) -> SkyModel:
        """
        Filters the sky using the reference frequency in Hz

        :param min_freq: Minimum frequency in Hz
        :param max_freq: Maximum frequency in Hz
        :return sky: Filtered copy of the sky
        """
        copied_sky = copy.deepcopy(self)
        if copied_sky.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before calling `filter_by_frequency`."
            )

        # Create mask
        filter_mask = (copied_sky.sources[:, 6] >= min_freq) & (
            copied_sky.sources[:, 6] <= max_freq
        )
        filter_mask = self.rechunk_array_based_on_self(filter_mask)

        # Apply the filter mask and drop the unmatched rows
        copied_sky.sources = copied_sky.sources.where(filter_mask, drop=True)

        return copied_sky

    def get_wcs(self) -> WCS:
        """
        Gets the currently active world coordinate system astropy.wcs
        For details see https://docs.astropy.org/en/stable/wcs/index.html

        :return: world coordinate system
        """
        return self.wcs

    def set_wcs(self, wcs: WCS) -> None:
        """
        Sets a new world coordinate system astropy.wcs
        For details see https://docs.astropy.org/en/stable/wcs/index.html

        :param wcs: world coordinate system
        """
        self.wcs = wcs

    def setup_default_wcs(
        self,
        phase_center: IntFloatList = [0, 0],
    ) -> WCS:
        """
        Defines a default world coordinate system astropy.wcs
        For more details see https://docs.astropy.org/en/stable/wcs/index.html

        :param phase_center: ra-dec location

        :return: wcs
        """
        w = WCS(naxis=2)
        w.wcs.crpix = [0, 0]  # coordinate reference pixel per axis
        w.wcs.cdelt = [-1, 1]  # coordinate increments on sphere per axis
        w.wcs.crval = phase_center
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]  # coordinate axis type
        self.wcs = w
        return w

    @staticmethod
    def get_fits_catalog(path: str) -> Table:
        """
        Gets astropy.table.table.Table from a .fits catalog

        :param path: Location of the .fits file

        :return: fits catalog
        """
        return Table.read(path)

    def explore_sky(
        self,
        phase_center: IntFloatList,
        stokes: StokesType = "Stokes I",
        idx_to_plot: Optional[NDArray[np.int_]] = None,
        xlim: Optional[Tuple[IntFloat, IntFloat]] = None,
        ylim: Optional[Tuple[IntFloat, IntFloat]] = None,
        figsize: Optional[Tuple[IntFloat, IntFloat]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        cfun: Optional[Callable[..., NPFloatInpBroadType]] = np.log10,
        cmap: Optional[str] = "plasma",
        cbar_label: Optional[str] = None,
        with_labels: bool = False,
        wcs: Optional[WCS] = None,
        wcs_enabled: bool = True,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        A scatter plot of the `SkyModel` (self) where the sources
        are projected according to the `phase_center`

        :param phase_center: [RA,DEC]
        :param stokes: `SkyModel` stoke flux
        :param idx_to_plot: If you want to plot only a subset of the sources, set
                            the indices here.
        :param xlim: RA-limit of plot
        :param ylim: DEC-limit of plot
        :param figsize: figsize as tuple
        :param title: plot title
        :param xlabel: xlabel
        :param ylabel: ylabel
        :param cfun: flux scale transformation function for scatter-coloring
        :param cmap: matplotlib color map
        :param cbar_label: color bar label
        :param with_labels: Plots object ID's if set?
        :param wcs: If you want to use a custom astropy.wcs, ignores `phase_center` if
                    set
        :param wcs_enabled: Use wcs transformation?
        :param filename: Set to path/fname to save figure (set extension to fname to
                         overwrite .png default)
        :param kwargs: matplotlib kwargs for scatter & Collections, e.g. customize `s`,
                       `vmin` or `vmax`
        """
        # To avoid having to read the data multiple times, we read it once here
        if idx_to_plot is not None:
            data = self.sources[idx_to_plot].as_numpy()
        else:
            data = self.sources.as_numpy()
        if wcs_enabled:
            if wcs is None:
                wcs = self.setup_default_wcs(phase_center)
            px, py = wcs.wcs_world2pix(
                self[:, 0], self[:, 1], 0
            )  # ra-dec transformation
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
        else:
            px, py = self[:, 0], self[:, 1]

        flux = None
        if cmap is not None:
            flux = self[:, SkyModel._STOKES_IDX[stokes]]
            if cfun is not None:
                if cfun in [np.log10, np.log] and any(flux <= 0):
                    warn(
                        KaraboWarning(
                            "Warning: flux with value <= 0 found, setting "
                            "those to np.nan to avoid "
                            "logarithmic errors (only affects the colorbar)"
                        )
                    )

                    flux = np.where(flux > 0, flux, np.nan)
                flux = cfun(flux)  # type: ignore

        # handle matplotlib kwargs
        # not set as normal args because default assignment depends on args
        if "vmin" not in kwargs:
            kwargs["vmin"] = np.nanmin(flux)  # type: ignore
        if "vmax" not in kwargs:
            kwargs["vmax"] = np.nanmax(flux)  # type: ignore

        if wcs_enabled:
            slices = get_slices(wcs)
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw=dict(projection=wcs, slices=slices)
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(px, py, c=flux, cmap=cmap, **kwargs)

        if with_labels:
            unique_keys, indices = np.unique(
                data[SkyModel._SOURCES_DIM1], return_index=True
            )
            for i, txt in enumerate(unique_keys):
                if self.shape[0] > 1:
                    ax.annotate(
                        txt,
                        (px[indices][i], py[indices][i]),
                    )
                else:
                    ax.annotate(txt, (px, py))
        # Add grid
        ax.grid()
        plt.axis("equal")
        if cbar_label is None:
            cbar_label = ""
        plt.colorbar(sc, label=cbar_label)
        if title is not None:
            plt.title(title)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.show(block=False)
        plt.pause(1)

        if isinstance(filename, str):
            fig.savefig(fname=filename)

    @staticmethod
    def get_OSKAR_sky(
        sky: Union[SkySourcesType, SkyModel],
        precision: PrecisionType,
    ) -> oskar.Sky:
        """
        Get OSKAR sky model object from the defined Sky Model

        :return: oskar sky model
        """
        if sky.shape[1] > 12:
            return oskar.Sky.from_array(sky[:, :12], precision)
        else:
            return oskar.Sky.from_array(sky, precision)

    @staticmethod
    def read_healpix_file_to_sky_model_array(
        file: str,
        channel: int,
        polarisation: Polarisation,
    ) -> Tuple[NDArray[np.float_], int]:
        """
        Read a healpix file in hdf5 format.
        The file should have the map keywords:

        :param file: hdf5 file path (healpix format)
        :param channel: Channels of observation (between 0 and maximum numbers of
                        channels of observation)
        :param polarisation: 0 = Stokes I, 1 = Stokes Q, 2 = Stokes U, 3 = Stokes  V
        :return:
        """
        arr = get_healpix_image(file)
        filtered = arr[channel][polarisation.value]
        ra, dec, nside = convert_healpix_2_radec(filtered)
        return xr.DataArray(np.vstack((ra, dec, filtered)).transpose()), nside

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.sources is None:
            raise AttributeError(
                "`sources` is None and therefore has no `shape` attribute."
            )
        return self.sources.shape

    @property
    def num_sources(self) -> int:
        if self.sources is None:
            return 0
        else:
            return self.shape[0]

    def __getitem__(self, key: Any) -> SkySourcesType:
        """
        Allows to get access to self.sources in an np.ndarray like manner
        If casts the selected array/scalar to float64 if possible
        (usually if source_id is not selected)

        :param key: slice key

        :return: sliced self.sources
        """
        if self.sources is None:
            raise AttributeError("`sources` is None and therefore can't be accessed.")
        sources = cast(SkySourcesType, self.sources[key])
        try:
            sources = sources.astype(np.float64)
        except ValueError:
            pass  # nothing toDo here
        return sources

    def __setitem__(
        self,
        key: Any,
        value: Union[NPFloatInpBroadType, str],
    ) -> None:
        """
        Allows to set values in an np.ndarray like manner

        :param key: slice key
        :param value: values to store
        """
        if self.sources is None:
            value_ = cast(SkySourcesType, value)
            self.add_point_sources(sources=value_)
        else:
            self.sources[key] = value

    def save_sky_model_as_csv(self, path: str) -> None:
        """
        Save source array into a csv.
        :param path: path to save the csv file in.
        """
        df = pd.DataFrame(self.sources)
        df["source id (object)"] = self.sources[SkyModel._SOURCES_DIM1].values
        df.to_csv(
            path,
            index=False,
            header=[
                "right ascension (deg)",
                "declination (deg)",
                "stokes I Flux (Jy)",
                "stokes Q Flux (Jy)",
                "stokes U Flux (Jy)",
                "stokes V Flux (Jy)",
                "reference_frequency (Hz)",
                "spectral index (N/A)",
                "rotation measure (rad / m^2)",
                "major axis FWHM (arcsec)",
                "minor axis FWHM (arcsec)",
                "position angle (deg)",
                "source id (object)",
            ],
        )

    def save_sky_model_to_txt(
        self,
        path: str,
        cols: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
    ) -> None:
        if self.sources is None:
            raise AttributeError("Can't save sky-model because `sources` is None.")
        np.savetxt(path, self.sources[:, cols])

    @staticmethod
    def __convert_ra_dec_to_cartesian(
        ra: IntFloat, dec: IntFloat
    ) -> NDArray[np.float_]:
        x = math.cos(math.radians(ra)) * math.cos(math.radians(dec))
        y = math.sin(math.radians(ra)) * math.cos(math.radians(dec))
        z = math.sin(math.radians(dec))
        r = np.array([x, y, z])
        norm = cast(np.float_, np.linalg.norm(r))
        if norm == 0:
            return r
        return r / norm

    def get_cartesian_sky(self) -> NDArray[np.float_]:
        if self.sources is None:
            raise AttributeError("Can't create cartesian-sky when `sources` is None.")
        cartesian_sky = np.squeeze(
            np.apply_along_axis(
                lambda row: [
                    self.__convert_ra_dec_to_cartesian(float(row[0]), float(row[1]))
                ],
                axis=1,
                arr=self.sources,
            )
        )
        return cartesian_sky

    @staticmethod
    def get_sky_model_from_h5_to_xarray(
        path: str,
        prefix_mapping: SkyPrefixMapping,
        extra_columns: Optional[List[str]] = None,
        chunksize: Union[int, str] = "auto",
    ) -> SkyModel:
        """
        Load a sky model dataset from an HDF5 file and
        converts it to an xarray DataArray.

        Parameters
        ----------
        path : str
            Path to the input HDF5 file.
        prefix_mapping : SkyPrefixMapping
            Mapping column names to their corresponding dataset paths
            in the HDF5 file.
            If the column is not present in the HDF5 file, set its value to None.
        extra_columns : Optional[List[str]], default=None
            A list of additional column names to include in the output DataArray.
            If not provided, only the default columns will be included.
        chunksize : Union[int, str], default=1000
            Chunk size for Dask arrays. This determines the size of chunks that
            the data will be divided into when read from the file. Can be an
            integer or 'auto'. If 'auto', Dask will choose an optimal chunk size.

        Returns
        -------
        xr.DataArray
            A 2D xarray DataArray containing the sky model data. Rows represent data
            points and columns represent different data fields ('ra', 'dec', ...).
        """
        f = h5py.File(path, "r")
        data_arrays = []

        for field in fields(prefix_mapping):
            field_value: Optional[str] = getattr(prefix_mapping, field.name)
            if field_value is None:
                shape = f[prefix_mapping.ra].shape
                dask_array = da.zeros(shape, chunks=(chunksize,))
            else:
                dask_array = da.from_array(f[field_value], chunks=(chunksize,))
            data_arrays.append(xr.DataArray(dask_array, dims=[SkyModel._SOURCES_DIM1]))

        if extra_columns is not None:
            for col in extra_columns:
                dask_array = da.from_array(f[col], chunks=(chunksize,))
                data_arrays.append(
                    xr.DataArray(dask_array, dims=[SkyModel._SOURCES_DIM1])
                )

        sky = cast(xr.DataArray, xr.concat(data_arrays, dim="columns"))
        sky = sky.T
        sky = sky.chunk({SkyModel._SOURCES_DIM1: chunksize, "columns": sky.shape[1]})
        return SkyModel(sky)

    @staticmethod
    def get_GLEAM_Sky(frequencies: Optional[List[GLEAM_freq]] = None) -> SkyModel:
        """
        get_GLEAM_Sky - Returns a SkyModel object containing sources with flux densities
        at the specified frequencies from the GLEAM survey.

        Parameters:
            frequencies (list): A list of frequencies in MHz for which the flux
            densities are required. Available frequencies are:
            [76, 84, 92, 99, 107, 115, 122, 130, 143, 151, 158, 166,
            174, 181, 189, 197, 204, 212, 220, 227]. Default is to return
            all frequencies.

        Returns:
            SkyModel: A SkyModel object containing sources with flux densities
            at the specified frequencies (Hz).

        Example:
            >>> gleam_sky = SkyModel.get_GLEAM_Sky([76, 107, 143])
            >>> print(gleam_sky)
            <SkyModel object at 0x7f8a1545fc10>
            >>> print(gleam_sky.num_sources)
            921259
        """
        if frequencies is None:
            frequencies = [
                76,
                84,
                92,
                99,
                107,
                115,
                122,
                130,
                143,
                151,
                158,
                166,
                174,
                181,
                189,
                197,
                204,
                212,
                220,
                227,
            ]
        # Get path to Gleamsurvey
        survey = GLEAMSurveyDownloadObject()
        path = survey.get()

        prefix_mapping = SkyPrefixMapping(
            ra="RAJ2000",
            dec="DEJ2000",
            stokes_i="Fp",
            minor="a",
            pa="b",
            id="GLEAM",
        )

        return SkyModel.get_sky_model_from_fits(
            path=path,
            frequencies=frequencies,  # type: ignore[arg-type]
            prefix_mapping=prefix_mapping,
            concat_freq_with_prefix=True,
            filter_data_by_stokes_i=True,
            frequency_to_mhz_multiplier=1e6,
        )

    @staticmethod
    def get_sky_model_from_fits(
        path: str,
        frequencies: List[int],
        prefix_mapping: SkyPrefixMapping,
        concat_freq_with_prefix: bool = False,
        filter_data_by_stokes_i: bool = False,
        frequency_to_mhz_multiplier: float = 1e6,
        chunksize: Union[int, str] = "auto",
        memmap: bool = False,
    ) -> SkyModel:
        """
        Reads data from a FITS file and creates an xarray Dataset containing
        information about celestial sources at given frequencies.

        Parameters
        ----------
        path : str
            Path to the input FITS file.
        frequencies : list
            List of frequencies of interest in MHz.
        prefix_mapping : SkyPrefixMapping
            Maps column names in the FITS file to column names
            used in the output Dataset.
            Keys should be the original column names, values should be the
            corresponding column names in the Dataset.
            Any column names not present in the dictionary will be excluded
            from the output Dataset.
        concat_freq_with_prefix : bool, optional
            If True, concatenates the frequency with the prefix for each column.
            Defaults to False.
        filter_data_by_stokes_i : bool, optional
            If True, filters the data by Stokes I. Defaults to False.
        frequency_to_mhz_multiplier : float, optional
            Factor to convert the frequency units to MHz. Defaults to 1e6.
        chunksize : int or str, optional
            The size of the chunks to use when creating the Dask arrays. This can
            be an integer representing the number of rows per chunk, or a string
            representing the size of each chunk in bytes (e.g. '64MB', '1GB', etc.)
            or 'auto'.
        memmap : bool, optional
            Whether to use memory mapping when opening the FITS file. Defaults to False.
            Allows for reading of larger-than-memory files.

        Returns
        -------
        xr.DataArray
            xarray DataArray object containing information about celestial sources
            at given frequencies.

        Notes
        -----
        The FITS file should have a data table in its first HDU.
        Valid columns are:
        "ra", "dec", "stokes_i", "stokes_q", "stokes_u", "stokes_v", "ref_freq",
        "spectral_index", "rm", "major", "minor", "pa", and "id".
        Any additional columns will be ignored.

        The `prefix_mapping` values should map the required columns to their
        corresponding column names in the input FITS file. For example:

        If a column name in the FITS file is not present in `prefix_mapping`
        values, it will be excluded from the output Dataset.

        Examples
        --------
        >>> sky_data = SkyModel.get_sky_model_from_fits(
        ...     path="input.fits",
        ...     frequencies=[100, 200, 300],
        ...     prefix_mapping=SkyPrefixMapping(
        ...         ra="RAJ2000",
        ...         dec="DEJ2000",
        ...         stokes_i="Fp",
        ...         minor="a",
        ...         pa="b",
        ...         id="pa",
        ...     ),
        ...     concat_freq_with_prefix=True,
        ...     filter_data_by_stokes_i=True,
        ...     chunks='auto',
        ...     memmap=False,
        ... )

        """
        with fits.open(path, memmap=memmap) as hdul:
            header = hdul[1].header
            data = hdul[1].data

        column_names = [header[f"TTYPE{i+1}"] for i in range(header["TFIELDS"])]
        data_dict = {name: data[name] for name in column_names}

        dataset = xr.Dataset(data_dict)
        data_arrays = []

        for freq in frequencies:
            freq_str = str(freq).zfill(3)

            if filter_data_by_stokes_i and prefix_mapping.stokes_i is not None:
                dataset_filtered = dataset.dropna(
                    dim=f"{prefix_mapping.stokes_i}{freq_str}"
                )
            else:
                dataset_filtered = dataset

            data = []
            if prefix_mapping.id is not None:
                source_names = xr.DataArray(
                    dataset_filtered[prefix_mapping.id].values,
                    dims=[SkyModel._SOURCES_DIM1],
                )
            else:
                source_names = xr.DataArray(
                    np.arange(len(dataset_filtered[prefix_mapping.ra])),
                    dims=[SkyModel._SOURCES_DIM1],
                )
            for field in fields(prefix_mapping):
                col = field.name
                pm_col: Optional[str] = getattr(prefix_mapping, field.name)
                if col == "id":
                    continue
                if pm_col is not None:
                    if concat_freq_with_prefix and col not in ["ra", "dec"]:
                        col_name = pm_col + freq_str
                    else:
                        col_name = pm_col

                    freq_data = xr.DataArray(
                        dataset_filtered[col_name].values,
                        dims=[SkyModel._SOURCES_DIM1],
                        coords={SkyModel._SOURCES_DIM1: source_names},
                    )
                elif col == "ref_freq":
                    freq_data = xr.DataArray(
                        np.full(
                            len(dataset_filtered[prefix_mapping.ra]),
                            freq * frequency_to_mhz_multiplier,
                        ),
                        dims=[SkyModel._SOURCES_DIM1],
                        coords={SkyModel._SOURCES_DIM1: source_names},
                    )
                else:
                    freq_data = xr.DataArray(
                        np.zeros(len(dataset_filtered[prefix_mapping.ra])),
                        dims=[SkyModel._SOURCES_DIM1],
                        coords={SkyModel._SOURCES_DIM1: source_names},
                    )
                data.append(freq_data)

            data_array = xr.concat(data, dim="columns")

            data_arrays.append(data_array)

        for freq_dataset in data_arrays:
            freq_dataset.chunk({SkyModel._SOURCES_DIM1: chunksize})

        result_dataset = cast(
            xr.DataArray,
            xr.concat(data_arrays, dim=SkyModel._SOURCES_DIM1)
            .chunk({SkyModel._SOURCES_DIM1: chunksize})
            .T,
        )

        return SkyModel(result_dataset)

    @staticmethod
    def get_BATTYE_sky() -> SkyModel:
        """
        Downloads BATTYE survey data and generates a sky
        model using the downloaded data.

        Source:
        The BATTYE survey data was provided by Jennifer Studer
        (https://github.com/jejestern)

        Returns:
            SkyModel: A sky model generated from the BATTYE survey data.
            The sky model contains the following information:

            - 'Right Ascension' (ra): The right ascension coordinates
                of the celestial objects.
            - 'Declination' (dec): The declination coordinates of the
                celestial objects.
            - 'Flux' (i): The flux measurements of the celestial objects.
            - 'Observed Redshift': Additional observed redshift information
                of the celestial objects.

            Note: Other properties such as 'stokes_q', 'stokes_u', 'stokes_v',
             'ref_freq', 'spectral_index', 'rm', 'major', 'minor', 'pa', and 'id'
            are not included in the sky model.


        """
        survey = BATTYESurveyDownloadObject()
        path = survey.get()
        column_mapping = SkyPrefixMapping(
            ra="Right Ascension",
            dec="Declination",
            stokes_i="Flux",
        )
        extra_columns = ["Observed Redshift"]

        sky = SkyModel.get_sky_model_from_h5_to_xarray(
            path=path, prefix_mapping=column_mapping, extra_columns=extra_columns
        )

        sky.sources[:, 1] *= -1

        return sky

    @staticmethod
    def get_MIGHTEE_Sky() -> SkyModel:
        """
        Downloads the MIGHTEE catalog and creates a SkyModel object.

        Returns
        -------
        SkyModel
            SkyModel object containing information about celestial sources
            in the MIGHTEE survey.

        Notes
        -----
        The MIGHTEE catalog is downloaded using the MIGHTEESurveyDownloadObject class.

        The SkyModel object contains columns for "ra", "dec", "i", "q", "u", "v",
        "ref_freq", "major", "minor", "pa", and "id".
        The "ref_freq" column is set to 76 MHz, and the "q", "u", and "v" columns
        are set to zero. The "major", "minor", "pa", and "id" columns are obtained
        from the "IM_MAJ", "IM_MIN", "IM_PA", and "NAME" columns of the catalog,
        respectively.
        """
        survey = MIGHTEESurveyDownloadObject()
        path = survey.get()
        prefix_mapping = SkyPrefixMapping(
            ra="RA",
            dec="DEC",
            stokes_i="NU_EFF",
            major="IM_MAJ",
            minor="IM_MIN",
            pa="IM_PA",
            id="NAME",
        )

        return SkyModel.get_sky_model_from_fits(
            path=path,
            frequencies=[76],
            prefix_mapping=prefix_mapping,
            concat_freq_with_prefix=False,
            filter_data_by_stokes_i=False,
            frequency_to_mhz_multiplier=1e6,
            memmap=False,
        )

    @staticmethod
    def get_random_poisson_disk_sky(
        min_size: Tuple[IntFloat, IntFloat],
        max_size: Tuple[IntFloat, IntFloat],
        flux_min: IntFloat,
        flux_max: IntFloat,
        r: int = 3,
    ) -> SkyModel:
        sky_array = xr.DataArray(
            get_poisson_disk_sky(min_size, max_size, flux_min, flux_max, r)
        )
        print(sky_array.shape)
        return SkyModel(sky_array)
