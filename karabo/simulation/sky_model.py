from __future__ import annotations

import copy
import enum
import math
import re
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)
from warnings import warn

import dask.array as da
import h5py
import matplotlib.pyplot as plt
import numpy as np
import oskar
import pandas as pd
import xarray as xr
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import ColDefs
from astropy.io.fits.fitsrec import FITS_rec
from astropy.table import Table
from astropy.units.core import PrefixUnit, Unit, UnitBase
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.wcs import WCS
from numpy.typing import NDArray
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.sky_model.sky_model import SkyComponent
from typing_extensions import assert_never
from xarray.core.coordinates import DataArrayCoordinates

from karabo.data.external_data import (  # MIGHTEESurveyDownloadObject,
    GLEAMSurveyDownloadObject,
    HISourcesSmallCatalogDownloadObject,
)
from karabo.error import KaraboSkyModelError
from karabo.simulation.line_emission_helpers import (
    convert_frequency_to_z,
    convert_z_to_frequency,
)
from karabo.simulator_backend import SimulatorBackend
from karabo.util._types import (
    FilePathType,
    IntFloat,
    IntFloatList,
    NPFloatInpBroadType,
    NPFloatLike,
    PrecisionType,
)
from karabo.util.hdf5_util import convert_healpix_2_radec, get_healpix_image
from karabo.util.math_util import get_poisson_disk_sky
from karabo.util.plotting_util import get_slices
from karabo.warning import KaraboWarning

StokesType = Literal["Stokes I", "Stokes Q", "Stokes U", "Stokes V"]
SkySourcesColName = Literal[  # preserve python-var-name compatibility
    "ra",
    "dec",
    "stokes_i",
    "stokes_q",
    "stokes_u",
    "stokes_v",
    "ref_freq",
    "spectral_index",
    "rm",
    "major",
    "minor",
    "pa",
    "true_redshift",
    "observed_redshift",
    "id",
]

_NPSkyType = Union[NDArray[np.float_], NDArray[np.object_]]
_SkySourcesType = Union[_NPSkyType, xr.DataArray]
_SourceIdType = Union[
    List[str],
    List[int],
    List[float],
    NDArray[np.object_],
    NDArray[np.int_],
    NDArray[np.float_],
    DataArrayCoordinates[xr.DataArray],
]
_SkyPrefixMappingValueType = Union[str, List[str]]


class Polarisation(enum.Enum):
    STOKES_I = (0,)
    STOKES_Q = (1,)
    STOKES_U = (2,)
    STOKES_V = 3


@dataclass
class SkyPrefixMapping:
    """Defines the relation between col-names of a .fits file and `SkyModel.sources`.

    Field-names of sources-data must be according to `SkySourcesColName`.
    """

    ra: _SkyPrefixMappingValueType
    dec: _SkyPrefixMappingValueType
    stokes_i: _SkyPrefixMappingValueType
    stokes_q: Optional[_SkyPrefixMappingValueType] = None
    stokes_u: Optional[_SkyPrefixMappingValueType] = None
    stokes_v: Optional[_SkyPrefixMappingValueType] = None
    ref_freq: Optional[_SkyPrefixMappingValueType] = None
    spectral_index: Optional[_SkyPrefixMappingValueType] = None
    rm: Optional[_SkyPrefixMappingValueType] = None
    major: Optional[_SkyPrefixMappingValueType] = None
    minor: Optional[_SkyPrefixMappingValueType] = None
    pa: Optional[_SkyPrefixMappingValueType] = None
    true_redshift: Optional[_SkyPrefixMappingValueType] = None
    observed_redshift: Optional[_SkyPrefixMappingValueType] = None
    id: Optional[_SkyPrefixMappingValueType] = None


@dataclass
class SkySourcesUnits:
    """Represents the units of `SkyModel.sources`

    This class is useful for unit-conversion from different Prefixes
        https://docs.astropy.org/en/stable/units/standard_units.html
        and different units which can be converted to another (like deg to arcmin)

    `UnitBase` covers
    - `Unit`: e.g. u.Jy, u.Hz
    - `CompositeUnit`: e.g. u.Jy/u.beam, u.rad/u.m**2
    - `PrefixUnit`: e.g. u.MHz, u.GHz

    Just assign another unit to the constructor in case the default doesn't fit
    for a specific field. It's useful if e.g. `stokes_i` is u.Jy/u.beam in the
    .fits file instead.
    """

    # field-names of unit-values must match `SkySourcesColName`
    ra: UnitBase = u.deg  # `Unit`
    dec: UnitBase = u.deg  # `Unit`
    stokes_i: UnitBase = u.Jy  # `Unit`
    stokes_q: UnitBase = u.Jy  # `Unit`
    stokes_u: UnitBase = u.Jy  # `Unit`
    stokes_v: UnitBase = u.Jy  # `Unit`
    ref_freq: UnitBase = u.Hz  # `Unit`
    rm: UnitBase = u.rad / u.m**2  # `CompositeUnit`
    major: UnitBase = u.arcsec  # `Unit`
    minor: UnitBase = u.arcsec  # `Unit`
    pa: UnitBase = u.deg  # `Unit`

    @classmethod
    def format_sky_prefix_mapping(
        cls,
        cols: ColDefs,
        prefix_mapping: SkyPrefixMapping,
        encoded_freq: Optional[UnitBase],
    ) -> Tuple[SkyPrefixMapping, int, Dict[str, float]]:
        """Formats `SkyPrefixMapping` fields from str to list[str] if formattable.

        This function doesn't do any formatting if `encoded_freq` is None.

        Args:
            cols: Columns of .fits file
            prefix_mapping: Mapping to format.
            encoded_freq: astropy.unit frequency encoded (e.g. u.MHz)

        Raises:
            RuntimeError: In case the number of formatting columns of `cols`
                are not of equal for each frequency-channel.

        Returns:
            Formatted `prefix_mapping`, number of formatted `prefix_mapping` fields,
                and a dict containing col-name -> frequency (Hz).
        """
        num_formattings = 0
        names_and_freqs: Dict[str, float] = dict()
        if encoded_freq is None:
            return prefix_mapping, num_formattings, names_and_freqs
        prefix_mapping_copied = deepcopy(prefix_mapping)  # to preserve mutable object
        for field in fields(prefix_mapping_copied):
            field_value = getattr(prefix_mapping_copied, field.name)
            if isinstance(field_value, str) and cls.is_pos_formattable(field_value):
                col_names_and_freqs = cls.extract_names_and_freqs(
                    string=field_value, cols=cols, unit=encoded_freq
                )
                if num_formattings == 0:
                    num_formattings = len(col_names_and_freqs)
                else:
                    if num_formattings != len(col_names_and_freqs):
                        raise RuntimeError(
                            "Number of formatted sky-prefixes don't match for each "
                            + f"frequency-channel! {num_formattings=} != "
                            + f"{len(col_names_and_freqs)=}"
                        )
                names_and_freqs.update(col_names_and_freqs)
                setattr(
                    prefix_mapping_copied,
                    field.name,
                    [field_name for field_name in col_names_and_freqs],
                )
        return prefix_mapping_copied, num_formattings, names_and_freqs

    def get_unit_scales(
        self,
        cols: ColDefs,
        unit_mapping: Dict[str, UnitBase],
        prefix_mapping: SkyPrefixMapping,
    ) -> Dict[str, float]:
        """Converts all units specified in `prefix_mapping` into a factor, which can be
        used for multiplication of the according data-array, to convert it to the
        `SkyModel.sources` standard-units.

        If a unit in the .fits file doesn't match the default units of this dataclass,
        just change it during instantiation to match the .fits file unit.

        Args:
            cols: Columns from `hdul[1].columns`
            unit_mapping: `Mapping from col-unit (from .fits file) to `astropy.units`.
                Be aware to use the very-same unit. E.g. if it's prefixed in the .fits,
                also prefix in the `astropy.units` (e.g. "MHZ":u.MHz, NOT "MHZ":u.Hz).
            prefix_mapping: Mapping from `SkyModel.sources` to col-name of .fits file.


        Returns:
            Scales to convert unit to standard-unit.
        """
        cols_to_field_name_of_interest: Dict[str, str] = dict()
        for field in fields(prefix_mapping):
            if (
                col_names_of_interest := getattr(prefix_mapping, field.name)
            ) is not None:
                if isinstance(col_names_of_interest, list):
                    for col_name_of_interest in col_names_of_interest:
                        cols_to_field_name_of_interest[
                            col_name_of_interest
                        ] = field.name
                elif isinstance(col_names_of_interest, str):
                    cols_to_field_name_of_interest[col_names_of_interest] = field.name
                else:
                    # should never happen, but `getattr` has no type-hints
                    raise TypeError(
                        f"{type(col_names_of_interest)=} must be list[str] | str."
                    )
        scales: Dict[str, float] = dict()
        for col in cols:
            if (col_name := col.name) in cols_to_field_name_of_interest.keys() and (
                unit := col.unit
            ) is not None:
                astropy_unit_of_fits_col = unit_mapping[unit]
                # here, it assumes that unit-field-names of `SkySourcesUnits` and
                # `SkyPrefixMapping` are the same
                sky_sources_unit: UnitBase = getattr(
                    self, cols_to_field_name_of_interest[col_name]
                )
                scales[col_name] = astropy_unit_of_fits_col.to(sky_sources_unit)
        return scales

    @classmethod
    def is_pos_formattable(cls, string: str) -> bool:
        """Checks if `string` is positional formattable.

        Args:
            string: str to check.

        Returns:
            Result if it's formattable.
        """
        return "{0}" in string

    @classmethod
    def extract_names_and_freqs(
        cls,
        string: str,
        cols: ColDefs,
        unit: Union[Unit, PrefixUnit],
    ) -> Dict[str, float]:
        """Extracts all col-names and it's according frequency (in Hz).

        `string` has to be formattable, meaning it must containt a {0} for beeing
        the frequency placeholder. Otherwise it will fail.

        Important: This function doesn't consider the unit of

        Args:
            string: Column-name with {0} frequency-placeholder.
            cols: Columns to search through.
            unit: Frequency-unit encoded in col-names (e.g. u.MHz)

        Returns:
            Extracted col-names with it's frequency as a list of tuples.
        """
        if not cls.is_pos_formattable(string=string):
            raise ValueError(f"{string=}" + "must contain '{0}' to be formattable!")
        number_format = r"(\d+)"  # just supports int-numbers atm
        pattern = re.compile(f"^{string.format(number_format)}$")
        names_and_freqs: Dict[str, float] = dict()
        for col in cols:
            col_name = col.name
            if not isinstance(col_name, str):
                raise RuntimeError(
                    f"`col_name` should be of type `str`, but is {type(col_name)=}"
                )
            match = pattern.match(string=col_name)
            if match:
                extracted_freq = float(match.group(1))
                extracted_freq = float((extracted_freq * unit).to(u.Hz).value)  # type: ignore[attr-defined] # noqa: E501
                names_and_freqs[col_name] = extracted_freq
        return names_and_freqs


XARRAY_DIM_0_DEFAULT, XARRAY_DIM_1_DEFAULT = cast(
    Tuple[str, str], xr.DataArray([[]]).dims
)


class SkyModel:
    """
    Class containing all information of the to be observed Sky.

    `SkyModel.sources` is a `xarray.DataArray`
        ( https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html ).
    `np.ndarray` are also supported as input type for `SkyModel.sources`,
        however, the values in `SkyModel.sources` are converted to `xarray.DataArray`.

    `SkyModel.compute` method is used to load the data into memory as a numpy array.
    It should be called after all the filtering and other operations are completed
    and to avoid doing the same calculation multiple thems when e.g. on a cluster.

    :ivar sources:  List of all point sources in the sky as `xarray.DataArray`.
                    The source_ids reside in `SkyModel.source_ids` if provided
                    through `xarray.sources.coords` with an arbitrary string key
                    as index or `np.ndarray` as idx SOURCES_COLS.
                    A single point source is described using the following col-order:

                    - [0] right ascension (deg)
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
                    - [12] true redshift: defaults to 0
                    - [13] observed redshift: defaults to 0
                    - [14] object-id: just for `np.ndarray`
                        it is removed in the `xr.DataArray`
                        and exists then in `xr.DataArray.coords` as index.
    :ivar wcs: World Coordinate System (WCS) object representing the coordinate
        transformation between pixel coordinates and celestial coordinates
        (e.g., right ascension and declination).
    :ivar precision: The precision of numerical values used in the SkyModel.
        Has to be of type np.float_.
    :ivar h5_file_connection: An open connection to an HDF5 (h5) file
        that can be used to store or retrieve data related to the SkyModel.
    """

    SOURCES_COLS = 14
    _STOKES_IDX: Dict[StokesType, int] = {
        "Stokes I": 2,
        "Stokes Q": 3,
        "Stokes U": 4,
        "Stokes V": 5,
    }
    COL_IDX: Dict[SkySourcesColName, int] = {  # according to docstring
        "ra": 0,
        "dec": 1,
        "stokes_i": 2,
        "stokes_q": 3,
        "stokes_u": 4,
        "stokes_v": 5,
        "ref_freq": 6,
        "spectral_index": 7,
        "rm": 8,
        "major": 9,
        "minor": 10,
        "pa": 11,
        "true_redshift": 12,
        "observed_redshift": 13,
        "id": 14,
    }
    COL_NAME: Dict[int, SkySourcesColName] = dict(zip(COL_IDX.values(), COL_IDX.keys()))

    def __init__(
        self,
        sources: Optional[_SkySourcesType] = None,
        wcs: Optional[WCS] = None,
        precision: Type[np.float_] = np.float64,
        h5_file_connection: Optional[h5py.File] = None,
    ) -> None:
        """
        Initialize a SkyModel object.

        Parameters
        ----------
        sources : {xarray.DataArray, np.ndarray}, optional
            List of all point sources in the sky.
            It can be provided as an `xarray.DataArray` or `np.ndarray`.
            If provided as an `np.ndarray`, the values are converted to
            `xarray.DataArray`.
        wcs : WCS, optional
            World Coordinate System (WCS) object representing the coordinate
            transformation between pixel coordinates and celestial coordinates.
        precision : np.dtype, optional
            The precision of numerical values used in the SkyModel.
            It should be a NumPy data type (e.g., np.float64).
        h5_file_connection : h5py.File, optional
            An open connection to an HDF5 (h5) file
            that can be used to store or retrieve data related to the SkyModel.
        """
        self.__sources_dim_sources = XARRAY_DIM_0_DEFAULT
        self.__sources_dim_data = XARRAY_DIM_1_DEFAULT
        self._sources: Optional[xr.DataArray] = None
        self.precision = precision
        self.wcs = wcs
        self.sources = sources  # type: ignore [assignment]
        self.h5_file_connection = h5_file_connection

    def __get_empty_sources(self, n_sources: int) -> xr.DataArray:
        empty_sources = np.hstack((np.zeros((n_sources, SkyModel.SOURCES_COLS)),))
        return xr.DataArray(
            empty_sources, dims=[self._sources_dim_sources, self._sources_dim_data]
        )

    def __set_sky_xarr_dims(self, sources: _SkySourcesType) -> None:
        if isinstance(sources, np.ndarray):
            pass  # nothing to do here
        elif isinstance(sources, xr.DataArray):  # checks xarray dims through setter
            self._sources_dim_sources, self._sources_dim_data = cast(
                Tuple[str, str], sources.dims
            )
        else:
            assert_never(f"{type(sources)} is not a valid `SkySourcesType`.")

    def close(self) -> None:
        """
        Closes the connection to the HDF5 file.

        This method closes the connection to the HDF5 file if it is open and
        sets the `h5_file_connection` attribute to `None`.
        """
        if self.h5_file_connection:
            self.h5_file_connection.close()
            self.h5_file_connection = None

    @staticmethod
    def copy_sky(sky: SkyModel) -> SkyModel:
        if sky.h5_file_connection is not None:
            h5_connection = sky.h5_file_connection
            sky.h5_file_connection = None
        else:
            h5_connection = None

        copied_sky = copy.deepcopy(sky)
        if h5_connection is not None:
            copied_sky.h5_file_connection = h5_connection

        return copied_sky

    def compute(self) -> None:
        """
        Loads the lazy data into a numpy array, wrapped in a xarray.DataArray.

        This method loads the lazy data from the sources into a numpy array,
        which is then wrapped in a xarray.DataArray object. It performs the computation
        necessary to obtain the actual data and stores it in the `_sources` attribute.
        After the computation is complete,
        it calls the `close` method to close the connection to the HDF5 file.

        Returns:
        None
        """
        if self.sources is not None:
            computed_sources = self.sources.compute()
        self.sources = None
        self.sources = computed_sources
        self.close()

    def _check_sources(self, sources: _SkySourcesType) -> None:
        self.__set_sky_xarr_dims(sources=sources)
        if isinstance(sources, np.ndarray):
            pass
        elif isinstance(sources, xr.DataArray):
            if self.sources is None:
                return None
            if self.sources.indexes.keys() != sources.indexes.keys():
                sky_keys = list(self.sources.indexes.keys())
                provided_keys = list(sources.indexes.keys())
                raise KaraboSkyModelError(
                    f"Provided index name {provided_keys} (object id's) of `sources` "
                    + f"don't match already existing indicex name {sky_keys}."
                )
        else:
            assert_never(f"{type(sources)} is not a valid `SkySourcesType`.")
        return None

    def to_sky_xarray(self, sources: _SkySourcesType) -> xr.DataArray:
        """Converts a `np.ndarray` or `xr.DataArray` to `SkyModel.sources`
            compatible `xr.DataArray`.

        Args:
            sources: Array to convert. Col-order see `SkyModel` description.

        Returns:
            `SkyModel.sources` compatible `xr.DataArray`.
        """
        if len(sources.shape) != 2:
            raise KaraboSkyModelError(
                "`sources` must be 2-dimensional but "
                + f"is {len(sources.shape)}-dimensional."
            )
        if sources.shape[1] < 3:
            raise KaraboSkyModelError(
                "`sources` requires min 3 cols: `right_ascension`, "
                + "`declination` and `stokes I flux`."
            )
        if isinstance(sources, xr.DataArray):
            if sources.shape[1] < SkyModel.SOURCES_COLS:
                fill = self.__get_empty_sources(n_sources=sources.shape[0])
                if len(sources.coords) > 0:
                    fill.coords[self._sources_dim_sources] = sources.coords[
                        self._sources_dim_sources
                    ]
                missing_cols = SkyModel.SOURCES_COLS - sources.shape[1]
                fill[:, :-missing_cols] = sources
                da = fill
            else:
                da = sources
        elif isinstance(sources, np.ndarray):
            # For numpy ndarrays, we delete the ID column of the sources
            if sources.shape[1] in (
                1 + SkyModel.SOURCES_COLS,
                13,
            ):  # sources have IDs. 13 is for backwards compatibility
                index_of_ids_column = sources.shape[1] - 1
                source_ids = sources[:, index_of_ids_column]
                sources = np.delete(sources, np.s_[index_of_ids_column], axis=1)  # type: ignore [assignment] # noqa: E501
                sources = sources.astype(self.precision)
                da = xr.DataArray(
                    sources,
                    dims=[self._sources_dim_sources, self._sources_dim_data],
                    coords={self._sources_dim_sources: source_ids},
                )
            elif sources.shape[1] == SkyModel.SOURCES_COLS:
                da = xr.DataArray(
                    sources,
                    dims=[self._sources_dim_sources, self._sources_dim_data],
                )
            else:
                fill = self.__get_empty_sources(n_sources=sources.shape[0])
                missing_cols = SkyModel.SOURCES_COLS - sources.shape[1]
                fill[:, :-missing_cols] = sources
                da = fill
        else:
            assert_never(f"{type(sources)} is not a valid `SkySourcesType`.")

        return da

    def add_point_sources(self, sources: _SkySourcesType) -> None:
        """Add new point sources to the sky model.

        :param sources: `np.ndarray` with shape (number of sources, 1 + SOURCES_COLS),
        where you can place the "source_id" at index SOURCES_COLS.
        OR an `xarray.DataArray` with shape (number of sources, SOURCES_COLS),
        where you can place the "source_id" at `xarray.DataArray.coord`
        or use `SkyModel.source_ids` later.

        The column indices correspond to:

            - [0] right ascension (deg)
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
            - [12] true redshift: defaults to 0
            - [13] observed redshift: defaults to 0
            - [14] source id (object): is in `SkyModel.source_ids` if provided
        """
        try:
            sds, sdd = self._sources_dim_sources, self._sources_dim_data
            self._check_sources(sources=sources)
            sky_sources = self.to_sky_xarray(sources=sources)
            if sky_sources.shape[1] > SkyModel.SOURCES_COLS:
                sky_sources[:, : SkyModel.SOURCES_COLS] = sky_sources[
                    :, : SkyModel.SOURCES_COLS
                ].astype(self.precision)
            if self.sources is not None:
                self.sources = xr.concat(
                    (self.sources, sky_sources), dim=self._sources_dim_sources
                )
            else:
                self._sources = sky_sources
        except BaseException:  # rollback of dim-names if sth goes wrong
            self._sources_dim_sources, self._sources_dim_data = sds, sdd
            raise

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
        - true redshift: defaults to 0
        - observed redshift: defaults to 0
        - source id (object): is in `SkyModel.source_ids` if provided

        :param path: file to read in
        :return: SkyModel
        """
        dataframe = pd.read_csv(path).to_numpy()

        if dataframe.shape[1] < 3:
            raise KaraboSkyModelError(
                f"CSV does not have the necessary 3 basic columns (RA, DEC and "
                f"STOKES I), but only {dataframe.shape[1]} columns."
            )

        if dataframe.shape[1] > SkyModel.SOURCES_COLS:
            print(
                f"""CSV has {dataframe.shape[1] - SkyModel.SOURCES_COLS + 1}
            too many rows. The extra rows will be cut off."""
            )

        sky = SkyModel(dataframe)
        return sky

    def to_np_array(self, with_obj_ids: bool = False) -> _NPSkyType:
        """
        Gets the sources as np.ndarray

        :param with_obj_ids: Option whether object ids should be included or not

        :return: the sources of the SkyModel as np.ndarray
        """
        if self.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before calling `to_np_array`."
            )
        if with_obj_ids:
            if self.source_ids is None:
                raise KaraboSkyModelError(
                    "There are no 'source_ids' available in `sources`."
                )
            return np.hstack(
                (
                    self.sources.to_numpy(),
                    self.source_ids[self._sources_dim_sources]
                    .to_numpy()
                    .reshape(-1, 1),
                )
            )
        else:
            return self.sources.to_numpy()

    def rechunk_array_based_on_self(self, array: xr.DataArray) -> xr.DataArray:
        if self.sources is None:
            raise KaraboSkyModelError("Rechunking of `sources` None is not allowed.")
        if self.sources.chunks is not None:
            chunk_size = max(self.sources.chunks[0][0], 1)
            chunks: Dict[str, Any] = {self._sources_dim_sources: chunk_size}
            array = array.chunk(chunks=chunks)
        else:
            pass
        return array

    @overload
    def filter_by_radius(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: Literal[False] = False,
    ) -> SkyModel:
        ...

    @overload
    def filter_by_radius(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: Literal[True],
    ) -> Tuple[SkyModel, NDArray[np.int_]]:
        ...

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
        copied_sky = SkyModel.copy_sky(self)
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

        if indices:
            return copied_sky, filtered_sources_idxs
        else:
            return copied_sky

    @overload
    def filter_by_radius_euclidean_flat_approximation(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: Literal[False] = False,
    ) -> SkyModel:
        ...

    @overload
    def filter_by_radius_euclidean_flat_approximation(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: Literal[True],
    ) -> Tuple[SkyModel, NDArray[np.int_]]:
        ...

    def filter_by_radius_euclidean_flat_approximation(
        self,
        inner_radius_deg: IntFloat,
        outer_radius_deg: IntFloat,
        ra0_deg: IntFloat,
        dec0_deg: IntFloat,
        indices: bool = False,
    ) -> Union[SkyModel, Tuple[SkyModel, NDArray[np.int_]]]:
        """
        Filters sources within an annular region using a flat Euclidean distance
        approximation suitable for large datasets managed by Xarray.

        This function is designed for scenarios where the dataset size precludes
        in-memory spherical geometry calculations. By leveraging a flat Euclidean
        approximation, it permits the use of Xarray's out-of-core computation
        capabilities, thus bypassing the limitations imposed by the incompatibility
        of `astropy.visualization.wcsaxes.SphericalCircle` with Xarray's data
        structures. Although this method trades off geometric accuracy against
        computational efficiency, it remains a practical choice for large angular
        fields where the curvature of the celestial sphere can be reasonably
        neglected.

        Parameters
        ----------
        inner_radius_deg : IntFloat
            The inner radius of the annular search region, in degrees.
        outer_radius_deg : IntFloat
            The outer radius of the annular search region, in degrees.
        ra0_deg : IntFloat
            The right ascension of the search region's center, in degrees.
        dec0_deg : IntFloat
            The declination of the search region's center, in degrees.
        indices : bool, optional
            If True, returns the indices of the filtered sources in addition to the
            SkyModel object. Defaults to False.

        Returns
        -------
        SkyModel or tuple of (SkyModel, NDArray[np.int_])
            The filtered SkyModel object, and optionally the indices of the filtered
            sources if `indices` is set to True.

        Raises
        ------
        KaraboSkyModelError
            If the `sources` attribute is not populated in the SkyModel instance prior
            to invoking this function.

        Notes
        -----
        Use this function for large sky models where a full spherical geometry
        calculation is not feasible due to memory constraints. It is particularly
        beneficial when working with Xarray and Dask, facilitating scalable data
        analysis on datasets that are too large to fit into memory.
        """
        copied_sky = SkyModel.copy_sky(self)

        if copied_sky.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before calling `filter_by_radius`."
            )

        # Calculate distances to phase center using flat Euclidean approximation
        x = (copied_sky[:, 0] - ra0_deg) * np.cos(np.radians(dec0_deg))
        y = copied_sky[:, 1] - dec0_deg
        distances_sq = np.add(np.square(x), np.square(y))

        # Filter sources based on inner and outer radius
        filter_mask = cast(  # distances_sq actually an xr.DataArray because x & y are
            xr.DataArray,
            (distances_sq >= np.square(inner_radius_deg))
            & (distances_sq <= np.square(outer_radius_deg)),
        ).compute()

        copied_sky.sources = copied_sky.sources[filter_mask]

        copied_sky.sources = self.rechunk_array_based_on_self(copied_sky.sources)

        if indices:
            filtered_indices = np.where(filter_mask)[0]
            return copied_sky, filtered_indices
        else:
            return copied_sky

    def filter_by_column(
        self,
        col_idx: int,
        min_val: IntFloat,
        max_val: IntFloat,
    ) -> SkyModel:
        """
        Filters the sky based on a specific column index

        :param col_idx: Column index to filter by
        :param min_val: Minimum value for the column
        :param max_val: Maximum value for the column
        :return sky: Filtered copy of the sky
        """
        copied_sky = SkyModel.copy_sky(self)
        if copied_sky.sources is None:
            raise KaraboSkyModelError(
                "`sources` is None, add sources before filtering."
            )

        # Create mask
        filter_mask = (copied_sky.sources[:, col_idx] >= min_val) & (
            copied_sky.sources[:, col_idx] <= max_val
        )
        filter_mask = self.rechunk_array_based_on_self(filter_mask).compute()

        # Apply the filter mask and drop the unmatched rows
        copied_sky.sources = copied_sky.sources.where(filter_mask, drop=True)

        return copied_sky

    def filter_by_flux(
        self,
        min_flux_jy: IntFloat,
        max_flux_jy: IntFloat,
    ) -> SkyModel:
        return self.filter_by_column(2, min_flux_jy, max_flux_jy)

    def filter_by_frequency(
        self,
        min_freq: IntFloat,
        max_freq: IntFloat,
    ) -> SkyModel:
        return self.filter_by_column(6, min_freq, max_freq)

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
        if self.sources is None:
            raise KaraboSkyModelError("Can't plot sky if `sources` is None.")
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
            flux = self[:, SkyModel._STOKES_IDX[stokes]].to_numpy()
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
                flux = cast(NDArray[np.float_], cfun(flux))

        # handle matplotlib kwargs
        # not set as normal args because default assignment depends on args
        if "vmin" not in kwargs:
            kwargs["vmin"] = np.nanmin(flux)  # type: ignore [arg-type]
        if "vmax" not in kwargs:
            kwargs["vmax"] = np.nanmax(flux)  # type: ignore [arg-type]

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
                data[self._sources_dim_sources], return_index=True
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
        sky: Union[_SkySourcesType, SkyModel],
        precision: PrecisionType = "double",
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
    ) -> Tuple[xr.DataArray, int]:
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

    @property
    def sources(self) -> Optional[xr.DataArray]:
        return self._sources

    @sources.setter
    def sources(self, value: Optional[_SkySourcesType]) -> None:
        """Sources setter.

        Does also allow numpy-arrays.
        But mypy doesn't allow getter and setter to have different dtypes (issue 3004).
        Just set "# type: ignore [assignment]" in case you don't have exctly an `xarray`

        Args:
            value: sources, `xarray.DataArray` or `np.ndarray`
        """
        self._sources = None
        self._sources_dim_sources = XARRAY_DIM_0_DEFAULT
        self._sources_dim_data = XARRAY_DIM_1_DEFAULT
        if value is not None:
            self.add_point_sources(sources=value)

    @property
    def _sources_dim_sources(self) -> str:
        if self.sources is None:
            return XARRAY_DIM_0_DEFAULT
        else:
            return self.__sources_dim_sources

    @_sources_dim_sources.setter
    def _sources_dim_sources(self, value: str) -> None:
        if (
            self._sources_dim_sources != XARRAY_DIM_0_DEFAULT
            and self._sources_dim_sources != value
        ):
            raise KaraboSkyModelError(
                "Provided dim_0 name is not consistent with existing dim-name of "
                + f"`sources`. Provided dim-name is {value} but existing is "
                + f"{self._sources_dim_sources}."
            )
        self.__sources_dim_sources = value

    @property
    def _sources_dim_data(self) -> str:
        if self.sources is None:
            return XARRAY_DIM_1_DEFAULT
        else:
            return self.__sources_dim_data

    @_sources_dim_data.setter
    def _sources_dim_data(self, value: str) -> None:
        if (
            self._sources_dim_data != XARRAY_DIM_1_DEFAULT
            and self._sources_dim_data != value
        ):
            raise KaraboSkyModelError(
                "Provided dim_1 name is not consistent with existing dim-name of "
                + f"`sources`. Provided dim-name is {value} but existing is "
                + f"{self._sources_dim_data}."
            )
        self.__sources_dim_data = value

    @property
    def source_ids(self) -> Optional[DataArrayCoordinates[xr.DataArray]]:
        if self.sources is not None and len(self.sources.coords) > 0:
            return self.sources.coords
        else:
            return None

    @source_ids.setter
    def source_ids(
        self,
        value: Optional[_SourceIdType],
    ) -> None:
        if self.sources is None or self._sources is None:
            raise KaraboSkyModelError(
                "Setting source-ids on empty `sources` is not allowed."
            )
        if value is None:
            if self._sources_dim_sources in self._sources.indexes:
                self._sources = self._sources.reset_index(
                    self._sources_dim_sources, drop=True
                )
        else:
            if isinstance(value, DataArrayCoordinates):
                if self._sources_dim_sources not in value:
                    raise KaraboSkyModelError(
                        f"Coord key {self._sources_dim_sources} not found in "
                        + "provided `DataArrayCoordinates`."
                    )
                n_provided = value[self._sources_dim_sources].shape[0]
            else:
                n_provided = len(value)
            if self.sources.shape[0] != n_provided:
                raise KaraboSkyModelError(
                    f"Number of provided sources of `source_ids` {n_provided} does "
                    f"not match the number of existing sources {self.sources.shape[0]}."
                )
            self.sources.coords[self._sources_dim_sources] = value

    def __getitem__(self, key: Any) -> xr.DataArray:
        """
        Allows to get access to self.sources in an np.ndarray like manner
        If casts the selected array/scalar to float64 if possible
        (usually if source_id is not selected)

        :param key: slice key

        :return: sliced self.sources
        """
        if self.sources is None:
            raise AttributeError("`sources` is None and therefore can't be accessed.")
        return self.sources[key]

    def __setitem__(
        self,
        key: Any,
        value: Union[_SkySourcesType, NPFloatLike],
    ) -> None:
        """
        Allows to set values in an np.ndarray like manner

        :param key: slice key
        :param value: values to store
        """
        if self.sources is None:
            raise KaraboSkyModelError("Can't acces `sources` because it's None.")
        # access `sources.getter`, not `sources.setter` which is fine
        self.sources[key] = value

    def save_sky_model_as_csv(self, path: str) -> None:
        """
        Save source array into a csv.
        :param path: path to save the csv file in.
        """
        if self.sources is None:
            raise KaraboSkyModelError("Can't save `sources` because they're None.")
        df = pd.DataFrame(self.sources)
        df["source id (object)"] = self.sources[self._sources_dim_sources].values
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
                "true redshift",
                "observed redshift",
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
        prefix_mapping: SkyPrefixMapping = SkyPrefixMapping(
            ra="Right Ascension",
            dec="Declination",
            stokes_i="Flux",
            observed_redshift="Observed Redshift",
        ),
        load_as: Literal["numpy_array", "dask_array"] = "dask_array",
        chunksize: Union[int, Literal["auto"]] = "auto",
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
        load_as : Literal["numpy_array", "dask_array"], default="dask_array"
            What type of array to load the data inside the xarray Data Array as.
        chunksize : Union[int, str], default=auto
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
        data_arrays: List[xr.DataArray] = []

        for field in fields(prefix_mapping):
            field_value: Optional[str] = getattr(prefix_mapping, field.name)
            if field_value is None:
                shape = f[prefix_mapping.ra].shape
                dask_array = da.zeros(shape, chunks=(chunksize,))  # type: ignore [attr-defined] # noqa: E501
            else:
                dask_array = da.from_array(f[field_value], chunks=(chunksize,))  # type: ignore [attr-defined] # noqa: E501
            data_arrays.append(xr.DataArray(dask_array, dims=[XARRAY_DIM_0_DEFAULT]))

        if load_as == "numpy_array":
            data_arrays = [x.compute() for x in data_arrays]
        sky = xr.concat(data_arrays, dim=XARRAY_DIM_1_DEFAULT)
        sky = sky.T
        chunks: Dict[str, Any] = {
            XARRAY_DIM_0_DEFAULT: chunksize,
            XARRAY_DIM_1_DEFAULT: sky.shape[1],
        }
        sky = sky.chunk(chunks=chunks)
        return SkyModel(sky, h5_file_connection=f)

    @classmethod
    def get_GLEAM_Sky(cls) -> SkyModel:
        """
        get_GLEAM_Sky - Returns a SkyModel object containing sources with flux densities
        from the GLEAM survey.

        Returns:
            SkyModel: A SkyModel object containing sources with flux densities.
        """
        survey = GLEAMSurveyDownloadObject()
        path = survey.get()

        encoded_freq = u.MHz
        unit_mapping = {
            "Jy": u.Jy,
            "Jy/beam": u.Jy / u.beam,
            "deg": u.deg,
            "arcsec": u.arcsec,
        }
        prefix_mapping = SkyPrefixMapping(
            ra="RAJ2000",
            dec="DEJ2000",
            stokes_i="Fp{0}",
            major="a{0}",
            minor="b{0}",
            pa="pa{0}",
            id="GLEAM",
        )
        units_sources = SkySourcesUnits(
            stokes_i=u.Jy / u.beam,
        )

        return cls.get_sky_model_from_fits(
            fits_file=path,
            prefix_mapping=prefix_mapping,
            unit_mapping=unit_mapping,
            units_sources=units_sources,
            encoded_freq=encoded_freq,
        )

    @classmethod
    def get_sky_model_from_fits(
        cls,
        fits_file: FilePathType,
        prefix_mapping: SkyPrefixMapping,
        unit_mapping: Dict[str, UnitBase],
        units_sources: Optional[SkySourcesUnits] = None,
        encoded_freq: Optional[UnitBase] = None,
        chunksize: Union[int, Literal["auto"]] = "auto",
        memmap: bool = False,
    ) -> SkyModel:
        """
        Reads data from a FITS file and creates an xarray Dataset containing
        information about celestial sources at given frequencies.

        Parameters
        ----------
        path : str
            Path to the input FITS file.
        prefix_mapping : SkyPrefixMapping
            Maps column names in the FITS file to column names
            used in the output Dataset.
            Keys should be the original column names, values should be the
            corresponding column names in the Dataset.
            Any column names not present in the dictionary will be excluded
            from the output Dataset.
        min_freq: float, optional
            Min frequency in Hz for pre-filtering.
        max_freq: float, optional
            Max frequency in Hz for pre-filtering.
        min_stokes_i: float, optional
            Min stokes-i in Jy for pre-filtering.
        max_stokes_i: float, optional
            Max stokes-i in Jy for pre-filtering.
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

        The `prefix_mapping` values should map the required columns to their
        corresponding column names in the input FITS file. For example:

        If a column name in the FITS file is not present in `prefix_mapping`
        values, it will be excluded from the output Dataset.
        """
        if units_sources is None:
            units_sources = SkySourcesUnits()
        with fits.open(fits_file, memmap=memmap) as hdul:
            data: FITS_rec = hdul[1].data
            cols: ColDefs = hdul[1].columns

        (
            prefix_mapping,
            num_formatted,
            names_and_freqs,
        ) = SkySourcesUnits.format_sky_prefix_mapping(
            cols=cols,
            prefix_mapping=prefix_mapping,
            encoded_freq=encoded_freq,
        )
        unit_scales = units_sources.get_unit_scales(
            cols=cols,
            unit_mapping=unit_mapping,
            prefix_mapping=prefix_mapping,
        )

        cache: Dict[str, NDArray[Any]] = dict()
        sources: Optional[xr.DataArray] = None
        col_data: NDArray[Any]
        if num_formatted == 0:
            num_formatted = 1  # to handle non-formatted case for for-loop
        for i in range(num_formatted):
            freq: Optional[float] = None
            sources_i: Optional[xr.DataArray] = None
            for spm_field in fields(prefix_mapping):
                if (
                    field_values := getattr(prefix_mapping, spm_field.name)
                ) is not None:
                    if not isinstance(
                        field_values, list
                    ):  # check if it's a formatted field or not
                        if (
                            field_values not in cache.keys()
                        ):  # to avoid io-operations of the same data
                            if field_values in unit_scales.keys():
                                cache[field_values] = (
                                    data[field_values] * unit_scales[field_values]
                                )
                            else:
                                cache[field_values] = data[field_values]
                        col_data = cache[field_values]
                    else:  # here, we deal with a formatted SkyPrefixMapping field
                        col_name: str = field_values[i]
                        col_freq = names_and_freqs[col_name]
                        if freq is None:
                            freq = col_freq
                        else:
                            if col_freq != freq:
                                raise RuntimeError(
                                    "Order of formatted sky-prefixes is not right! "
                                    + "This is probably an error of Karabo. Please "
                                    + "check `SkyPrefixMapping` list-order: "
                                    + f"{prefix_mapping}"
                                )
                        if col_name in unit_scales.keys():
                            col_data = (
                                data[col_name] * unit_scales[col_name]
                            )  # unit-scale correction
                        else:
                            col_data = data[col_name]
                    # create xarray for current encoding
                    if sources_i is None:
                        sources_i = xr.DataArray(
                            data=np.zeros(
                                shape=(col_data.shape[0], SkyModel.SOURCES_COLS - 1)
                            )
                        )
                    if (field_name := spm_field.name) == "id":
                        sources_i.coords[XARRAY_DIM_0_DEFAULT] = col_data
                    else:
                        col_idx = SkyModel.COL_IDX[field_name]  # type: ignore[index]
                        sources_i[:, col_idx] = col_data
            if sources_i is None:
                raise ValueError(
                    "This is an implementation error from Karabo. "
                    + "Please open an issue at "
                    + "https://github.com/i4Ds/Karabo-Pipeline"
                )
            if freq is not None:
                sources_i[:, cls.COL_IDX["ref_freq"]] = np.repeat(
                    freq, repeats=col_data.shape[0]
                )
            if sources is None:
                sources = sources_i
            else:
                sources = xr.concat((sources, sources_i), dim=XARRAY_DIM_0_DEFAULT)

        return SkyModel(sources=sources)

    @staticmethod
    def get_BATTYE_sky(which: Literal["full", "diluted"] = "diluted") -> SkyModel:
        raise DeprecationWarning(
            """This catalog has an error in the source flux values.
            This method will be removed in a future version.
            Please use get_sample_simulated_catalog() instead."""
        )

    @staticmethod
    def get_sample_simulated_catalog() -> SkyModel:
        """
        Downloads a sample simulated HI source catalog and generates a sky
        model using the downloaded data. The catalog size is around 8MB.

        Source:
        The simulated catalog data was provided by Luis Machado
        (https://github.com/lmachadopolettivalle) in collaboration
        with the ETHZ Cosmology Research Group.

        Returns:
            SkyModel: The corresponding sky model.
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
        survey = HISourcesSmallCatalogDownloadObject()
        path = survey.get()
        sky = SkyModel.get_sky_model_from_h5_to_xarray(path=path)
        if sky.sources is None:
            raise KaraboSkyModelError("`sky.sources` is None, which is unexpected.")

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
        raise NotImplementedError(
            "The current version of Karabo doesn't support the mightee-sky. "
            + "The progress regarding this issue can be tracked at "
            + "https://github.com/i4Ds/Karabo-Pipeline/issues/547"
        )
        # survey = MIGHTEESurveyDownloadObject()
        # path = survey.get()
        # prefix_mapping = SkyPrefixMapping(
        #     ra="RA",
        #     dec="DEC",
        #     stokes_i="NU_EFF",
        #     major="IM_MAJ",
        #     minor="IM_MIN",
        #     pa="IM_PA",
        #     id="NAME",
        # )

        # return SkyModel.get_sky_model_from_fits(
        #     path=path,
        #     frequencies=[76],
        #     prefix_mapping=prefix_mapping,
        #     concat_freq_with_prefix=False,
        #     filter_data_by_stokes_i=False,
        #     frequency_to_mhz_multiplier=1e6,
        #     memmap=False,
        # )

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
        return SkyModel(sky_array)

    @staticmethod
    def sky_test() -> SkyModel:
        """

        Construction of a sky model which can be used for testing and visualising the
        simulation with equal distributed point sources around the phase center ra=20,
        deg=-30.

        Returns:
             The test sky model.
        """
        sky = SkyModel()
        sky_data = np.zeros((81, SkyModel.SOURCES_COLS))
        a = np.arange(-32, -27.5, 0.5)
        b = np.arange(18, 22.5, 0.5)
        dec_arr, ra_arr = np.meshgrid(a, b)
        sky_data[:, 0] = ra_arr.flatten()
        sky_data[:, 1] = dec_arr.flatten()
        sky_data[:, 2] = 1

        sky.add_point_sources(sky_data)

        return sky

    @staticmethod
    def sky_from_h5_with_redshift_filtered(
        path: str, ra_deg: float, dec_deg: float, outer_rad: float = 5.0
    ) -> SkyModel:
        raise DeprecationWarning(
            """This method will be removed in a future release.
        To obtain the same functionality, use
        sky = SkyModel.get_sky_model_from_h5_to_xarray()
        and sky.filter_by_radius_euclidean_flat_approximation()."""
        )

    @overload
    def convert_to_backend(
        self,
        backend: Literal[SimulatorBackend.OSKAR] = SimulatorBackend.OSKAR,
        desired_frequencies_hz: Literal[None] = None,
        verbose: bool = False,
    ) -> SkyModel:
        ...

    @overload
    def convert_to_backend(
        self,
        backend: Literal[SimulatorBackend.RASCIL],
        desired_frequencies_hz: NDArray[np.float_],
        verbose: bool = False,
    ) -> List[SkyComponent]:
        ...

    def convert_to_backend(
        self,
        backend: SimulatorBackend = SimulatorBackend.OSKAR,
        desired_frequencies_hz: Optional[NDArray[np.float_]] = None,
        verbose: bool = False,
    ) -> Union[SkyModel, List[SkyComponent]]:
        """Convert an existing SkyModel instance into
        a format acceptable by a desired backend.
        backend: Determines how to return the SkyModel source catalog.
            OSKAR: return the current SkyModel instance, since methods in Karabo
            support OSKAR-formatted source np.array values.
            RASCIL: convert the current source array into a
            list of RASCIL SkyComponent instances.
        desired_frequencies_hz: List of frequencies corresponding to endpoints
        of desired frequency channels. This field is required
        to convert sources into RASCIL SkyComponents.
            The array contains endpoint frequencies for the desired channels.
            E.g. [100e6, 110e6, 120e6] corresponds to 2 frequency channels,
            which start at 100 MHz and 110 MHz, both with a bandwidth of 10 MHz.
        verbose: Determines whether to display additional print statements.
        """

        if backend is SimulatorBackend.OSKAR:
            if verbose is True:
                print(
                    """Desired backend is OSKAR.
                    Will not modify existing SkyModel instance."""
                )
            return self
        elif backend is SimulatorBackend.RASCIL:
            if verbose is True:
                print(
                    """Desired backend is RASCIL.
                    Will convert sources into a list of
                    RASCIL SkyComponent instances."""
                )

            desired_frequencies_hz = cast(NDArray[np.float_], desired_frequencies_hz)

            desired_frequencies_hz = np.sort(desired_frequencies_hz)

            # 1. Remove sources that fall outside all desired frequency channels
            # 2. Assign each source to the frequency channel closest
            # to its corresponding redshift (using np.digitize)
            # 3. For each source, create a SkyComponent,
            # with flux array equal to 0 on all channels,
            # except for its closest channel, where all its flux will belong
            # This is equivalent to having the source's SED equal to a delta function
            # at the frequency corresponding to its redshift,
            # which is true for line emission catalogues.
            redshift_limits = convert_frequency_to_z(
                np.array(
                    [
                        np.max(desired_frequencies_hz),
                        np.min(desired_frequencies_hz),
                    ]
                )
            )
            min_redshift, max_redshift = cast(
                Tuple[np.float_, np.float_], redshift_limits
            )

            if self.sources is None:
                return []

            assert self.sources is not None

            redshift_mask = (self.sources[:, 13] <= max_redshift) & (
                self.sources[:, 13] >= min_redshift
            )
            if verbose is True:
                print(min_redshift, max_redshift)
                print(self.sources)
            ras = self.sources[:, 0][redshift_mask]  # Degrees
            decs = self.sources[:, 1][redshift_mask]  # Degrees
            fluxes = self.sources[:, 2][redshift_mask]  # Jy * MHz
            redshifts = self.sources[:, 13][redshift_mask]
            if verbose is True:
                print(
                    f"""Reduced size of source catalog, after removing sources
                    outside of desired frequency range: {redshifts.shape}"""
                )

            # For each source, find the channel to which it belongs
            source_channel_indices = np.digitize(
                convert_z_to_frequency(redshifts),
                desired_frequencies_hz[
                    :-1
                ],  # Only provide starting points for frequency channels,
                # i.e. omit the ending of the last channel
                right=False,
            )

            # E.g. if channel starts are [1e8, 2e8],
            # then a source at frequency 1.5e8 should fall into the 0th channel.
            # However, digitize returns index 1 for such a source.
            # Therefore, we subtract 1 from the return value of np.digitize
            source_channel_indices -= 1

            skycomponents: List[SkyComponent] = []
            for ra, dec, flux, index in zip(
                ras,
                decs,
                fluxes,
                source_channel_indices,
            ):
                # 1 == npolarisations, fixed as 1 (stokesI) for now
                # TODO eventually handle full stokes source catalogs
                flux_array = np.zeros((len(desired_frequencies_hz) - 1, 1))

                # Access [0] since this is the stokesI flux,
                # and [index] to place the source's flux onto
                # the correct frequency channel (since this works for line emission)
                flux_array[index][0] = flux

                skycomponents.append(
                    SkyComponent(
                        direction=SkyCoord(
                            ra=ra,
                            dec=dec,
                            unit="deg",
                            frame="icrs",
                            equinox="J2000",
                        ),
                        frequency=desired_frequencies_hz[
                            :-1
                        ],  # Equal to image's channels, to prevent
                        # RASCIL from interpolating the fluxes
                        name=f"pointsource{ra}{dec}",
                        flux=flux_array,  # shape: nchannels, npolarisations
                        shape="Point",
                        polarisation_frame=PolarisationFrame("stokesI"),
                        params=None,
                    )
                )
            return skycomponents

        assert_never(backend)
