"""Module to access CASA Measurement Set metadata.

Details see MS v2.0: https://casacore.github.io/casacore-notes/229.html
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union, overload
from warnings import warn

import numpy as np
from casacore.tables import table as casa_table
from numpy.typing import NDArray
from typing_extensions import Self

_TDataClass = TypeVar("_TDataClass")
MS_VERSION = Literal["1.0", "2.0"]  # atm [09-2024]


def _get_cols(
    table: casa_table,
    *,
    subtable_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Utility function to extract casacore `table` KV.

    Args:
        table: Table to get columns from.
        subtable_id: Row number. If not specified, no specific row will be selected.

    Raises:
        RuntimeError: If no column is successfully extracted.

    Returns:
        Dict containing extracted table data with lower keys.
    """
    cols: Dict[str, Any] = {}
    name: str
    for name in table.colnames():
        try:
            col = table.getcol(columnname=name)
            if subtable_id is not None and (
                isinstance(col, list) or isinstance(col, np.ndarray)
            ):
                col = col[subtable_id]  # these two types are row specific
            cols[name.lower()] = col
        except RuntimeError:  # can happen if col-name not present in `table`
            cols[name.lower()] = None
    if len(cols) == 0:
        err_msg = "No cols found in `table`."
        raise RuntimeError(err_msg)
    return cols


def _create_table(
    table: casa_table,
    classtype: Type[_TDataClass],
    *,
    subtable_id: Optional[int] = None,
) -> _TDataClass:
    """Creates an instance of `classtype` from `table` values.

    If a dataclass field is missing in the table, this function will produce
        an according warning (mainly for devs).

    This function doesn't provide any runtime type safety measures because such an
        implementation is just too cumbersome and prone to errors.

    Args:
        table: Casacore table to extract data from.
        classtype: Dataclass to create an instance from.
        subtable_id: Row number. If not specified, no specific row will be selected.

    Returns:
        Created dataclass.
    """
    cols = _get_cols(table=table, subtable_id=subtable_id)
    field_types = {f.name: f.type for f in fields(classtype)}  # type: ignore[arg-type]
    field_names = list(field_types.keys())
    missing_fields = set(field_names) - set(cols.keys())
    if len(missing_fields) > 0:
        wmsg = f"{missing_fields=} are missing in `table` {cols.keys()=}."
        warn(wmsg, category=RuntimeWarning, stacklevel=2)
    fields_dict = {
        field_name: cols[field_name] if field_name in cols else None
        for field_name in field_names
    }
    return classtype(**fields_dict)


@dataclass
class _CasaTableABC(ABC):
    """Abstract table dataclass."""

    @classmethod
    @abstractmethod
    def _table_name(cls) -> str:
        """Table name for `from_ms` function.

        Returns:
            Table name.
        """
        raise NotImplementedError()

    @classmethod
    def _get_casa_table_instance(
        cls,
        ms_path: Union[str, Path],
    ) -> casa_table:
        """Gets a casa table instance from `_CasaTable ABC` inherited dataclass.

        Args:
            ms_path: Measurement set path.

        Returns:
            Casa table instance.
        """
        table_name = cls._table_name()
        if table_name == "MAIN":
            table_path = ms_path
        else:
            table_path = os.path.join(ms_path, table_name)
        with redirect_stdout(None):
            table_instance = casa_table(table_path)
        return table_instance

    @classmethod
    def from_ms(
        cls,
        ms_path: Union[str, Path],
    ) -> Self:
        """Gets CASA Measurement Sets from `ms_path`.

        Args:
            ms_path: Measurement set path.

        Returns:
            Dataclass containing CASA Measurement Set table metadata.
        """
        table_instance = cls._get_casa_table_instance(ms_path=ms_path)
        self = _create_table(
            table=table_instance,
            classtype=cls,
            subtable_id=None,  # None means full table
        )
        return self

    @classmethod
    def get_col(
        cls,
        ms_path: Union[str, Path],
        col: str,
        *,
        startrow: int = 0,
        nrow: int = -1,  # 500M is about 4GB for float64
    ) -> Any:  # usually list[str], NDArray[np.float64|np.int32|np.bool], dict[str,Any]
        """Get a specific column `col` of `ms_path`.

        This is a very useful function in particular for the main table since it
            can hold a large amount of data and loading a whole main table is just
            not reasonable.

        Args:
            ms_path: Measurement Set path.
            col: Column name (upper & lower-case allowed).
            startrow: Start row index.
            nrow: Number of rows to select. -1 is default and means all rows. This value
                can be used to avoid potential memory issues (dep on dtype of `col).

        Returns:
            According column of table.
        """
        table_instance = cls._get_casa_table_instance(ms_path=ms_path)
        return table_instance.getcol(col.upper(), startrow=startrow, nrow=nrow)

    @classmethod
    def nrows(
        cls,
        ms_path: Union[str, Path],
    ) -> int:
        """Returns the number of rows of the according dataclass MS table.

        Args:
            ms_path: Measurement Set path.

        Returns:
            Number of rows.
        """
        table_instance = cls._get_casa_table_instance(ms_path=ms_path)
        nrows = int(table_instance.nrows())
        return nrows


@dataclass
class MSMeta:
    """Utility class to extract metadata from CASA Measurement Sets.

    Args:
        observation: Observations.
        polarization: Polarizations.
        antenna: Antennas.
        field: Fields.
        spectral_window: Spectral Windows.
        ms_version: CASA MS version.
    """

    observation: MSObservationTable
    polarization: MSPolarizationTable
    antenna: MSAntennaTable
    field: MSFieldTable
    spectral_window: MSSpectralWindowTable
    ms_version: MS_VERSION

    @classmethod
    def from_ms(
        cls,
        ms_path: Union[str, Path],
    ) -> Self:
        """Gets CASA Measurement Sets metadata from `ms_path`.

        Note: each subtable ID like `obs_id` are set to 0 as default! This should be
            fine for Karabo creates MS visibilities. However, for external ones, be
            sure to set the ID's accordingly.

        Args:
            ms_path: Measurement set path.

        Returns:
            Dataclass containing CASA Measurement Sets metadata.
        """
        obs_table = MSObservationTable.from_ms(ms_path=ms_path)
        pol_table = MSPolarizationTable.from_ms(ms_path=ms_path)
        antenna_table = MSAntennaTable.from_ms(ms_path=ms_path)
        field_table = MSFieldTable.from_ms(ms_path=ms_path)
        spectral_window_table = MSSpectralWindowTable.from_ms(ms_path=ms_path)
        ms_version = MSMainTable.ms_version(ms_path=ms_path)
        return cls(
            observation=obs_table,
            polarization=pol_table,
            antenna=antenna_table,
            field=field_table,
            spectral_window=spectral_window_table,
            ms_version=ms_version,
        )


@dataclass
class MSMainTable(_CasaTableABC):
    """Utility class to extract main table metadata from CASA Measurement Sets.

    Each row (dim0) of the main table represents a measurement.

    The loading of this dataclass can get out of hand if the measurement set has many
        measurement sets. There we suggest to access the according column via `get_col`
        or directly from casacore to avoid eager loading.

    Args:
        time: Mid-point (not centroid) of data interval [s].
        antenna1: First antenna (index to sub-table).
        antenna2: Second antenna (index to sub-table).
        feed1: Feed for `antenna1`.
        feed2: Feed for `antenna2`.
        data_desc_id: Index to the DATA_DESCRIPTION sub-table.
        processor_id: Index to the PROCESSOR sub-table.
        field_id: Field identifier.
        interval: Data sampling interval. This is the nominal data interval and does not
            include the effects of bad data or partial integration [s].
        exposure: Effective data interval, including bad data and partial averaging [s].
        time_centroid: Time centroid [s].
        scan_number: Arbitrary scan number to identify data taken in the same logical
            scan. Not required to be unique.
        array_id: Subarray identifier, which identifies data in separate subarrays.
        observation_id: Observation identifier which identifies data from separate
            observations.
        state_id: State identifier.
        uvw: Coordinates for the baseline from ANTENNA2 to ANTENNA1, i.e. the baseline
            is equal to the difference POSITION2 - POSITION1. The UVW given are for the
            TIME_CENTROID, and correspond in general to the reference type for the
            PHASE_DIR of the relevant field. I.e. J2000 if the phase reference direction
            is given in J2000 coordinates. However, any known reference is valid. Note
            that the choice of baseline direction and UVW definition (W towards source
            direction; V in plane through source and system’s pole; U in direction of
            increasing longitude coordinate) also determines the sign of the phase of
            the recorded data.
    """

    time: NDArray[np.float64]
    antenna1: NDArray[np.int32]
    antenna2: NDArray[np.int32]
    feed1: NDArray[np.int32]
    feed2: NDArray[np.int32]
    data_desc_id: NDArray[np.int32]
    processor_id: NDArray[np.int32]
    field_id: NDArray[np.int32]
    interval: NDArray[np.float64]
    exposure: NDArray[np.float64]
    time_centroid: NDArray[np.float64]
    scan_number: NDArray[np.int32]
    array_id: NDArray[np.int32]
    observation_id: NDArray[np.int32]
    state_id: NDArray[np.int32]
    uvw: NDArray[np.float64]

    @classmethod
    def _table_name(cls) -> str:
        return "MAIN"

    @classmethod
    def ms_version(
        cls,
        ms_path: Union[str, Path],
    ) -> MS_VERSION:
        """Gets the CASA MS version of `ms_path`.

        Args:
            ms_path: Measurement set path.

        Returns:
            Version of CASA Measurement Set.
        """
        ms_path
        ct = cls._get_casa_table_instance(ms_path=ms_path)
        version: MS_VERSION = str(  # type: ignore[assignment]
            ct.getkeyword("MS_VERSION")
        )  # noqa: E501
        return version


@dataclass
class MSObservationTable(_CasaTableABC):
    """Utility class to extract observational metadata from CASA Measurement Sets.

    OBSERVATION_ID is the row-nr (dim0).

    Args:
        telescope_name: Telescope name.
        time_range: Start, end times [s] (div by 86400 to get mjd-utc).
        observer: Name of observer(s).
        log: Observing log.
        schedule_type: Schedule type.
        schedule: Project schedule.
        project: Project identification string.
        release_date: Target release date [s].
        flag_row: Row flag.
    """

    telescope_name: List[str]
    time_range: NDArray[np.float64]
    observer: List[str]
    log: Optional[Dict[str, Any]]
    schedule_type: List[str]
    schedule: Optional[Dict[str, Any]]
    project: List[str]
    release_date: NDArray[np.float64]
    flag_row: NDArray[np.bool_]

    @classmethod
    def _table_name(cls) -> str:
        return "OBSERVATION"


@dataclass
class MSPolarizationTable(_CasaTableABC):
    """Utility class to extract polarization metadata from CASA Measurement Sets.

    POLARIZATION_ID is the row-nr (dim0).

    Args:
        num_corr: Number of correlations.
        corr_type: Polarization of correlation; dim1=`num_corr`.
        corr_product: Receptor cross-products; dim1=2, dim2=`num_corr`.
        flag_row: True if data in this row are invalid, else False.
    """

    num_corr: NDArray[np.int32]
    corr_type: NDArray[np.int32]
    corr_product: NDArray[np.int32]
    flag_row: NDArray[np.bool_]

    @classmethod
    def _table_name(cls) -> str:
        return "POLARIZATION"

    @overload
    @classmethod
    def get_stokes_type(cls, corr_type: List[int]) -> List[str]:
        ...

    @overload
    @classmethod
    def get_stokes_type(cls, corr_type: int) -> str:
        ...

    @classmethod
    def get_stokes_type(cls, corr_type: Union[int, List[int]]) -> Union[str, List[str]]:
        """Gets the stokes type(s) from corr type(s).

        Args:
            corr_type: Correlation type(s).

        Returns:
            Stokes type(s).
        """
        stokes_types = [
            "Undefined",
            "I",
            "Q",
            "U",
            "V",
            "RR",
            "RL",
            "LR",
            "LL",
            "XX",
            "XY",
            "YX",
            "YY",
            "RX",
            "RY",
            "LX",
            "LY",
            "XR",
            "XL",
            "YR",
            "YL",
            "PP",
            "PQ",
            "QP",
            "QQ",
            "RCircular",
            "LCircular",
            "Linear",
            "Ptotal",
            "Plinear",
            "PFtotal",
            "PFlinear",
            "Pangle",
        ]
        n_types = len(stokes_types)
        corr_to_stokes = {i: stokes for i, stokes in enumerate(stokes_types)}
        if isinstance(corr_type, int):
            if corr_type > n_types:
                corr_type = 0
            return corr_to_stokes[corr_type]
        else:
            return [
                corr_to_stokes[corr] if corr < n_types else corr_to_stokes[0]
                for corr in corr_type
            ]


@dataclass
class MSAntennaTable(_CasaTableABC):
    """Utility class to extract antenna metadata from CASA Measurement Sets.

    ANTENNA_ID is the row-nr (dim0).

    Args:
        name: Antenna name.
        station: Station name.
        type: Antenna type.
        mount: Mount type:alt-az, equatorial, X-Y, orbiting, bizarre.
        position: Antenna X,Y,Z phase reference positions [m], dim1=3.
        offset: Axes oﬀset of mount to FEED REFERENCE point [m], dim1=3.
        dish_diameter: Diameter of dish.
        flag_row: True if data in this row are invalid, else False.
    """

    name: List[str]
    station: List[str]
    type: List[str]
    mount: List[str]
    position: NDArray[np.float64]
    offset: NDArray[np.float64]
    dish_diameter: NDArray[np.float64]
    flag_row: NDArray[np.bool_]

    @classmethod
    def _table_name(cls) -> str:
        return "ANTENNA"

    def baseline_dists(self) -> NDArray[np.float64]:
        """Computes the baseline euclidean distances in [m].

        This function assumes that `self.position` is in ITRF.

        Returns:
            Baseline distances nxn where n is the number of antennas.
        """
        dists: NDArray[np.float64] = np.linalg.norm(
            self.position[:, np.newaxis, :] - self.position[np.newaxis, :, :], axis=2
        )
        return dists


@dataclass
class MSFieldTable(_CasaTableABC):
    """Utility class to extract field metadata from CASA Measurement Sets.

    FIELD_ID is the row-nr (dim0).

    Args:
        name: Name of field.
        code: Special characteristics of field.
        time: Time origin for the directions and rates [s].
        num_poly: Series order.
        delay_dir: Direction of delay center [rad], shape=(2,`num_poly`+1).
            Can be expressed as a polynomial in time.
        phase_dir: Phase center [rad, shape=(2,`num_poly`+1).
            Can be expressed as a polynomial in time.
        reference_dir: Reference center [rad], shape=(2,`num_poly`+1).
            Can be expressed as a polynomial in time.
        source_id: Points to an entry in the optional SOURCE subtable, a value of −1
            indicates there is no corresponding source defined.
        flag_row: True if data in this row are invalid, else False.
    """

    name: List[str]
    code: List[str]
    time: NDArray[np.float64]
    num_poly: NDArray[np.int32]
    delay_dir: NDArray[np.float64]
    phase_dir: NDArray[np.float64]
    reference_dir: NDArray[np.float64]
    source_id: NDArray[np.int32]
    flag_row: NDArray[np.bool_]

    @classmethod
    def _table_name(cls) -> str:
        return "FIELD"


@dataclass
class MSSpectralWindowTable(_CasaTableABC):
    """Utility class to extract spectral window metadata from CASA Measurement Sets.

    SPECTRAL_WINDOW_ID is the row-nr (dim0).

    Args:
        name: Spectral window name; user specified.
        ref_frequency: The reference frequency [Hz]. A frequency representative of
            this spectral window, usually the sky frequency corresponding to the DC
            edge of the baseband. Used by the calibration system if a fixed scaling
            frequency is required or in algorithms to identify the observing band.
        chan_freq: Center frequencies [Hz] for each channel in the data matrix;
            dim1=`n_channels`. These can be frequency-dependent, to accommodate
            instruments such as acousto-optical spectrometers. Note that the channel
            frequencies may be in ascending or descending frequency order.
        chan_width: Channel width [Hz] of each spectral channel; dim1=`n_channels`.
        meas_freq_ref: Frequency measure reference for CHAN_FREQ. This allows a
            row-based reference for this column in order to optimize the choice of
            measure reference when Doppler tracking is used.
        effective_bw: The effective noise bandwidth of each spectral channel [Hz];
            dim1=`n_channels`.
        resolution: The effective spectral resolution [Hz] of each channel;
            dim1=`n_channels`.
        total_bandwidth: The total bandwidth [Hz] for this spectral window.
        net_sideband: The net sideband for this spectral window.
        if_conv_chain: Identification of the electronic signal path for the case of
            multiple (simultaneous) IFs. (e.g. VLA: AC=0, BD=1, ATCA: Freq1=0, Freq2=1).
        freq_group: The frequency group to which the spectral window belongs. This is
            used to associate spectral windows for joint calibration purposes.
        freq_group_name: The frequency group name; user specified.
        flag_row: True if the row does not contain valid data.
    """

    name: List[str]
    ref_frequency: NDArray[np.float64]
    chan_freq: NDArray[np.float64]
    chan_width: NDArray[np.float64]
    meas_freq_ref: NDArray[np.int32]
    effective_bw: NDArray[np.float64]
    resolution: NDArray[np.float64]
    total_bandwidth: NDArray[np.float64]
    net_sideband: NDArray[np.int32]
    if_conv_chain: NDArray[np.int32]
    freq_group: NDArray[np.int32]
    freq_group_name: List[str]
    flag_row: NDArray[np.bool_]

    @classmethod
    def _table_name(cls) -> str:
        return "SPECTRAL_WINDOW"
