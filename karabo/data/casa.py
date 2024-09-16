"""Module to access CASA Measurement Set metadata.

Details see MS v2.0: https://casacore.github.io/casacore-notes/229.html
"""

from __future__ import annotations

import os
from contextlib import redirect_stdout
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from warnings import warn

import numpy as np
from casacore.tables import table
from numpy.typing import NDArray
from typing_extensions import Self

_TDataClass = TypeVar("_TDataClass")


def _get_cols(
    table: table,
    *,
    subtable_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Utility function to extract casacore `table` KV.

    Args:
        table: Table to get columns from.
        subtable_id: Row number. If not specified, no specific row will be selected.
        fill_str: Fill non-existing string colnames with empty str?

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
        err_msg = "No cols found in `table"
        raise RuntimeError(err_msg)
    return cols


def _create_table(
    table: table,
    classtype: Type[_TDataClass],
    *,
    subtable_id: Optional[int] = None,
) -> _TDataClass:
    """Creates an instance of of `classtype` from `table` values.

    This function assumes that ALL fields of `classtype` are available in `table` in
        either lower OR upper as table keys. It's not an issue whether `table` has
        more keys than `classtype`.

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
class CasaMSMeta:
    """Utility class to extract metadata from CASA Measurement Sets."""

    observation: MSObservationTable
    polarization: MSPolarizationTable
    antennas: MSAntennaTable

    @classmethod
    def from_ms(
        cls,
        ms_path: Union[str, Path],
        *,
        obs_id: int = 0,
        pol_id: int = 0,
    ) -> Self:
        """Gets CASA Measurement Sets metadata from `ms_path`.

        Note: each subtable ID like `obs_id` are set to 0 as default! This should be
            fine for Karabo creates MS visibilities. However, for external ones, be
            sure to set the ID's accordingly.

        Args:
            ms_path: Measurement set path.
            obs_id: Observation ID (is the row number of the MS).
            pol_id: Polarization ID (index into polarization sub-table).

        Returns:
            Dataclass containing CASA Measurement Sets metadata.
        """
        casa_ms_obs = MSObservationTable.from_ms(ms_path=ms_path, obs_id=obs_id)
        casa_ms_pol = MSPolarizationTable.from_ms(ms_path=ms_path, pol_id=pol_id)
        casa_ms_antennas = MSAntennaTable.from_ms(ms_path=ms_path)
        return cls(
            observation=casa_ms_obs,
            polarization=casa_ms_pol,
            antennas=casa_ms_antennas,
        )


@dataclass
class MSObservationTable:
    """Utility class to extract observational metadata from CASA Measurement Sets.

    Args:
        telescope_name: Telescope name.
        time_range: Start, end times [s].
        observer: Name of observer(s).
        log: Observing log.
        schedule_type: Schedule type.
        schedule: Project schedule.
        project: Project identification string.
        release_date: Target release date [s].
        flag_row: Row flag.
    """

    telescope_name: str
    time_range: NDArray[np.float64]
    observer: str
    log: Optional[Dict[str, Any]]
    schedule_type: str
    schedule: Dict[str, Any]
    project: str
    release_date: Optional[float]
    flag_row: bool

    @classmethod
    def from_ms(
        cls,
        ms_path: Union[str, Path],
        *,
        obs_id: int = 0,
    ) -> Self:
        """Gets CASA Measurement Sets observational metadata from `ms_path`.

        Args:
            ms_path: Measurement set path.
            obs_id: Observation ID (is the row number of the MS).

        Returns:
            Dataclass containing CASA Measurement Sets observational metadata.
        """
        with redirect_stdout(None):
            obs_table = table(os.path.join(ms_path, "OBSERVATION"))
        self = _create_table(
            table=obs_table,
            classtype=cls,
            subtable_id=obs_id,
        )
        self.time_range = self.time_range / 86400  # converted to mjd-utc
        return self


@dataclass
class MSPolarizationTable:
    """Utility class to extract polarization metadata from CASA Measurement Sets.

    Args:
        num_corr: Number of correlations.
        corr_type: Polarization of correlation, shape=(`num_corr`).
        corr_product: Receptor cross-products, shape=(2, `num_corr`).
        flag_row: Row flag.
    """

    num_corr: int
    corr_type: NDArray[np.int32]
    corr_product: NDArray[np.int32]
    flag_row: bool

    @classmethod
    def from_ms(
        cls,
        ms_path: Union[str, Path],
        *,
        pol_id: int = 0,
    ) -> Self:
        """Gets CASA Measurement Sets polarization metadata from `ms_path`.

        Args:
            ms_path: Measurement set path.
            pol_id: Polarization ID (index into polarization sub-table).

        Returns:
            Dataclass containing CASA Measurement Sets polarization metadata.
        """
        with redirect_stdout(None):
            pol_table = table(os.path.join(ms_path, "POLARIZATION"))
        self = _create_table(
            table=pol_table,
            classtype=cls,
            subtable_id=pol_id,
        )
        return self


@dataclass
class MSAntennaTable:
    """Utility class to extract antenna metadata from CASA Measurement Sets.

    This class is contains all antennas and therefore is not a subtable.

    Args:
        name: Antenna names.
        station: Station names.
        type: Antenna types.
        mount: Mount type:alt-az, equatorial, X-Y, orbiting, bizarre.
        position: Antenna X,Y,Z phase reference positions [m], shape=(`n_antennas`,3).
        offset: Axes oﬀset of mount to FEED REFERENCE point [m], shape=(`n_antennas`,3).
        dish_diameter: Diameter of dish, shape=(`n_antennas`).
        flag_row: Row ﬂag, shape=(`n_antennas`).
    """

    name: List[str]
    station: List[str]
    type: List[str]
    mount: List[str]
    position: NDArray[np.float64]
    offset: NDArray[np.float64]
    dish_diameter: NDArray[np.float64]
    flag_row: NDArray[np.bool_]  # TODO: Change when np >=2.0

    @classmethod
    def from_ms(
        cls,
        ms_path: Union[str, Path],
    ) -> Self:
        """Gets CASA Measurement Sets antenna metadata from `ms_path`.

        Args:
            ms_path: Measurement set path.
            antenna_id: Antenna ID (index into antenna sub-table).

        Returns:
            Dataclass containing CASA Measurement Sets antenna metadata.
        """
        with redirect_stdout(None):
            antenna_table = table(os.path.join(ms_path, "ANTENNA"))
        self = _create_table(
            table=antenna_table,
            classtype=cls,
            subtable_id=None,
        )
        return self

    @classmethod
    def n_antennas(
        cls,
        ms_path: Union[str, Path],
    ) -> int:
        """Gets the number of antennas of `ms_path`.

        This function doesn't consider anything like different antenna types.

        Args:
            ms_path: Measurement set path.

        Returns:
            Number of antennas.
        """
        with redirect_stdout(None):
            antenna_table = table(os.path.join(ms_path, "ANTENNA"))
        pos_array: NDArray[np.float64] = antenna_table.getcol("POSITION")
        return pos_array.shape[0]
