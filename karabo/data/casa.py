from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union

import numpy as np
from casacore.tables import table
from numpy.typing import NDArray
from typing_extensions import Self

from karabo.karabo_resource import HiddenPrints

_TDataClass = TypeVar("_TDataClass")


def _get_cols(
    table: table,
    *,
    row: int = 0,
    fill_str: bool = False,
) -> Dict[str, Any]:
    """Utility function to extract casacore `table` KV.

    Args:
        table: Table to get columns from.
        row: Row number.
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
            if isinstance(col, list) or isinstance(col, np.ndarray):
                col = col[row]  # these two types are row specific
            cols[name.lower()] = col
        except RuntimeError as e:  # can happen if col-name not present in `table`
            if fill_str and "SSMIndStringColumn" in str(e):
                cols[name.lower()] = ""
            else:
                pass  # col just doesn't get added because it doesn't exist
    if len(cols) == 0:
        err_msg = "No cols found in `table"
        raise RuntimeError(err_msg)
    return cols


def _create_dataclass(
    table: table,
    classtype: Type[_TDataClass],
    *,
    row: int = 0,
    fill_str: bool = False,
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
        row: Row to access (e.g. for observation, "OBSERVATION_ID" is the row number).
        fill_str: Fill non-existing string colnames with empty str?

    Returns:
        Created dataclass.
    """
    cols = _get_cols(table=table, row=row, fill_str=fill_str)
    field_types = {f.name: f.type for f in fields(CasaMSObservation)}
    field_names = list(field_types.keys())
    missing_fields = set(field_names) - set(cols.keys())
    if len(missing_fields) > 0:  # ensures safe access to `cols[field_name]`
        err_msg = f"{missing_fields=} are missing in `table` {cols.keys()=}."
        raise RuntimeError(err_msg)
    fields_dict = {field_name: cols[field_name] for field_name in field_names}
    return classtype(**fields_dict)


@dataclass
class CasaMSMeta:
    """Utility class to extract metadata from CASA Measurement Sets."""

    observation: CasaMSObservation

    @classmethod
    def from_ms(
        cls,
        ms_path: Union[str, Path],
        *,
        obs_id: int = 0,
    ) -> Self:
        """Gets CASA Measurement Sets metadata from `ms_path`.

        Args:
            ms_path: Measurement set path.
            obs_id: Observation id (is the row number of the MS).

        Returns:
            Dataclass containing CASA Measurement Sets metadata.
        """
        casa_ms_obs = CasaMSObservation.from_ms(ms_path=ms_path, obs_id=obs_id)
        return cls(observation=casa_ms_obs)


@dataclass
class CasaMSObservation:
    """Utility class to extract observational metadata from CASA Measurement Sets"""

    flag_row: bool
    log: str
    observer: str
    project: str
    release_date: float
    schedule: Dict[str, Any]
    schedule_type: str
    telescope_name: str
    time_range: NDArray[np.float64]

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
            obs_id: Observation id (is the row number of the MS).

        Returns:
            Dataclass containing CASA Measurement Sets observational metadata.
        """
        with HiddenPrints(stdout=True, stderr=False):
            obs_table = table(os.path.join(ms_path, "OBSERVATION"))
        self = _create_dataclass(
            table=obs_table, classtype=cls, row=obs_id, fill_str=True
        )
        self.time_range = self.time_range / 86400  # converted to mjd-utc
        return self
