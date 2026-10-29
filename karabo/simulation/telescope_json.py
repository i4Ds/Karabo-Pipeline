"""Read and write Karabo telescope JSON files.

The JSON holds the complete telescope description: centre, station positions,
element layouts and station metadata (names, diameters, SEFDs). Element layouts
and SEFD curves can be shared between stations by key. Station order is list
order and defines the OSKAR station index.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import numpy as np

from karabo.error import KaraboError
from karabo.simulation.east_north_coordinate import EastNorthCoordinate
from karabo.simulation.station import validated_sefd
from karabo.util._types import FilePathType

if TYPE_CHECKING:
    from karabo.simulation.telescope import Telescope

FORMAT = "karabo-telescope"
VERSION = 1

TTelescope = TypeVar("TTelescope", bound="Telescope")


def read(cls: Type[TTelescope], path: FilePathType) -> TTelescope:
    """Reads a telescope JSON file into an instance of `cls`."""
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise KaraboError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise KaraboError(f"{path} must contain a JSON object.")
    return from_dict(cls, data)


def write(telescope: Telescope, path: FilePathType) -> None:
    """Writes `telescope` as JSON to `path`."""
    text = json.dumps(to_dict(telescope), indent=2, allow_nan=False)
    # Keep each innermost number array, e.g. a position row, on one line.
    text = re.sub(
        r"\[\s+([-+0-9.eE,\s]+?)\s+\]",
        lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]",
        text,
    )
    with open(path, "w") as f:
        f.write(text + "\n")


def from_dict(cls: Type[TTelescope], data: Dict[str, Any]) -> TTelescope:
    """Builds an instance of `cls` from parsed telescope JSON."""
    if data.get("format") != FORMAT:
        raise KaraboError(f"'format' must be '{FORMAT}', got {data.get('format')!r}.")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise KaraboError(f"'version' must be a positive integer, got {version!r}.")
    if version > VERSION:
        raise KaraboError(
            f"Telescope JSON version {version} is not supported "
            f"(supported up to {VERSION}). Please update Karabo."
        )
    try:
        centre = data["centre"]
        lon_lat_alt = [
            float(centre["longitude_deg"]),
            float(centre["latitude_deg"]),
            float(centre.get("altitude_m", 0.0)),
        ]
        if not np.all(np.isfinite(lon_lat_alt)):
            raise ValueError(f"numbers must be finite, got {centre!r}")
        telescope = cls(*lon_lat_alt)
    except (KeyError, TypeError, ValueError) as e:
        raise KaraboError(f"Invalid 'centre': {e!r}") from e
    if data.get("name") is not None:
        telescope.name = str(data["name"])
    telescope.sefd_reference = data.get("sefd_reference")

    stations = data.get("stations")
    if not isinstance(stations, list) or len(stations) == 0:
        raise KaraboError("'stations' must be a non-empty list.")
    layouts = data.get("element_layouts") or {}
    curves = data.get("sefd_curves") or {}
    for i, station in enumerate(stations):
        try:
            elements = _resolve(station.get("elements", [[0.0, 0.0, 0.0]]), layouts)
            if not isinstance(elements, list) or len(elements) == 0:
                raise ValueError(
                    f"'elements' must be a non-empty list, got {elements!r}"
                )
            telescope.add_station(
                *_row(station["position_m"]),
                name=station.get("name"),
                dish_diameter_m=station.get("dish_diameter_m"),
                sefd=_resolve(station.get("sefd"), curves),
            )
            for element in elements:
                telescope.stations[i].add_station_antenna(
                    EastNorthCoordinate(*_row(element))
                )
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise KaraboError(f"Invalid station {i}: {e!r}") from e
    return telescope


def to_dict(telescope: Telescope) -> Dict[str, Any]:
    """Converts `telescope` into a JSON-serialisable dict.

    Identical element layouts and SEFD curves are written once and shared.
    """
    layouts: Dict[str, Any] = {}
    curves: Dict[str, Any] = {}
    stations: List[Dict[str, Any]] = []
    for i, station in enumerate(telescope.stations):
        if len(station.antennas) == 0:
            raise KaraboError(f"Station {i} has no elements.")
        entry: Dict[str, Any] = {}
        if station.name is not None:
            entry["name"] = station.name
        entry["position_m"] = _coordinate(station.position)
        entry["elements"] = _share(
            layouts, "layout", [_coordinate(a) for a in station.antennas]
        )
        if station.dish_diameter_m is not None:
            entry["dish_diameter_m"] = station.dish_diameter_m
        sefd = validated_sefd(station.sefd)
        if sefd is not None:
            entry["sefd"] = _share(
                curves,
                "sefd",
                {band: curve.tolist() for band, curve in sefd.items()},
            )
        stations.append(entry)

    data: Dict[str, Any] = {"format": FORMAT, "version": VERSION}
    if telescope.name is not None:
        data["name"] = telescope.name
    data["centre"] = {
        "longitude_deg": float(telescope.centre_longitude),
        "latitude_deg": float(telescope.centre_latitude),
        "altitude_m": float(telescope.centre_altitude),
    }
    if telescope.sefd_reference is not None:
        data["sefd_reference"] = telescope.sefd_reference
    data["element_layouts"] = layouts
    if curves:
        data["sefd_curves"] = curves
    data["stations"] = stations
    return data


def _row(values: Any) -> List[float]:
    if not isinstance(values, (list, tuple)) or len(values) not in (3, 6):
        raise ValueError(f"expected 3 or 6 numbers, got {values!r}")
    row = [float(v) for v in values]
    if not np.all(np.isfinite(row)):
        raise ValueError(f"numbers must be finite, got {values!r}")
    return row + [0.0] * (6 - len(row))


def _resolve(value: Any, shared: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        if value not in shared:
            raise KeyError(f"unknown shared key {value!r}")
        return shared[value]
    return value


def _coordinate(c: EastNorthCoordinate) -> List[float]:
    errors = [float(c.x_error), float(c.y_error), float(c.z_error)]
    return [float(c.x), float(c.y), float(c.z)] + (errors if any(errors) else [])


def _share(shared: Dict[str, Any], prefix: str, value: Any) -> str:
    for key, existing in shared.items():
        if existing == value:
            return key
    key = f"{prefix}{len(shared)}"
    shared[key] = value
    return key
