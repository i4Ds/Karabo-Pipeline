"""Helpers for generating different types of signals."""

import multiprocessing
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units
from astropy.coordinates import Angle, SkyCoord
from scipy.ndimage import map_coordinates


# pylint: disable=too-many-arguments
def filter_dataframe_radec(
    df: pd.DataFrame,
    centre: SkyCoord,
    fov: Angle,
    ra_column: str,
    dec_column: str,
    wrap_offset: bool = False,
) -> tuple[pd.DataFrame, Angle, Angle]:
    """
    Filter a dataframe with RA-DEC coordinates.

    Expects the Center to be at (0°, 0°) with the RA axis having a range of (-180°,
    180°) and the DEC axis (-90°, 90°).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe that is to be filtered.
    centre : SkyCoord
        The center position of the filter.
    fov : Angle
        Field of View in the (RA, DEC) axis.
    ra_column : str
        Name of the RA column in the dataframe.
    dec_column : str
        Name of the DEC column in the dataframe.
    wrap_offset : bool
        If we reach a wrap point, if all the coordinates should be wrapped back into the
        positive space.

    Returns
    -------
    tuple[pd.DataFrame, Angle, Angle]
        The filtered dataframe (copy), lower left corner, upper right corner.
    """
    fov_ra: float = (fov.degree[0] / 2) * units.deg
    fov_dec: float = (fov.degree[1] / 2) * units.deg

    bottom_left = Angle([centre.ra - fov_ra, centre.dec - fov_dec], unit=units.deg)
    top_right = Angle([centre.ra + fov_ra, centre.dec + fov_dec], unit=units.deg)

    ra = df[ra_column]
    dec = df[dec_column]

    ra_filter = _filter_series(
        ra,
        Angle([bottom_left.degree[0], top_right.degree[0]], unit=units.deg),
        Angle([-180, 180], unit=units.deg),
    )
    dec_filter = _filter_series(
        dec,
        Angle([bottom_left.degree[1], top_right.degree[1]], unit=units.deg),
        Angle([-90, 90], unit=units.deg),
    )

    data = df[ra_filter & dec_filter].copy()

    if wrap_offset:
        _wrap_data(data, ra_column, dec_column, bottom_left, top_right)

    return data, bottom_left, top_right


def _wrap_data(
    data: pd.DataFrame,
    ra_column: str,
    dec_column: str,
    bottom_left: Annotated[Angle, Literal[2]],
    top_right: Annotated[Angle, Literal[2]],
) -> None:
    """
    Wrap any data that was fetched from a "wrapped" dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The data that may contain the "wrapped" data.
    ra_column : str
        Column name for the RA column
    dec_column : str
        Column name for the DEC column
    bottom_left : Annotated[Angle, Literal[2]]
        Bottom left corner Angle of the data
    top_right : Annotated[Angle, Literal[2]]
        Top right corner Angle of the data
    """
    if (top_right.degree[0] % 180) < bottom_left.degree[0]:
        ra_offset = 180 - top_right.degree[0]
        data[ra_column] += ra_offset
        data.loc[data[ra_column] <= -180, ra_column] += 360

    if (top_right.degree[1] % 90) < bottom_left.degree[1]:
        dec_offset = 90 - top_right.degree[1]
        data[dec_column] += dec_offset
        data.loc[data[dec_column] <= -90, dec_column] += 180


def _filter_series(
    series: Annotated[pd.Series, Literal["N"]],
    angle: Annotated[Angle, Literal[2]],
    data_range: Annotated[Angle, Literal[2]],
) -> Annotated[pd.Series, Literal["N"]]:
    """
    Filter Series based on an angle.

    If the first angle to be filtered is less than `data_range[0]`, it gets wrapped
    around to the `range[1]`.

    If an angle of angle = (-30°, 20°) and data_range[0]=0°, data_range[1]=180° is
    passed, the following range will be selected:

    150° to 180° plus 0° to 180°

    If an angle of angle = (-110°, -60°) and data_range[0]=-90°, data_range[1]=90° is
    passed, the following range will be selected:

    70° to 90° plus -90° to -60°

    Parameters
    ----------
    series : Annotated[pd.Series, Literal["N"]]
        The series to be filtered.
    angle : Annotated[Angle, Literal[2]]
        The angle filter the coordinates between. Requires two angles to be present.
    data_range : Annotated[Angle, Literal[2]]
        At what angle data series starts / ends.

    Returns
    -------
    Annotated[pd.Series, Literal["N"]]
        Filter series for selecting only the requested angle. Only contains boolean
        values.
    """
    start = data_range.degree[0]
    end = data_range.degree[1]
    angle_rng = end - start

    if angle.degree[0] >= start:
        if angle.degree[1] <= end:
            return (series >= angle.degree[0]) & (series < angle.degree[1])

        # Map to the right
        offset = angle.degree[1] - angle_rng
        # fmt: off
        return (
            (series >= angle.degree[0]) & (series <= end)
            | (series >= start) & (series < offset)
        )
        # fmt: on

    # Map to the left
    # fmt: off
    return (
        (series >= angle_rng + angle.degree[0]) & (series <= end)
        | (series >= start) & (series < angle.degree[1])
    )
    # fmt: on


# pylint: disable=too-many-locals
def map_radec_datapoints_to_grid(
    data: pd.DataFrame,
    grid_size: tuple[Annotated[int, Literal["X"]], Annotated[int, Literal["Y"]]],
    ra_column: str,
    dec_column: str,
    intensity_column: str,
    par_count: int = 5,
) -> Annotated[npt.NDArray[np.float_], Literal["X", "Y"]]:
    """
    Map the given datapoints with a destination to source mapping.

    For each pixel in the destination grid, the equivalent degree range in the
    source will be summed together and set in the destination grid.

    Parameters
    ----------
    data : pd.DataFrame,
        The data that is to be plotted onto the grid.
    grid_size : tuple[Annotated[int, Literal["X"]], Annotated[int, Literal["Y"]]]
        Size of the output grid.
    ra_column : str
        Name of the RA column in the dataframe.
    dec_column : str
        Name of the DEC column in the dataframe.
    intensity_column : str
        Name of the column to use for the intensities.
    par_count : int, optional
        How many processes to use to map the datapoints. By default 5.

    Returns
    -------
    Annotated[npt.NDArray[np.float_], Literal["X", "Y"]]
        A 2D numpy array representing an image with the dimensions of the grid_size
        parameter.
    """
    x_min, x_max = data[ra_column].agg(["min", "max"])
    y_min, y_max = data[dec_column].agg(["min", "max"])

    x_delta = (x_max - x_min) / grid_size[0]
    y_delta = (y_max - y_min) / grid_size[1]

    width = grid_size[1]

    args: list[
        tuple[pd.DataFrame, str, str, str, float, float, float, float, int, int, int]
    ] = []
    for x in range(int(grid_size[0] / par_count)):
        args.append(
            (
                data,
                ra_column,
                dec_column,
                intensity_column,
                x_min,
                x_delta,
                y_min,
                y_delta,
                width,
                x * par_count,
                par_count,
            )
        )

    with multiprocessing.Pool(par_count) as p:
        chunks: list[npt.NDArray[np.float_]] = p.starmap(
            _map_radec_datapoints_rows,
            args,
        )
    return np.nan_to_num(np.vstack(chunks))


def _map_radec_datapoints_rows(
    data: pd.DataFrame,
    ra_column: str,
    dec_column: str,
    intensity_column: str,
    x_min: float,
    x_delta: float,
    y_min: float,
    y_delta: float,
    width: int,
    x_start: int,
    par_count: int,
) -> npt.NDArray[np.float_]:
    grid = np.zeros((par_count, width))

    for x in range(par_count):
        x_beg = x_min + (x + x_start) * x_delta
        x_end = x_beg + x_delta
        ra_filter = (data[ra_column] >= x_beg) & (data[ra_column] < x_end)

        for y in range(width):
            y_beg = y_min + y * y_delta
            y_end = y_beg + y_delta
            dec_filter = (data[dec_column] >= y_beg) & (data[dec_column] < y_end)

            filt = data.loc[ra_filter & dec_filter, intensity_column]
            pixel_value = np.median(filt[filt > 0])
            grid[x, y] = pixel_value

    return grid


SciPyInterpolationModes = Literal[
    "reflect",
    "grid-mirror",
    "constant",
    "grid-constant",
    "nearest",
    "mirror",
    "grid-wrap",
    "wrap",
]


def interpolate_image(
    image: Annotated[npt.NDArray[np.float_], Literal["X", "Y"]],
    new_size: tuple[int, int],
    mode: SciPyInterpolationModes = "constant",
) -> Annotated[npt.NDArray[np.float_], Literal["N", "M"]]:
    """
    Interpolate an image from it's original size to `new_size`.

    Parameters
    ----------
    image : Annotated[npt.NDArray[np.float_], Literal["X", "Y"]]
        The image that is to be interpolated.
    new_size : tuple[int, int]
        The desired output size of the image
    mode : SciPyInterpolationModes, optional
        The interpolation mode, by default "constant".

    Returns
    -------
    Annotated[npt.NDArray[np.float_], Literal["N", "M"]]
        The interpolated input image with it's new dimensions.
    """
    original_shape = image.shape

    row_indices = np.linspace(0, original_shape[0] - 1, new_size[0])
    column_indices = np.linspace(0, original_shape[1] - 1, new_size[1])

    grid = np.meshgrid(row_indices, column_indices, indexing="ij")
    coordinates = np.stack(grid, axis=-1)

    interpolated_array = map_coordinates(image, coordinates.T, order=1, mode=mode)
    return interpolated_array.reshape(new_size).T
