"""General typings for the signal package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import tools21cm as t2c

from karabo.simulation.sky_model import SkyModel


class XFracDensLoaded(NamedTuple):
    """The Xfrac and dens files loaded into memory."""

    x_file: t2c.XfracFile
    """Loaded xfrac file object."""

    d_file: t2c.DensityFile
    """Loaded density file object."""

    x_frac: FloatArrayNxNxN
    """xfrac values in a 3D cube N*N*N."""

    dens: FloatArrayNxNxN
    """Density values in a 3D cub N*N*N."""

    z: float
    """Redshift value"""

    box_dims: float
    """Length of the volume along each direction in [Mpc]."""

    def xy_dims(self) -> tuple[FloatArrayN, FloatArrayN]:
        """
        Get the x, y dimensions of the loaded.

        Returns
        -------
        tuple[FloatArrayN, FloatArrayN]
            A tuple containing the x and y labels.
        """
        dx, dy = (
            self.box_dims / self.x_frac.shape[1],
            self.box_dims / self.x_frac.shape[2],
        )
        y, x = np.mgrid[
            slice(dy / 2, self.box_dims, dy), slice(dx / 2, self.box_dims, dx)
        ]
        return x, y


class XFracDensFilePair(NamedTuple):
    """A pair of matching XFrac and dens files (Same Redshift value)."""

    xfrac_path: Path
    """Path to the x-frac file."""

    dens_path: Path
    """Path to the dens file."""

    box_dims: float
    """Length of the volume along each direction in [Mpc]."""

    def load(self) -> XFracDensLoaded:
        """
        Load the files into memory.

        Returns
        -------
        XFracDensLoaded
            The loaded files.
        """
        x_file = t2c.XfracFile(self.xfrac_path)
        d_file = t2c.DensityFile(self.dens_path)

        return XFracDensLoaded(
            x_file=x_file,
            d_file=d_file,
            x_frac=x_file.xi,
            dens=d_file.cgs_density,
            z=x_file.z,
            box_dims=self.box_dims,
        )


class SegmentationOutput(NamedTuple):
    """Output of the segmentation."""

    image: Image3D
    xhii_stitch: npt.NDArray[np.bool_] | None
    mask_xhi: npt.NDArray[np.bool_]
    dt_smooth: npt.NDArray[np.float_]
    xhi_seg_err: npt.NDArray[np.float_] | None


@dataclass(frozen=True)
class BaseImage:
    """A general image, meant to be subclassed."""

    data: Annotated[npt.NDArray[np.float_], Literal["X", "Y"]]
    """Image data in a 2D array."""

    x_label: Annotated[npt.NDArray[np.float_], Literal["X"]]
    """X-labels."""

    y_label: Annotated[npt.NDArray[np.float_], Literal["Y"]]
    """Y-labels."""

    redshift: float
    """Redshift value. Negative number if not provided."""

    box_dims: float
    """Box dimensions used to create the image. Negative number if not provided."""


@dataclass(frozen=True)
class Image2D(BaseImage):
    """A 2D image."""


@dataclass(frozen=True)
class Image3D(BaseImage):
    """A 3D cube of images along the z-axis."""

    data: Annotated[npt.NDArray[np.float_], Literal["Z", "X", "Y"]]
    """Image data in a 3D cube."""

    z_label: Annotated[npt.NDArray[np.float_], Literal["Z"]]
    """Z-labels."""


@dataclass(frozen=True)
class Image2DOriented(Image2D):
    """A 2D image combined with a sky model."""

    sky_model: SkyModel
    """Sky Model for the orientation."""


EoRProfileT = Annotated[npt.NDArray[np.float_], Literal["N", 2]]

FloatArrayNxNxN = Annotated[npt.NDArray[np.float_], Literal["N", "N", "N"]]
FloatArrayN = Annotated[npt.NDArray[np.float_], Literal["N"]]
