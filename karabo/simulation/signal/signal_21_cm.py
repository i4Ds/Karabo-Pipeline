"""21cm Signal simulation."""
import re
from pathlib import Path
from typing import Callable, Final, Optional

import numpy as np
import requests
import tools21cm as t2c

from karabo.data.external_data import DownloadObject
from karabo.error import KaraboError
from karabo.simulation.signal.base_signal import BaseSignal
from karabo.simulation.signal.typing import Image3D, XFracDensFilePair


class Signal21cm(BaseSignal[Image3D]):
    """
    21cm Signal simulation wrapper.

    Examples
    --------
    >>> from karabo.simulation.signal.plotting import SignalPlotting
    >>> redshifts = Signal21cm.available_redshifts()
    >>> z1 = Signal21cm.get_xfrac_dens_file(z=redshifts[0], box_dims=244 / 0.7)
    >>> z2 = Signal21cm.get_xfrac_dens_file(z=redshifts[1], box_dims=244 / 0.7)
    >>> sig = Signal21cm([z1, z2])
    >>> signal_images = sig.simulate()
    >>> fig = SignalPlotting.brightness_temperature(signal_images[0])
    >>> fig.savefig("brightness_temperature.png")

    >>> from karabo.simulation.signal.plotting import SignalPlotting
    >>> z = Signal21cm.randomized_lightcones(200, 8)
    >>> SignalPlotting.brightness_temperature(z, z_layer=100)
    """

    XFRAC_URL = "https://ttt.astro.su.se/~gmell/244Mpc/244Mpc_f2_0_250/{xfrac_name}"
    DENS_URL = (
        "https://ttt.astro.su.se/~gmell/244Mpc/densities/nc250/coarser_densities/"
        + "{dens_name}"
    )

    def __init__(self, files: list[XFracDensFilePair]) -> None:
        """
        21cm Signal simulation.

        Parameters
        ----------
        files : list[XFracDensFilePair]
            The xfrac and dens files to be used in the
        """
        self.files: Final[list[XFracDensFilePair]] = files

    def simulate(self) -> list[Image3D]:
        """
        Simulate the 21cm signal as a 3D intensity cube.

        Returns
        -------
        list[Image3D]
            A list of 3D image cubes, based on the `self.files` list of provided xfrac
            and dens files. The pixel values are in Kelvin.

        Raises
        ------
        KaraboError
            If a pair of xfrac and dens files do not have the same redshift values.
        """
        cubes: list[Image3D] = []

        for file in self.files:
            loaded = file.load()

            if (redshift := loaded.x_file.z) != loaded.d_file.z:
                raise KaraboError(
                    "The redshift of the xfrac and dens files are not the same", file
                )

            x_frac = loaded.x_file.xi
            dens = loaded.d_file.cgs_density

            dz, dx, dy = (
                loaded.box_dims / x_frac.shape[0],
                loaded.box_dims / x_frac.shape[1],
                loaded.box_dims / x_frac.shape[2],
            )
            z = np.arange(dz / 2, loaded.box_dims, dz)
            y = np.arange(dy / 2, loaded.box_dims, dy)
            x = np.arange(dx / 2, loaded.box_dims, dx)

            # calc_dt returns mK, not Kelvin!
            d_t = t2c.calc_dt(x_frac, dens, redshift)
            d_t_subtracted = t2c.subtract_mean_signal(d_t, 0) / 1000
            cubes.append(
                Image3D(
                    data=d_t_subtracted,
                    x_label=x,
                    y_label=y,
                    z_label=z,
                    redshift=redshift,
                    box_dims=loaded.box_dims,
                )
            )

        return cubes

    @staticmethod
    def default_r_hii(redshift: float) -> float:
        """
        Lightcone HII region size calculation function (default implementation).

        Parameters
        ----------
        redshift : float
            Redshift, to determine the radius for.

        Returns
        -------
        float
            Lightcone radius.
        """
        return 30 * np.exp(-(redshift - 7.0) / 3)

    # pylint: disable=too-many-locals
    @classmethod
    def randomized_lightcones(
        cls,
        n_cells: int,
        z: float,
        r_hii: Optional[Callable[[float], float]] = None,
    ) -> Image3D:
        """
        Generate an image with randomized lightcones.

        Parameters
        ----------
        n_cells : int
            The count of cells to produce.
        z : float
            The redshift value for this image.
        r_hii : Callable[[float], float], optional
            Radius function of the HII region. By default None, resulting in the
            execution of `Signal21cm.default_r_hii`

        Notes
        -----
        Implementation according to
        https://tools21cm.readthedocs.io/examples/lightcone.html

        Returns
        -------
        Image3D
            The generated cube with multiple lightcones.
        """
        cube = np.zeros((n_cells, n_cells, n_cells))
        xx, yy, zz = np.meshgrid(
            np.arange(n_cells), np.arange(n_cells), np.arange(n_cells), sparse=True
        )

        if r_hii is None:
            r_hii = Signal21cm.default_r_hii

        r = r_hii(z)
        r2 = (xx - n_cells / 2) ** 2 + (yy - n_cells / 2) ** 2 + (zz - n_cells / 2) ** 2
        xx_0 = n_cells // 2
        yy_0 = n_cells // 2
        zz_0 = n_cells // 2
        cube0 = np.zeros((n_cells, n_cells, n_cells))
        cube0[r2 <= r**2] = 1
        cube0 = np.roll(
            np.roll(np.roll(cube0, -xx_0, axis=0), -yy_0, axis=1), -zz_0, axis=2
        )

        # Bubble 1
        xx1, yy1, zz1 = int(n_cells / 2), int(n_cells / 2), int(n_cells / 2)
        cube = cube + np.roll(
            np.roll(np.roll(cube0, xx1, axis=0), yy1, axis=1), zz1, axis=2
        )

        # Bubble 2
        xx2, yy2, zz2 = int(n_cells / 2), int(n_cells / 4), int(n_cells / 16)
        cube = cube + np.roll(
            np.roll(np.roll(cube0, xx2, axis=0), yy2, axis=1), zz2, axis=2
        )

        # Bubble 3
        xx3, yy3, zz3 = int(n_cells / 2 + 10), int(-n_cells / 4), int(-n_cells / 32)
        cube = cube + np.roll(
            np.roll(np.roll(cube0, xx3, axis=0), yy3, axis=1), zz3, axis=2
        )

        return Image3D(
            data=cube,
            x_label=np.arange(0, n_cells, 1, dtype=float),
            y_label=np.arange(0, n_cells, 1, dtype=float),
            z_label=np.arange(0, n_cells, 1, dtype=float),
            redshift=z,
            box_dims=0,
        )

    @staticmethod
    def get_xfrac_dens_file(z: float, box_dims: float) -> XFracDensFilePair:
        """
        Get the xfrac and dens files from the server.

        They are downloaded and cached on the first access.

        Parameters
        ----------
        z : float
            Redshift value.
        box_dims : float
            Box dimensions used for these files.

        Returns
        -------
        XFracDensFilePair
            A tuple of xfrac and dens files.
        """
        xfrac_name = f"xfrac3d_{z:.3f}.bin"
        dens_name = f"{z:.3f}n_all.dat"

        xfrac_path = DownloadObject(
            xfrac_name,
            Signal21cm.XFRAC_URL.format(xfrac_name=xfrac_name),
        ).get()
        dens_path = DownloadObject(
            dens_name,
            Signal21cm.DENS_URL.format(dens_name=dens_name),
        ).get()

        return XFracDensFilePair(
            xfrac_path=Path(xfrac_path), dens_path=Path(dens_path), box_dims=box_dims
        )

    @classmethod
    def available_redshifts_xfrac(cls) -> list[float]:
        """
        Get all available redshifts for xfrac files.

        Returns
        -------
        list[float]
            List of all available redshifts for xfrac files.
        """
        resp = requests.get(Signal21cm.XFRAC_URL.format(xfrac_name=""), timeout=30)
        all_redshifts_xfrac = re.findall(
            r'<a href="xfrac3d_([0-9]+\.[0-9]+)\.bin">', resp.text
        )

        return [float(x) for x in all_redshifts_xfrac]

    @classmethod
    def available_redshifts_dens(cls) -> list[float]:
        """
        Get all available redshifts for dens files.

        Returns
        -------
        list[float]
            List of all available redshifts for dens files.
        """
        resp = requests.get(Signal21cm.DENS_URL.format(dens_name=""), timeout=30)
        all_redshifts_dens = re.findall(
            r'<a href="([0-9]+\.[0-9]+)n_all.dat">', resp.text
        )
        return [float(x) for x in all_redshifts_dens]

    @classmethod
    def available_redshifts(cls) -> list[float]:
        """Get all available redshifts.

        Returns
        -------
        list[float]
            List of all available redshifts.
        """
        all_redshifts = set(cls.available_redshifts_dens()).intersection(
            set(cls.available_redshifts_xfrac())
        )

        return list(sorted(all_redshifts))
