"""Synchroton signal catalogue wrapper."""
from pathlib import Path
from typing import Annotated, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.wcs import WCS

from karabo.data.external_data import DownloadObject
from karabo.simulation.signal import helpers
from karabo.simulation.signal.base_signal import BaseSignal
from karabo.simulation.signal.typing import Image2D


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class SignalSynchroton(BaseSignal[Image2D]):
    """
    Synchroton signal.

    Examples
    --------
    >>> from karabo.simulation.signal.plotting import SignalPlotting
    >>> from astropy import units
    >>> cent = SkyCoord(ra=10 * units.degree, dec=10 * units.degree, frame="icrs")
    >>> sync_sig = SignalSynchroton(
    ...     centre=cent,
    ...     fov=Angle([20, 20], unit=units.degree),
    ...     grid_size=(100, 100),
    ... )
    >>> imgs = sync_sig.simulate()
    >>> SignalPlotting.general_img(imgs[0], "Synchroton")
    >>> # Map overview plot
    >>> SignalPlotting.general_polar_plot(
    ...     sync_sig.data["RA"],
    ...     sync_sig.data["DEC"],
    ...     sync_sig.data["intensity"],
    ...     title="Synchroton map",
    ...     log_bar=True,
    ... )
    """

    DEFAULT_FITS = (
        "https://lambda.gsfc.nasa.gov/data/foregrounds/haslam/images/"
        "lambda_mollweide_haslam408_dsds.fits"
    )
    MAX_GRID_X = 190
    MAX_GRID_Y = 190

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        centre: SkyCoord,
        fov: Angle,
        grid_size: tuple[Annotated[int, Literal["X"]], Annotated[int, Literal["Y"]]],
        diffuse_emission_path: Optional[Path] = None,
        data_table_id: int = 1,
        mask_table_id: int = 2,
    ) -> None:
        """
        ...

        Parameters
        ----------
        centre : SkyCoord
            Center point. lon = right ascention, lat = declination
        fov : Angle
            Field of view for the right ascention in degrees. Must be between 0 < fov <=
            180. lon = right ascention fov, lat = declination fov
        grid_size : tuple[Annotated[int, Literal["X"]], Annotated[int, Literal["Y"]]]
            Size of the simulated output image (X, Y).
        diffuse_emission_path : Optional[Path], optional
            Path to the diffuse emission .fits file in the mollweide projection. If
            none, the `lambda_mollweide_haslam408_dsds.fits` will be downloaded, cached
            and used as a data source. By default None.
        data_table_id : int, optional
            The table ID of the emisson data from the `diffuse_emission_path`s fits
            file. By default 1.
        mask_table_id : int, optional
            The table ID of the masking data. By default 2.
        """
        if diffuse_emission_path is None:
            diffuse_emission_path = Path(
                DownloadObject(
                    "lambda_mollweide_haslam408_dsds.fits",
                    SignalSynchroton.DEFAULT_FITS,
                ).get()
            )

        self.diffuse_emission_path = diffuse_emission_path
        self.data_table_id = data_table_id
        self.mask_table_id = mask_table_id

        self.centre = centre
        self.fov = fov
        self.grid_size = grid_size
        self.loaded_data = False

        self.wcs: WCS
        self.data: pd.DataFrame
        self.hdu: fits.HDUList

    def _load_data(self) -> None:
        if self.loaded_data:
            return

        self.loaded_data = True
        loaded_fits = fits.open(self.diffuse_emission_path)
        self.hdu = loaded_fits[self.data_table_id]
        mask_data = loaded_fits[self.mask_table_id].data

        # Correct byte order
        emission_data = QTable(self.hdu.data).to_pandas().to_numpy()
        self.wcs = WCS(self.hdu.header)
        indexes = np.where(mask_data > 0)
        world = self.wcs.pixel_to_world(*indexes)

        world_radec = world.transform_to("icrs").to_table().to_pandas().to_numpy()

        emission_data_masked = emission_data[indexes]
        world_non_nan = ~np.isnan(world_radec[:, 0])

        self.data = pd.DataFrame(
            {
                "RA": world_radec[:, 0][world_non_nan],
                "DEC": world_radec[:, 1][world_non_nan],
                "intensity": emission_data_masked[world_non_nan],
            }
        )

    def simulate(self) -> list[Image2D]:
        """
        Fetch the synchroton signal.

        Returns
        -------
        list[Image2D]
            The synchroton image.
        """
        self._load_data()
        pos_df, bottom_left, top_right = helpers.filter_dataframe_radec(
            df=self.data,
            centre=self.centre,
            fov=self.fov,
            ra_column="RA",
            dec_column="DEC",
        )
        grid_size = (
            min(SignalSynchroton.MAX_GRID_X, self.grid_size[0]),
            min(SignalSynchroton.MAX_GRID_Y, self.grid_size[1]),
        )
        grid_intensity = helpers.map_radec_datapoints_to_grid(
            pos_df,
            grid_size=grid_size,
            ra_column="RA",
            dec_column="DEC",
            intensity_column="intensity",
        )

        if grid_size != self.grid_size:
            grid_intensity = helpers.interpolate_image(grid_intensity, self.grid_size)

        x_label = np.linspace(
            bottom_left.degree[0], top_right.degree[0], num=self.grid_size[0]
        )
        y_label = np.linspace(
            bottom_left.degree[1], top_right.degree[1], num=self.grid_size[1]
        )

        grid_intensity = self._tb_conv(
            freq=grid_intensity,
            source_freq=408,
            target_freq=100,
            alpha=2.55,
        )

        return [
            Image2D(
                data=grid_intensity,
                x_label=x_label,
                y_label=y_label,
                redshift=-1,
                box_dims=-1,
            )
        ]

    @classmethod
    def _tb_conv(
        cls,
        freq: npt.NDArray[np.float_],
        source_freq: float,
        target_freq: float,
        alpha: float,
    ) -> npt.NDArray[np.float_]:
        """
        Brightness Temperature Conversion.

        https://science.nrao.edu/facilities/vla/proposing/TBconv

        Parameters
        ----------
        freq : npt.NDArray[np.float_]
            The brightness to be converted. The array may be in any shape.
        source_freq : float
            Source frequency of the brightness
        target_freq : float
            Target frequency.
        alpha : float
            Alpha

        Returns
        -------
        npt.NDArray[np.float_]
            Converted array in the same shape as the input array.
        """
        return freq * (target_freq / source_freq) ** alpha
