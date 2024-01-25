"""Signal plotting helpers."""
from typing import Annotated, Any, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tools21cm as t2c
from matplotlib import colors
from matplotlib.figure import Figure
from sklearn.metrics import matthews_corrcoef

from karabo.error import KaraboError
from karabo.simulation.signal.typing import (
    BaseImage,
    Image2D,
    Image3D,
    SegmentationOutput,
    XFracDensFilePair,
)


class SignalPlotting:
    """Signal plotting helpers."""

    @classmethod
    def xfrac_dens(cls, data: XFracDensFilePair) -> Figure:
        """
        Plot the xfrac and dens files.

        Parameters
        ----------
        data : XFracDensFilePair
            The xfrac and dens file pair.

        Returns
        -------
        Figure
            The figure that was plotted.
        """
        loaded = data.load()
        x, y = loaded.xy_dims()

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        fig.suptitle(f"$z={loaded.z},~x_v=${loaded.x_frac.mean():.2f}", size=18)
        axs[0].set_title("Density contrast slice")
        pcm_dens = axs[0].pcolormesh(x, y, loaded.dens[0] / loaded.dens.mean() - 1)
        fig.colorbar(pcm_dens, ax=axs[0], label="[K]")
        axs[0].set_xlabel(r"$x$ [Mpc]")
        axs[0].set_ylabel(r"$y$ [Mpc]")

        axs[1].set_title("Ionisation fraction slice")
        pcm_ion = axs[1].pcolormesh(x, y, loaded.x_frac[0])
        fig.colorbar(pcm_ion, ax=axs[1], label="[K]")
        axs[1].set_xlabel(r"$x$ [Mpc]")
        axs[1].set_ylabel(r"$y$ [Mpc]")

        return fig

    @classmethod
    def brightness_temperature(cls, data: BaseImage, z_layer: int = 0) -> Figure:
        """
        Plot the brightness temperature of a 2D image.

        Parameters
        ----------
        data : BaseImage
            The image to be plotted.

        z_layer : int, optional
            The Z layer to be used, when a Image3D is used.

        Returns
        -------
        Figure
            Figure of the plotted image
        """
        image_data = data.data
        x_label = data.x_label
        y_label = data.y_label

        if isinstance(data, Image3D):
            image_data = image_data[z_layer, :, :]

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.set_title(f"21 cm signal, z={data.redshift}")
        colour_bar = ax.pcolormesh(x_label, y_label, image_data)
        ax.set_xlabel(r"$x$ [Mpc]")
        ax.set_ylabel(r"$y$ [Mpc]")
        fig.colorbar(colour_bar, ax=ax, label="K")

        return fig

    @classmethod
    def power_spectrum_xfrac_dens(
        cls, data: Union[XFracDensFilePair, list[XFracDensFilePair]], kbins: int = 15
    ) -> Figure:
        """
        Plot the power spectrum the 21cm signal using xfrac and dens-files.

        Parameters
        ----------
        data : Union[XFracDensFilePair, list[XFracDensFilePair]]
            The xfrac and dens file pair or a list thereof.
        kbins : int, optional
            Count of bins for the spectrum plot, by default 15

        Returns
        -------
        Figure
            The generated plot figure.
        """
        if not isinstance(data, list):
            data = [data]

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_title("Spherically averaged power spectrum.")
        ax.set_xlabel(r"k (Mpc$^{-1}$)")
        ax.set_ylabel(r"P(k) k$^{3}$/$(2\pi^2)$")

        for elem in data:
            loaded = elem.load()

            d_t = t2c.calc_dt(loaded.x_frac, loaded.dens, loaded.z)
            d_t_subtracted = t2c.subtract_mean_signal(d_t, 0)
            ps_1d = t2c.power_spectrum_1d(
                d_t_subtracted,
                kbins=kbins,
                box_dims=loaded.box_dims,
            )

            ps = ps_1d[0]
            ks = ps_1d[1]
            ax.loglog(ks, ps * ks**3 / 2 / np.pi**2, label=loaded.z)

        ax.grid()
        ax.legend()
        return fig

    @classmethod
    def power_spectrum(
        cls,
        data: Union[
            BaseImage, SegmentationOutput, list[Union[BaseImage, SegmentationOutput]]
        ],
        kbins: int = 15,
        z_layer: int = 0,
    ) -> Figure:
        """
        Plot the power spectrum the 21cm signal.

        Parameters
        ----------
        data : Union[BaseImage, SegmentationOutput]
            Either a BaseImage or a SegmentationOutput
        kbins : int, optional
            Count of bins for the spectrum plot, by default 15
        z_layer : int
            If an Image3D is passed, then this layer will be used to plot the image. By
            default 0

        Returns
        -------
        Figure
            The generated plot figure.
        """
        if not isinstance(data, list):
            data = [data]

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_xlabel(r"k (Mpc$^{-1}$)")
        ax.set_ylabel(r"P(k) k$^{3}$/$(2\pi^2)$")
        ax.set_title("Spherically averaged power spectrum.")

        for elem in data:
            image: BaseImage
            if isinstance(elem, SegmentationOutput):
                image = elem.image
            else:
                image = elem

            d_t_subtracted: npt.NDArray[np.float_] = image.data
            if isinstance(image, Image3D):
                d_t_subtracted = d_t_subtracted[z_layer, :, :]

            ps_1d = t2c.power_spectrum_1d(
                d_t_subtracted,
                kbins=kbins,
                box_dims=image.box_dims,
            )

            ps = ps_1d[0]
            ks = ps_1d[1]
            ax.loglog(ks, ps * ks**3 / 2 / np.pi**2, label=image.redshift)

        ax.grid()
        ax.legend()
        return fig

    # pylint: disable=too-many-locals
    @classmethod
    def power_spectrum_image_vs_xfrac_dens(
        cls,
        image: Union[BaseImage, SegmentationOutput],
        xfrac_dens: XFracDensFilePair,
        kbins: int = 15,
        z_layer: int = 0,
    ) -> Figure:
        """
        Plot the power spectrum by using an image and the original xfrac/dens file pair.

        Parameters
        ----------
        image : Union[BaseImage, SegmentationOutput]
            Either a BaseImage or a SegmentationOutput
        xfrac_dens : XFracDensFilePair
            The xfrac and dens file pair.
        kbins : int, optional
            Count of bins for the spectrum plot, by default 15
        z_layer : int
            If an Image3D is passed, then this layer will be used to plot the image. By
            default 0

        Returns
        -------
        Figure
            The generated plot figure.
        """
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_xlabel(r"k (Mpc$^{-1}$)")
        ax.set_ylabel(r"P(k) k$^{3}$/$(2\pi^2)$")
        ax.set_title("Spherically averaged power spectrum.")

        # Load the correct image
        img: BaseImage
        if isinstance(image, SegmentationOutput):
            img = image.image
        else:
            img = image

        d_t_subtracted: npt.NDArray[np.float_] = img.data
        if isinstance(image, Image3D):
            d_t_subtracted = d_t_subtracted[z_layer, :, :]

        ps_1d = t2c.power_spectrum_1d(
            d_t_subtracted,
            kbins=kbins,
            box_dims=img.box_dims,
        )

        ps = ps_1d[0]
        ks = ps_1d[1]
        ax.loglog(ks, ps * ks**3 / 2 / np.pi**2, label="Image")

        # Load the xfrac/dens
        loaded = xfrac_dens.load()

        d_t_xd = t2c.calc_dt(loaded.x_frac, loaded.dens, loaded.z)
        d_t_subtracted_xd = t2c.subtract_mean_signal(d_t_xd, 0)
        ps_1d_xd = t2c.power_spectrum_1d(
            d_t_subtracted_xd,
            kbins=kbins,
            box_dims=loaded.box_dims,
        )

        ps = ps_1d_xd[0]
        ks = ps_1d_xd[1]
        ax.loglog(ks, ps * ks**3 / 2 / np.pi**2, label="Xfrac/dens")

        ax.grid()
        ax.legend()
        return fig

    # pylint: disable=too-many-arguments,too-many-locals
    @classmethod
    def general_img(
        cls,
        img: Image2D,
        title: str,
        tick_count: int = 5,
        x_label: str = "RA [°]",
        y_label: str = "DEC [°]",
        bar_label: str = "Temperature [K]",
        log_bar: bool = False,
    ) -> Figure:
        """
        Plot a general image with a temperature.

        Parameters
        ----------
        img : Image2D
            The image to be plotted.
        title : str
            Title to be shown in the figure.
        tick_count : int, optional
            The count of ticks to show anlong each axis, by default 5
        x_label : str, optional
            Label to be plotted along the X-axis.
        y_label : str, optional
            Label to be plotted along the Y-axis.
        bar_label: str, optional
            Label for the colour bar.
        log_bar : bool, optional
            If the colour bar should have a symmetric log norm applied.

        Returns
        -------
        Figure
            The resulting plot figure.
        """
        fig, ax = plt.subplots(1, 1)

        data = img.data
        add_kwargs: dict[str, Any] = {}
        if log_bar:
            add_kwargs["norm"] = colors.SymLogNorm(linthresh=0.01)

        im = ax.imshow(data, origin="lower", **add_kwargs)
        plt.colorbar(im, ax=ax, label=bar_label)

        x_pos = np.linspace(0, img.data.shape[0] - 1, tick_count)
        x_labels = np.around(
            np.linspace(img.x_label[0], img.x_label[-1], tick_count), 2
        )

        y_pos = np.linspace(0, img.data.shape[1] - 1, tick_count)
        y_labels = np.around(
            np.linspace(img.y_label[0], img.y_label[-1], tick_count), 2
        )

        ax.xaxis.set_ticks(x_pos, labels=x_labels)
        ax.yaxis.set_ticks(y_pos, labels=y_labels)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        fig.suptitle(title)

        return fig

    @classmethod
    def general_polar_plot(
        cls,
        ra_series: Annotated[npt.NDArray[np.float_], Literal["N"]],
        dec_series: Annotated[npt.NDArray[np.float_], Literal["N"]],
        intensities_series: Annotated[npt.NDArray[np.float_], Literal["N"]],
        title: str,
        bar_label: str = "Temperature [K]",
        log_bar: bool = False,
    ) -> Figure:
        """
        Plot a RA/DEC data in a polar plot with the intensity representing the colour.

        Parameters
        ----------
        ra_series : Annotated[npt.NDArray[np.float_], Literal["N"]]
            RA coordinates in degrees.
        dec_series : Annotated[npt.NDArray[np.float_], Literal["N"]]
            DEC coordinates in degrees.
        intensities_series : Annotated[npt.NDArray[np.float_], Literal["N"]]
            Intensities in Kelvin.
        title : str
            Title to be shown in the figure.
        bar_label: str, optional
            Label for the colour bar.
        log_bar : bool, optional
            If the colour bar should have a symmetric log norm applied.

        Returns
        -------
        Figure
            The resulting plot figure.
        """
        add_kwargs: dict[str, Any] = {}
        if log_bar:
            add_kwargs["norm"] = colors.SymLogNorm(linthresh=0.01)

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 10))
        sc = ax.scatter(
            ra_series / 180 * np.pi,
            dec_series,
            c=intensities_series,
            **add_kwargs,
        )
        plt.colorbar(sc, ax=ax, label=bar_label)
        ax.grid(True)

        ax.set_title(title, fontdict={"fontsize": 15})

        return fig


class SegmentationPlotting:
    """Plotting utilities for the segmentation."""

    # pylint: disable=too-many-locals
    @classmethod
    def seg_u_net_plotting(
        cls,
        segmented: SegmentationOutput,
        bin_count: int = 20,
    ) -> Figure:
        """
        Plot the first slice of the segU-net cube.

        Parameters
        ----------
        segmented : SegmentationOutput
            Output of the segmentation
        bin_count : int
            The number output bins in the histogram plot

        Returns
        -------
        Figure
            Plot of the SegU-Net segmentation

        Raises
        ------
        KaraboError
            If the input 'xhi_seg_err' is None
        """
        xhi_seg = segmented.image.data
        boxsize = segmented.image.box_dims
        mask_xhi = segmented.mask_xhi
        redshift = segmented.image.redshift
        xhi_seg_err = segmented.xhi_seg_err
        if xhi_seg_err is None:
            raise KaraboError("xhi_seg_err should not be None.")
        mask_xhi2 = mask_xhi[:128, :128, :128]

        phicoef_seg = matthews_corrcoef(
            mask_xhi2.flatten(), xhi_seg[:128, :128, :128].flatten()
        )

        fig, (ax1, ax2, ax3) = plt.subplots(
            figsize=(20, 6),
            ncols=3,
        )

        fig.suptitle(f"SegU-Net segmentation with redshift {redshift}")

        ax1.set_title(rf"($r_{{\phi}}={phicoef_seg:.3f}$)")
        ax1.imshow(
            xhi_seg[0],
            origin="lower",
            cmap="jet",
            extent=[0, boxsize, 0, boxsize],
        )
        ax1.contour(
            mask_xhi2[0],
            colors="lime",
            extent=[0, boxsize, 0, boxsize],
        )
        ax1.set_xlabel("x [Mpc]")

        ax2.set_title("Pixel-Error")
        im2 = ax2.imshow(
            xhi_seg_err[0],
            origin="lower",
            cmap="jet",
            extent=[0, boxsize, 0, boxsize],
        )
        fig.colorbar(
            im2,
            label=r"$\sigma_{std}$",
            ax=ax2,
        )
        ax2.set_xlabel("x [Mpc]")

        ax3.set_title("Pixel-Error-Histogram")
        ax3.hist(xhi_seg_err[0].flatten(), bins=bin_count)
        ax3.set_ylabel("Count")
        ax3.set_yscale("log")
        ax3.set_xlabel(r"$\sigma_{std}$")

        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        fig.tight_layout()
        return fig

    # pylint: disable=too-many-locals
    @classmethod
    def superpixel_plotting(
        cls,
        segmented: SegmentationOutput,
        signal_image: Image3D,
        log_sky: bool = False,
    ) -> Figure:
        """
        Plot the first slice of the superpixel cube.

        Parameters
        ----------
        segmented : SegmentationOutput
            output of the segmentation
        signal_image : Image3D
            Image cube
        log_sky : bool, optional
            If the colour bar of the sky plot should have a symmetric log norm applied.

        Returns
        -------
        Figure
            Plot of the Superpixel segmentation

        Raises
        ------
        KaraboError
            If the input 'xhii_stitch' is None
        """
        dt2 = signal_image.data
        box_dims = signal_image.box_dims
        redshift = signal_image.redshift
        mask_xhi = segmented.mask_xhi
        xhii_stitch = segmented.xhii_stitch
        if xhii_stitch is None:
            raise KaraboError("xhii_stitch should not be None")
        superpixel_map = segmented.image.data
        dt_smooth = segmented.dt_smooth

        dx, dy = box_dims / dt2.shape[1], box_dims / dt2.shape[2]
        y, x = np.mgrid[slice(dy / 2, box_dims, dy), slice(dx / 2, box_dims, dx)]
        phicoef_sup = matthews_corrcoef(mask_xhi.flatten(), 1 - xhii_stitch.flatten())

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

        fig.suptitle(f"Superpixel segmentation with redshift={redshift}")

        kwargs = {}
        if log_sky:
            kwargs["norm"] = colors.SymLogNorm(linthresh=0.01)

        ax1.set_title("dt_smooth")
        im1 = ax1.pcolormesh(x, y, dt_smooth[0], cmap="jet", **kwargs)
        fig.colorbar(im1, ax=ax1, label="[K]")

        ax2.set_title("superpixel_map")
        im2 = ax2.pcolormesh(x, y, superpixel_map[0], cmap="jet")
        fig.colorbar(im2, ax=ax2)

        ax3.set_title(rf"$r_{{\phi}}={phicoef_sup:.3f}$")
        ax3.pcolormesh(x, y, 1 - xhii_stitch[0], cmap="jet")
        ax3.contour(mask_xhi[0], colors="lime", extent=[0, box_dims, 0, box_dims])

        return fig
