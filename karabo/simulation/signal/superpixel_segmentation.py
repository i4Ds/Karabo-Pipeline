"""Segmentation with Superpixel."""

import tools21cm as t2c

from karabo.simulation.signal.base_segmentation import BaseSegmentation
from karabo.simulation.signal.typing import Image3D, SegmentationOutput


# pylint: disable=too-few-public-methods
class SuperpixelSegmentation(BaseSegmentation):
    """
    Superpixel based segmentation.

    Examples
    --------
    >>> from karabo.simulation.signal.plotting import SegmentationPlotting
    >>> from karabo.simulation.signal.signal_21_cm import Signal21cm
    >>> z1 = Signal21cm.get_xfrac_dens_file(z=7.059, box_dims=244 / 0.7)
    >>> sig = Signal21cm([z1])
    >>> seg = SuperpixelSegmentation(max_baseline=70.0,  max_iter=5, n_segments=1000)
    >>> signal_images = sig.simulate()
    >>> segmented = seg.segment(signal_images[0])
    >>> SegmentationPlotting.superpixel_plotting(segmented, signal_images[0])
    """

    def __init__(
        self, max_baseline: float = 70.0, n_segments: int = 1000, max_iter: int = 5
    ) -> None:
        """
        Superpixel based segmentation.

        Parameters
        ----------
        n_segments : int, optional
            Number of segments for the t2c.slice_cube function, by default 1000
        max_iter : int, optional
            Max number of iterations of the t2c.slice_cube function, by default 5
        """
        self.max_baseline = max_baseline
        self.n_segments = n_segments
        self.max_iter = max_iter

    def segment(self, image: Image3D) -> SegmentationOutput:
        """
        Superpixel based segmentation.

        Parameters
        ----------
        image : Image3D
            The constructed simulation

        Returns
        -------
        SegmentationOutput
            Superpixel cube
        """
        dt2 = image.data
        redshift = image.redshift
        box_dims = image.box_dims

        # Image is in Kelvin, we need mK
        dt2 *= 1000

        dt_smooth = t2c.smooth_coeval(
            cube=dt2,  # Data cube that is to be smoothed
            z=redshift,  # Redshift of the coeval cube
            box_size_mpc=box_dims,  # Box size in cMpc
            max_baseline=self.max_baseline,  # Maximum baseline of the telescope
            ratio=1,  # Ratio of smoothing scale in frequency direction
            nu_axis=2,
        )  # frequency axis

        labels = t2c.slic_cube(
            cube=dt_smooth,
            n_segments=self.n_segments,
            compactness=0.1,
            max_iter=self.max_iter,
            sigma=0,
            min_size_factor=0.5,
            max_size_factor=3,
            cmap=None,
        )

        superpixel_map = t2c.superpixel_map(dt_smooth, labels)

        xhii_stitch = t2c.stitch_superpixels(
            data=dt_smooth,
            labels=labels,
            bins="knuth",
            binary=True,
            on_superpixel_map=True,
        )

        mask_xhi = (
            t2c.smooth_coeval(
                cube=dt2,
                z=redshift,
                box_size_mpc=box_dims,
                max_baseline=self.max_baseline,
                nu_axis=2,
            )
            < 0.5
        )

        image_out = Image3D(
            data=superpixel_map,
            x_label=image.x_label,
            y_label=image.y_label,
            redshift=image.redshift,
            box_dims=image.box_dims,
            z_label=image.z_label,
        )

        return SegmentationOutput(
            image=image_out,
            xhii_stitch=xhii_stitch,
            mask_xhi=mask_xhi,
            dt_smooth=dt_smooth,
            xhi_seg_err=None,
        )
