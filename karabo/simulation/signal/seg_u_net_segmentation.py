"""Segmentation with SegU-net."""

import pkg_resources
import tools21cm as t2c

try:
    from tensorflow.keras.models import load_model
except ImportError:  # noqa: E722
    from tensorflow.python.keras.models import load_model

from karabo.simulation.signal.base_segmentation import BaseSegmentation
from karabo.simulation.signal.typing import Image3D, SegmentationOutput


# pylint: disable=too-few-public-methods
class FixedSegUNet(t2c.segmentation.segunet21cm):
    """
    Fixes the bug such that the Segmentation can run.

    Can be removed once Tools21cm is fixed.
    """

    def __init__(self, tta=1, verbose=False):
        """SegU-Net: segmentation of 21cm images with U-shape network.

           - tta (int): default 0 (super-fast, no pixel-error map) implement the error
             map with time-test aumentated technique in the prediction process
           - verbose (bool): default False, activate verbosity

        Description:
           tta = 0 : fast (~7 sec), it tends to be a few percent less accurate (<2%)
           then the other two cases, no pixel-error map (no TTA manipulation) tta = 1 :
           medium (~17 sec), accurate and preferable than tta=0, with pixel-error map (3
           samples) tta = 2 : slow (~10 min), accurate, with pixel-error map (~100
           samples)

        Returns:
           - X_seg (ndarray) : recovered binary field (1 = neutral and 0 = ionized
             regions)
           - X_err (ndarray) : pixel-error map of the recovered binary field

        Example:
         $ from tools21cm import segmentation $ seg = segmentation.segunet21cm(tta=1,
         verbose=True)   # load model (need to be done once) $ Xseg, Xseg_err =
         seg.prediction(x=dT3)

        Print of the Network's Configuration file:
         [TRAINING] BATCH_SIZE = 64 AUGMENT = NOISESMT IMG_SHAPE = 128, 128 CHAN_SIZE =
         256 DROPOUT = 0.05 KERNEL_SIZE = 3 EPOCHS = 100 LOSS = balanced_cross_entropy
         METRICS = iou, dice_coef, binary_accuracy, binary_crossentropy LR = 1e-3 RECOMP
         = False GPUS = 2 PATH =
         /home/michele/Documents/PhD_Sussex/output/ML/dataset/inputs/data2D_128_030920/

         [RESUME] RESUME_PATH = ./output/ML/dataset/outputs/new/02-10T23-52-36_128slice/
         BEST_EPOCH = 56 RESUME_EPOCH = 66
        """
        # pylint: disable=invalid-name
        self.TTA = tta
        self.VERBOSE = verbose

        if self.TTA == 2:
            # slow
            self.MANIP = self.IndependentOperations(verbose=self.VERBOSE)
        elif self.TTA == 1:
            # fast
            self.MANIP = {"opt0": [lambda a: a, 0, 0]}
        elif self.TTA == 0:
            # super-fast
            self.MANIP = {"opt0": [lambda a: a, 0, 0]}

        self.NR_MANIP = len(self.MANIP)

        # load model
        MODEL_NAME = pkg_resources.resource_filename(
            "tools21cm", "input_data/segunet_02-10T23-52-36_128slice_ep56.h5"
        )
        METRICS = {
            "balanced_cross_entropy": t2c.segmentation.balanced_cross_entropy,
            "iou": t2c.segmentation.iou,
            "dice_coef": t2c.segmentation.dice_coef,
        }
        self.MODEL_LOADED = load_model(MODEL_NAME, custom_objects=METRICS)
        print(f" Loaded model: {MODEL_NAME}")
        # pylint: enable=invalid-name


# pylint: disable=too-few-public-methods
class SegUNetSegmentation(BaseSegmentation):
    """
    SegU-net based segmentation. Using its own init for t2c (Tools21cm).

    Examples
    --------
    >>> from karabo.simulation.signal.plotting import SegmentationPlotting
    >>> from karabo.simulation.signal.signal_21_cm import Signal21cm
    >>> z = Signal21cm.get_xfrac_dens_file(z=7.059, box_dims=244 / 0.7)
    >>> sig = Signal21cm([z])
    >>> signal_images = sig.simulate()
    >>> seg = SegUNetSegmentation(max_baseline=70.0, tta=2)
    >>> segmented = seg.segment(signal_images[0])
    >>> SegmentationPlotting.seg_u_net_plotting(segmented=segmented)
    """

    def __init__(self, max_baseline: float = 70.0, tta: int = 2) -> None:
        """
        SegU.net based segmentation.

        Parameters
        ----------
        max_baselines : float, optional
            Max number of baselines , by default 70.0
        tta : int, optional
            0=super-fast, 1=fast, 2=super-slow, by default 2
        """
        self.max_baseline = max_baseline
        self.tta = tta

    def segment(self, image: Image3D) -> SegmentationOutput:
        """
        SegU-net based segmentation. Using its own init for t2c (Tools21cm).

        Parameters
        ----------
        image : Image3D
            The constructed simulation

        Returns
        -------
        SegmentationOutput
            SegU-net cube
        """
        dt2 = image.data
        redshift = image.redshift
        boxsize = image.box_dims

        # Image is in Kelvin, we need mK
        dt2 *= 1000

        dt_smooth = t2c.smooth_coeval(
            cube=dt2,  # Data cube that is to be smoothed
            z=redshift,  # Redshift of the coeval cube
            box_size_mpc=boxsize,  # Box size in cMpc
            max_baseline=self.max_baseline,  # Maximum baseline of the telescope
            ratio=1.0,  # Ratio of smoothing scale in frequency direction
            nu_axis=2,
        )  # frequency axis

        mask_xhi = (
            t2c.smooth_coeval(
                cube=dt2,
                z=redshift,
                box_size_mpc=boxsize,
                max_baseline=self.max_baseline,
                nu_axis=2,
            )
            < 0.5
        )

        segment = FixedSegUNet(tta=self.tta, verbose=True)
        cut = dt_smooth[:128, :128, :128]
        xhi_seg, xhi_seg_err = segment.prediction(x=cut)

        image_out = Image3D(
            data=xhi_seg,
            x_label=image.x_label,
            y_label=image.y_label,
            redshift=image.redshift,
            box_dims=image.box_dims,
            z_label=image.z_label,
        )

        return SegmentationOutput(
            image=image_out,
            xhii_stitch=None,
            mask_xhi=mask_xhi,
            dt_smooth=dt_smooth,
            xhi_seg_err=xhi_seg_err,
        )
