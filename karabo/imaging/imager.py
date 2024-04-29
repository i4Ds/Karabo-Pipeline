from __future__ import annotations

import os
from typing import List, Optional, Union

import numpy as np
import oskar
from astropy.coordinates import SkyCoord
from rascil.processing_components import create_visibility_from_ms
from ska_sdp_datamodels.visibility import Visibility as RASCILVisibility
from typing_extensions import assert_never

from karabo.imaging.image import Image
from karabo.simulation.visibility import Visibility
from karabo.simulator_backend import SimulatorBackend
from karabo.util._types import FilePathType
from karabo.util.file_handler import FileHandler


class Imager:
    """Imaging functionality using the visibilities
    of an observation.

    In addition, it provides the calculation of the pixel coordinates of point sources.

    Parameters
    ---------------------------------------------
    visibility : Visibility | RASCILVisibility, required
        Visibility object containing the visibilities of an observation.
    logfile : str, default=None,
        Name of logfile (default is to construct one from msname)
    performance_file : str, default=None
        Name of json file to contain performance information
    ingest_dd : List[int], default=[0],
        Data descriptors in MS to read (all must have the same number of channels)
    ingest_vis_nchan : int, default=3,
        Number of channels in a single data descriptor in the MS
    ingest_chan_per_vis : int, default=1,
        Number of channels per blockvis (before any average)
    ingest_average_blockvis : Union[bool, str], default=False,
        Average all channels in blockvis.
    imaging_phasecentre : str, default=None
        Phase centre (in SkyCoord string format)
    imaging_pol : str, default="stokesI"
        RASCIL polarisation frame for image
    imaging_nchan : int, default=1,
        Number of channels per image
    imaging_ng_threads : int, default=4,
        Number of Nifty Gridder threads to use (4 is a good choice)
    imaging_w_stacking : Union[bool, str], default=True
        Use the improved w stacking method in Nifty Gridder?
    imaging_flat_sky : Union[bool, str], default=False
        If using a primary beam, normalise to flat sky?
    imaging_npixel : int, default=None
        Number of pixels in ra, dec: Should be a composite of 2, 3, 5
    imaging_cellsize : float, default=None
        Cellsize (radians). Default is to calculate
    override_cellsize : bool, default=False
        Override the cellsize if it is above the critical cellsize
    imaging_weighting : str, default="uniform"
        Type of weighting uniform or robust or natural
    imaging_robustness : float, default=0.0
        Robustness for robust weighting
    imaging_gaussian_taper : float, default=None
        Size of Gaussian smoothing, implemented as taper in weights (rad)
    imaging_dopsf : Union[bool, str], default=False
        Make the PSF instead of the dirty image?
    imaging_dft_kernel : str, default=None
        DFT kernel: cpu_looped | cpu_numba | gpu_raw
    """

    def __init__(
        self,
        visibility: Union[Visibility, RASCILVisibility],
        logfile: Optional[str] = None,
        performance_file: Optional[str] = None,
        ingest_dd: List[int] = [0],
        ingest_vis_nchan: Optional[int] = None,
        ingest_chan_per_vis: int = 1,
        ingest_average_blockvis: Union[bool, str] = False,
        imaging_phasecentre: Optional[str] = None,
        imaging_pol: str = "stokesI",
        imaging_nchan: int = 1,
        imaging_context: str = "ng",
        imaging_ng_threads: int = 4,
        imaging_w_stacking: Union[bool, str] = True,
        imaging_flat_sky: Union[bool, str] = False,
        imaging_npixel: Optional[int] = None,
        imaging_cellsize: Optional[float] = None,
        override_cellsize: bool = False,
        imaging_weighting: str = "uniform",
        imaging_robustness: float = 0.0,
        imaging_gaussian_taper: Optional[float] = None,
        imaging_dopsf: Union[bool, str] = False,
        imaging_uvmax: Optional[float] = None,
        imaging_uvmin: float = 0,
        imaging_dft_kernel: Optional[
            str
        ] = None,  # DFT kernel: cpu_looped | cpu_numba | gpu_raw
    ) -> None:
        self.visibility = visibility
        self.logfile = logfile
        self.performance_file = performance_file
        self.ingest_dd = ingest_dd
        self.ingest_vis_nchan = ingest_vis_nchan
        self.ingest_chan_per_vis = ingest_chan_per_vis
        self.ingest_average_blockvis = ingest_average_blockvis
        self.imaging_phasecentre = imaging_phasecentre
        self.imaging_pol = imaging_pol
        self.imaging_nchan = imaging_nchan
        self.imaging_context = imaging_context
        self.imaging_ng_threads = imaging_ng_threads
        self.imaging_w_stacking = imaging_w_stacking
        self.imaging_flat_sky = imaging_flat_sky
        self.imaging_npixel = imaging_npixel
        self.imaging_cellsize = imaging_cellsize
        self.override_cellsize = override_cellsize
        self.imaging_weighting = imaging_weighting
        self.imaging_robustness = imaging_robustness
        self.imaging_gaussian_taper = imaging_gaussian_taper
        self.imaging_dopsf = imaging_dopsf
        self.imaging_dft_kernel = imaging_dft_kernel
        self.imaging_uvmax = imaging_uvmax
        self.imaging_uvmin = imaging_uvmin

    def _oskar_imager(
        self,
        fits_path: FilePathType,
        visibility: Visibility,
        combine_across_frequencies: bool = True,
    ) -> Image:
        if combine_across_frequencies is False:
            raise NotImplementedError(
                """For the OSKAR backend, the dirty image will
                always have intensities added across all frequency channels.
                Therefore, combine_across_frequencies==False
                is not currently supported.
                """
            )
        imager = oskar.Imager()
        imager.set(
            input_file=visibility.vis_path,
            output_root=fits_path,
            cellsize_arcsec=3600 * np.degrees(self.imaging_cellsize),
            image_size=self.imaging_npixel,
        )
        if self.imaging_phasecentre is not None:
            phase_centre = SkyCoord(self.imaging_phasecentre, frame="icrs")
            ra = phase_centre.ra.degree
            dec = phase_centre.dec.degree

            imager.set_vis_phase_centre(ra, dec)

        imager.run(return_images=1)

        # OSKAR adds _I.fits to the fits_path set by the user
        image = Image(path=f"{fits_path}_I.fits")

        # OSKAR Imager always produces one image by
        # combining all frequency channels.
        # To maintain the same data shape when compared to other imagers (e.g. RASCIL),
        # We add an axis for frequency, and modify the header accordingly
        assert len(image.data.shape) == 4

        image.header["NAXIS"] = 4
        image.header["NAXIS4"] = 1

        image.write_to_file(path=f"{fits_path}_I.fits", overwrite=True)

        return image

    def get_dirty_image(
        self,
        fits_path: Optional[FilePathType] = None,
        combine_across_frequencies: bool = True,
        imaging_backend: Optional[SimulatorBackend] = None,
    ) -> Image:
        """Get Dirty Image of visibilities passed to the Imager.

        Note: If `fits_path` is provided and already exists, then this function will
        overwrite `fits_path`.

        Args:
            fits_path: Path to where the .fits file will get saved.
            combine_across_frequencies: if True, will return an object with
                one entry in the frequency axis. This is created by adding pixel values
                across all frequency channels. If False, will return an object with
                one or more entries in the frequency axis.
                NOTE: for OSKAR, the default behavior corresponds to a True value.
                A False value is currently not supported for OSKAR.
                For RASCIL, the default behavior corresponds to a False value.
                A True value will trigger an additional step to add images across
                frequency channels.
            imaging_backend: Backend to use for computing dirty image from visibilities.
                Defaults to None, which leads to the Imager selecting the same backend
                    used to compute the visibilities.

        Returns:
            Dirty Image, with 4D data of shape
                (frequency, polarisation, pixel_x, pixel_y)
        """
        # Validate requested filepath
        if fits_path is None:
            tmp_dir = FileHandler().get_tmp_dir(
                prefix="Imager-Dirty-",
                purpose="disk-cache for dirty.fits",
            )
            fits_path = os.path.join(tmp_dir, "dirty.fits")

        # If imaging_backend is None, use same backend applied
        # when computing visibilities
        if imaging_backend is None:
            if isinstance(self.visibility, Visibility):
                imaging_backend = SimulatorBackend.OSKAR
            elif isinstance(self.visibility, RASCILVisibility):
                imaging_backend = SimulatorBackend.RASCIL
            else:
                assert_never(self.visibility)

        # Perform imaging based on selected backend
        if imaging_backend is SimulatorBackend.OSKAR:
            if isinstance(self.visibility, RASCILVisibility):
                raise NotImplementedError(
                    """OSKAR Imager applied to
                    RASCIL Visibilities is currently not supported.
                    For RASCIL Visibilities please use the RASCIL Imager."""
                )

            image = self._oskar_imager(
                fits_path,
                self.visibility,
                combine_across_frequencies=combine_across_frequencies,
            )

            return image
        elif imaging_backend is SimulatorBackend.RASCIL:
            vis = self.visibility
            if isinstance(vis, Visibility):
                # Convert OSKAR Visibility to RASCIL-compatible format
                block_visibilities = create_visibility_from_ms(str(vis.ms_file_path))

                if len(block_visibilities) != 1:
                    raise EnvironmentError("Visibilities are too large")
                vis = block_visibilities[0]

            # Compute dirty image from visibilities
            image = self._rascil_imager(
                fits_path, vis, combine_across_frequencies=combine_across_frequencies
            )

            return image

        assert_never(imaging_backend)
