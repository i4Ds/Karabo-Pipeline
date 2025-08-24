from __future__ import annotations

import glob
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Union

from typing_extensions import override

from karabo.imaging.image import Image
from karabo.imaging.imager_base import (
    DirtyImager,
    DirtyImagerConfig,
    ImageCleaner,
    ImageCleanerConfig,
)
from karabo.simulation.visibility import Visibility
from karabo.util._types import FilePathType
from karabo.util.file_handler import FileHandler

_WSCLEAN_BINARY = "wsclean"


def _get_command_prefix(tmp_dir: str) -> str:
    return (
        # wsclean always uses the current directory as the working directory
        f"cd {tmp_dir} && "
        # Avoids the following wsclean error:
        # This software was linked to a multi-threaded version of OpenBLAS.
        # OpenBLAS multi-threading interferes with other multi-threaded parts of
        # the code, which has a severe impact on performance. Please disable
        # OpenBLAS multi-threading by setting the environment variable
        # OPENBLAS_NUM_THREADS to 1.
        "OPENBLAS_NUM_THREADS=1 "
    )


@dataclass
class WscleanDirtyImagerConfig(DirtyImagerConfig):
    """Config / parameters of an WSCleanDirtyImager.

    Adds parameters specific to WSCleanDirtyImager.

    Attributes:
        imaging_npixel (int): see DirtyImagerConfig
        imaging_cellsize (float): see DirtyImagerConfig
        combine_across_frequencies (bool): see DirtyImagerConfig
        intervals_out (Optional[int]): split the measurement set in the given number
            of intervals, and image each interval separately

    """

    intervals_out: Optional[int] = None


class WscleanDirtyImager(DirtyImager):
    """Dirty imager based on the WSClean library.

    WSClean is integrated by calling the wsclean command line tool.
    The parameters in the config (DirtyImagerConfig) attribute are passed to wsclean.
    Use the create_image_custom_command function if you need to set params
    not available in DirtyImagerConfig.

    Attributes:
        config (DirtyImagerConfig): Config containing parameters for
            dirty imaging

    """

    TMP_PREFIX_DIRTY = "WSClean-dirty-"
    TMP_PURPOSE_DIRTY = "Disk cache for WSClean dirty images"

    OUTPUT_FITS_DIRTY = "wsclean-dirty.fits"

    def __init__(self, config: WscleanDirtyImagerConfig) -> None:
        """Initializes the instance with a config.

        Args:
            config (DirtyImagerConfig): see config attribute

        """
        super().__init__()
        self.config: WscleanDirtyImagerConfig = config

    @override
    def create_dirty_image(
        self,
        visibility: Visibility,
        /,
        *,
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        if visibility.format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility.format} is not supported, "
                "currently only MS is supported for WSClean imaging"
            )
        # TODO combine_across_frequencies
        # -channels-out <count>?
        if self.config.combine_across_frequencies is False:
            raise NotImplementedError(
                "combine_across_frequencies=False is currently not supported "
                "for the WSClean imager."
            )

        if self.config.intervals_out:
            raise NotImplementedError(
                "The parameter '-intervals-out' cannot be used with this function. "
                "If you want to split the visibilities into single time frames use "
                "the function 'create_dirty_image_series' instead."
            )

        tmp_dir = FileHandler().get_tmp_dir(
            prefix=self.TMP_PREFIX_DIRTY,
            purpose=self.TMP_PURPOSE_DIRTY,
        )

        command = _get_command_prefix(tmp_dir) + (
            f"{_WSCLEAN_BINARY} "
            + f"-size {self.config.imaging_npixel} {self.config.imaging_npixel} "
            + f"-scale {math.degrees(self.config.imaging_cellsize)}deg "
            + f"{visibility.path}"
        )

        print(f"WSClean command: {command}")
        completed_process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            # Raises exception on return code != 0
            check=True,
        )
        print(f"WSClean output:\n[{completed_process.stdout}]")

        default_output_fits_path = os.path.join(tmp_dir, self.OUTPUT_FITS_DIRTY)
        if output_fits_path is None:
            output_fits_path = default_output_fits_path
        else:
            shutil.copyfile(default_output_fits_path, output_fits_path)

        return Image(path=output_fits_path)

    def create_dirty_image_series(
        self,
        visibility: Visibility,
        /,
        *,
        output_fits_path: Optional[FilePathType] = None,
    ) -> List[FilePathType]:
        """
        This function splits a measurement set into the given number of
        (time) intervals. Each interval is then imaged seperately.
        The file name of all the image products of an interval will have
        the interval number preceded by a 't', e.g. wsclean-t0069-dirty.fits

        Returns:
            A list of absolute paths to the single .fits files created.
        """
        if visibility.format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility.format} is not supported, "
                "currently only MS is supported for WSClean imaging"
            )
        # TODO combine_across_frequencies
        # -channels-out <count>?
        if self.config.combine_across_frequencies is False:
            raise NotImplementedError(
                "combine_across_frequencies=False is currently not supported "
                "for the WSClean imager."
            )

        if self.config.intervals_out is None:
            raise ValueError(
                "You must set the parameter '-intervals-out' to call this "
                "function. If you want to create a single dirty image then "
                "call the function 'create_dirty_image()'."
            )

        if self.config.intervals_out <= 0:
            raise ValueError(
                "The parameter '-intervals-out' must be set to a value > 0."
            )

        tmp_dir = FileHandler().get_tmp_dir(
            prefix=self.TMP_PREFIX_DIRTY,
            purpose=self.TMP_PURPOSE_DIRTY,
        )

        command = _get_command_prefix(tmp_dir) + (
            f"{_WSCLEAN_BINARY} "
            + f"-size {self.config.imaging_npixel} {self.config.imaging_npixel} "
            + f"-scale {math.degrees(self.config.imaging_cellsize)}deg "
        )
        if self.config.intervals_out:
            command = command + f"-intervals-out {self.config.intervals_out} "

        command = command + f"{visibility.path}"

        print(f"WSClean command: {command}")
        completed_process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            # Raises exception on return code != 0
            check=True,
        )
        print(f"WSClean output:\n[{completed_process.stdout}]")

        fits_files: List[FilePathType] = []

        for file in glob.iglob(os.path.join(tmp_dir, "*-dirty.fits")):
            fits_files.append(file)

        self.fits_files = sorted(fits_files)

        return self.fits_files

    def animate(self, **kwargs) -> FuncAnimation:

        viewer = FitsSequenceViewer(self.fits_files)

        anim = viewer.show(**kwargs)

        return anim


# TODO Set kw_only=True after update to Python 3.10
# Right now, if one inherited superclass has a default-argument, you have to set
# defaults for all your attributes as well.
@dataclass
class WscleanImageCleanerConfig(ImageCleanerConfig):
    """Config / parameters of a WscleanImageCleaner.

    Adds parameters specific to WscleanImageCleaner.

    Attributes:
        niter (Optional[int]): Maximum number of clean iterations to perform.
            Defaults to 50000.
        mgain (Optional[float]): Cleaning gain for major iterations: Ratio of peak that
            will be subtracted in each major iteration. To use major iterations, 0.85 is
            a good value. Defaults to 0.8.
        auto_threshold (Optional[int]): Relative clean threshold. Estimate noise level
            using a robust estimator and stop at sigma x stddev. Defaults to 3.

    """

    niter: Optional[int] = 50000
    mgain: Optional[float] = 0.8
    auto_threshold: Optional[int] = 3


class WscleanImageCleaner(ImageCleaner):
    """Image cleaner based on the WSClean library.

    WSClean is integrated by calling the wsclean command line tool.
    The parameters in the config (WscleanImageCleanerConfig) attribute
    are passed to wsclean.
    Use the create_image_custom_command function if you need to set params
    not available in WscleanImageCleanerConfig.
    Parameters in the config that are explicitly set to None will not be passed to the
    command line tool, which will then resort to its own default values.

    Attributes:
        config (WscleanImageCleanerConfig): Config containing parameters for
            WSClean image cleaning.

    """

    TMP_PREFIX_CLEANED = "WSClean-cleaned-"
    TMP_PURPOSE_CLEANED = "Disk cache for WSClean cleaned images"

    OUTPUT_FITS_CLEANED = "wsclean-image.fits"

    def __init__(self, config: WscleanImageCleanerConfig) -> None:
        """Initializes the instance with a config.

        Args:
            config (WscleanImageCleanerConfig): see config attribute

        """
        super().__init__()
        self.config = config

    @override
    def create_cleaned_image(
        self,
        visibility: Visibility,
        /,
        *,
        dirty_fits_path: Optional[FilePathType] = None,
        output_fits_path: Optional[FilePathType] = None,
    ) -> Image:
        if visibility.format != "MS":
            raise NotImplementedError(
                f"Visibility format {visibility.format} is not supported, "
                "currently only MS is supported for WSClean imaging"
            )

        tmp_dir = FileHandler().get_tmp_dir(
            prefix=self.TMP_PREFIX_CLEANED,
            purpose=self.TMP_PURPOSE_CLEANED,
        )
        prefix = "pre_existing"
        if dirty_fits_path is not None:
            shutil.copyfile(
                dirty_fits_path,
                os.path.join(tmp_dir, f"{prefix}-dirty.fits"),
            )
        command = _get_command_prefix(tmp_dir) + (
            f"{_WSCLEAN_BINARY} "
            + (f"-reuse-dirty {prefix} " if dirty_fits_path is not None else "")
            + f"-size {self.config.imaging_npixel} {self.config.imaging_npixel} "
            + f"-scale {math.degrees(self.config.imaging_cellsize)}deg "
            + (f"-niter {self.config.niter} " if self.config.niter is not None else "")
            + (f"-mgain {self.config.mgain} " if self.config.mgain is not None else "")
            + (
                f"-auto-threshold {self.config.auto_threshold} "
                if self.config.auto_threshold is not None
                else ""
            )
            + str(visibility.path)
        )
        print(f"WSClean command: [{command}]")
        completed_process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            # Raises exception on return code != 0
            check=True,
        )
        print(f"WSClean output:\n[{completed_process.stdout}]")

        default_output_fits_path = os.path.join(tmp_dir, self.OUTPUT_FITS_CLEANED)
        if output_fits_path is None:
            output_fits_path = default_output_fits_path
        else:
            shutil.copyfile(default_output_fits_path, output_fits_path)

        return Image(path=output_fits_path)


TMP_PREFIX_CUSTOM = "WSClean-custom-"
TMP_PURPOSE_CUSTOM = "Disk cache for WSClean custom command images"


def create_image_custom_command(
    command: str,
    output_filenames: Union[str, List[str]] = "wsclean-image.fits",
) -> Union[Image, List[Image]]:
    """Create a dirty or cleaned image using your own command.

    Allows the use of the full WSClean functionality with all parameters.
    Command has to start with 'wsclean '.
    The working directory the command runs in will be a temporary directory.
    Use absolute paths to reference files or directories like the measurement set.

    Args:
        command: Command to execute.
            Example: wsclean -size 2048 2048
            -scale 0.00222222deg -niter 50000 -mgain 0.8
            -abs-threshold 100µJy /tmp/measurements.MS
        output_filenames: WSClean output filename(s)
            (relative to the working directory) that should be returned
            as Image objects. Can be a string for one file or a list of strings
            for multiple files.
            Example 1: "wsclean-image.fits"
            Example 2: ['wsclean-image.fits', 'wsclean-residual.fits']

    Returns:
        - If output_filenames is a **string**, returns an Image object of the file \
            output_filenames.
        - If output_filenames is a **list of strings**, returns a list of \
            Image objects, one object per filename in output_filenames.

    """

    tmp_dir = FileHandler().get_tmp_dir(
        prefix=TMP_PREFIX_CUSTOM,
        purpose=TMP_PURPOSE_CUSTOM,
    )
    expected_command_prefix = f"{_WSCLEAN_BINARY} "
    if not command.startswith(expected_command_prefix):
        raise ValueError(
            "Unexpected command. Expecting command to start with "
            f'"{expected_command_prefix}".'
        )
    command = _get_command_prefix(tmp_dir) + command
    print(f"WSClean command: [{command}]")
    completed_process = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        # Raises exception on return code != 0
        check=True,
    )
    print(f"WSClean output:\n[{completed_process.stdout}]")

    if isinstance(output_filenames, str):
        return Image(path=os.path.join(tmp_dir, output_filenames))
    else:
        return [
            Image(path=os.path.join(tmp_dir, output_filename))
            for output_filename in output_filenames
        ]


import warnings
from typing import List, Optional, Tuple, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from IPython.display import HTML, display

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u

# Optional reprojection (only used if reproject_if_needed=True)
try:
    from reproject import reproject_interp

    _HAS_REPROJECT = True
except Exception:
    _HAS_REPROJECT = False


# ----------------- small helpers -----------------


def _as_skycoord(x) -> SkyCoord:
    """
    Accept SkyCoord, (ra, dec) in degrees, or Nx2 array [[ra,dec],...], and return SkyCoord.
    """
    if x is None:
        return None
    if isinstance(x, SkyCoord):
        return x
    if isinstance(x, tuple) and len(x) == 2:
        ra, dec = x
        return SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg, frame="icrs")
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return SkyCoord(arr[:, 0] * u.deg, arr[:, 1] * u.deg, frame="icrs")
    raise ValueError(
        "Expected SkyCoord, (ra, dec) in degrees, or Nx2 array of degrees."
    )


def _wcs_compatible(w1: WCS, w2: WCS, tol: float = 1e-9) -> bool:
    """
    Tolerant WCS geometry check: same celestial axes/ctype/units and same CD/PC matrix (±tol).
    Ignores CRVAL/CRPIX/DATE-OBS.
    """
    w1c, w2c = w1.celestial, w2.celestial
    if w1c.naxis != w2c.naxis:
        return False
    if list(w1c.wcs.ctype) != list(w2c.wcs.ctype):
        return False
    if list(w1c.wcs.cunit) != list(w2c.wcs.cunit):
        return False

    def cd_like(w):
        if w.wcs.has_cd():
            return w.wcs.cd[:2, :2]
        pc = w.wcs.get_pc()[:2, :2]
        cdelt = np.diag(w.wcs.cdelt[:2])
        return pc @ cdelt

    return np.allclose(cd_like(w1c), cd_like(w2c), rtol=0.0, atol=tol)


# ----------------- main class -----------------


class FitsSequenceViewer:
    """
    Lazy FITS sequence viewer → HTML (FuncAnimation).
    Supports unfilled circular markers with radii specified in **degrees**.
    The number of sources is fixed (same N each frame).

    You can supply:
      - static sky positions (same N every frame), OR
      - per-frame sky positions (list length = n_frames, each length = N).
    """

    def __init__(self, fits_files: Union[List[str], List[FilePathType]]):
        if not fits_files:
            raise ValueError("No FITS files provided.")
        self.files = list(fits_files)
        self.n = len(self.files)

    # ---------- Data loading ----------
    @staticmethod
    def _load_frame(path: str) -> Tuple[np.ndarray, WCS]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            with fits.open(path, memmap=True) as hdul:
                arr = hdul[0].data[0, 0]  # adjust indexing if needed
                wcs = WCS(hdul[0].header).celestial
        return arr, wcs

    # ---------- Normalization ----------
    @staticmethod
    def _estimate_global_limits(
        files: List[str],
        percentiles=(0.1, 99.9),
        samples_per_frame: int = 5000,
        seed: int = 0,
    ) -> Tuple[float, float]:
        import random

        rng = random.Random(seed)
        pools = []
        for p in files:
            arr, _ = FitsSequenceViewer._load_frame(p)
            flat = arr.ravel()
            if flat.size <= samples_per_frame:
                pools.append(flat.astype(np.float32))
            else:
                idxs = np.array(rng.sample(range(flat.size), samples_per_frame))
                pools.append(flat[idxs].astype(np.float32))
        samples = np.concatenate(pools, axis=0)
        vmin, vmax = np.percentile(samples, percentiles)
        return float(vmin), float(vmax)

    # ---------- Public API ----------
    def create_animation(
        self,
        interval_ms: int = 200,
        normalize: str = "global",  # "global" or "per_frame"
        percentiles=(0.1, 99.9),
        samples_per_frame: int = 5000,
        cmap: str = "jet",
        show_colorbar: bool = True,
        cbar_label: str = "Flux [Jy/bm]",
        show_grid: bool = True,
        title_fmt: str = "Max Flux : {max:.3e} Jy/beam\nFrame : {idx}",
        # Reprojection / WCS handling
        reproject_if_needed: bool = True,
        # --------- Markers (fixed N per frame) ---------
        # EITHER provide static positions (applied every frame) ...
        markers_world: Optional[
            Union[SkyCoord, Tuple[Sequence[float], Sequence[float]], np.ndarray]
        ] = None,
        # ... OR provide per-frame positions (list length = n_frames, each SkyCoord|Nx2 of length N)
        markers_world_per_frame: Optional[
            List[Union[SkyCoord, Tuple[Sequence[float], Sequence[float]], np.ndarray]]
        ] = None,
        # Labels (length N) applied to all frames (optional)
        marker_labels: Optional[Sequence[str]] = None,
        # Radii in **degrees**: scalar or sequence length N (applied to all frames)
        marker_radius_deg: Optional[Union[float, Sequence[float]]] = None,
        # Marker style (unfilled circle; edgecolor/linewidth/zorder, etc.)
        marker_style: Optional[dict] = None,
        label_style: Optional[dict] = None,
        # optional saving
        save_gif: bool = False,
        output_gif: str = "animation.gif",
        save_mp4: bool = False,
        output_mp4: str = "animation.mp4",
    ) -> FuncAnimation:
        """
        Build the Matplotlib animation and return an HTML object. The figure is closed
        to suppress the static inline image; display the returned HTML to see the animation.
        """

        # --- reference frame & WCS ---
        first_arr, ref_wcs = self._load_frame(self.files[0])
        ref_shape = first_arr.shape

        # Check if reprojection is needed
        needs_reproj = any(
            not _wcs_compatible(ref_wcs, self._load_frame(p)[1]) for p in self.files[1:]
        )
        if needs_reproj and not reproject_if_needed:
            raise ValueError(
                "Frames have incompatible WCS for a single HTML animation.\n"
                "Set reproject_if_needed=True (requires `reproject`) or pre-align frames."
            )
        if needs_reproj and reproject_if_needed and not _HAS_REPROJECT:
            raise ImportError(
                "reproject is not installed. Install with `pip install reproject` "
                "or disable reprojection."
            )

        # Normalize
        if normalize == "global":
            vmin, vmax = self._estimate_global_limits(
                self.files, percentiles=percentiles, samples_per_frame=samples_per_frame
            )
        elif normalize == "per_frame":
            vmin = vmax = None
        else:
            raise ValueError("normalize must be 'global' or 'per_frame'.")

        # If reprojecting, make first_arr consistent with ref grid
        if needs_reproj:
            arr0, w0 = self._load_frame(self.files[0])
            first_arr, _ = reproject_interp((arr0, w0), ref_wcs, shape_out=ref_shape)

        # --- figure & initial image ---
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection=ref_wcs)
        if vmin is None or vmax is None:
            v0, V0 = np.percentile(first_arr, percentiles)
        else:
            v0, V0 = vmin, vmax
        im = ax.imshow(
            first_arr, cmap=cmap, animated=True, vmin=v0, vmax=V0, origin="lower"
        )

        if show_grid:
            ax.coords.grid(True, ls="dotted")
        ax.coords[0].set_axislabel("Right Ascension")
        ax.coords[1].set_axislabel("Declination")

        cbar = None
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, label=cbar_label)
            cbar.ax.tick_params(labelsize=6)

        ax.set_title(title_fmt.format(max=float(np.max(first_arr)), idx=0), fontsize=10)
        plt.tight_layout()

        # --- markers (fixed N each frame) ---
        if marker_style is None:
            marker_style = dict(edgecolor="yellow", linewidth=1.8, zorder=3)
        marker_style = {**marker_style, "fill": False}  # unfilled circles

        if label_style is None:
            label_style = dict(
                color="yellow", fontsize=8, ha="left", va="bottom", zorder=4
            )

        # Determine whether positions are static or per-frame
        has_per_frame = markers_world_per_frame is not None
        if has_per_frame and len(markers_world_per_frame) != self.n:
            raise ValueError(
                "markers_world_per_frame must have length = number of frames."
            )

        # Get N (number of sources)
        if has_per_frame:
            sc0 = _as_skycoord(markers_world_per_frame[0])
            N = len(sc0.ra)
        else:
            sc0 = _as_skycoord(markers_world)
            N = 0 if sc0 is None else len(sc0.ra)

        # Labels length check (if provided)
        if marker_labels is not None and len(marker_labels) != N:
            raise ValueError("marker_labels must have length N (number of markers).")

        # Radii (deg) → array length N (applied to all frames)
        if marker_radius_deg is None:
            rdeg = None
        elif np.isscalar(marker_radius_deg):
            rdeg = np.full(N, float(marker_radius_deg), dtype=float)
        else:
            rdeg = np.asarray(marker_radius_deg, float)
            if rdeg.size != N:
                raise ValueError("marker_radius_deg must be scalar or length N.")
        # Convert deg → pixels using geometric mean pixel scale
        sx_deg_pix, sy_deg_pix = proj_plane_pixel_scales(ref_wcs)[:2]  # deg per pix
        s_geom = float(np.sqrt(abs(sx_deg_pix * sy_deg_pix))) if N > 0 else 1.0
        rpix = (rdeg / s_geom) if rdeg is not None else None

        def world_to_pix(sc: SkyCoord, wcs_for: WCS) -> np.ndarray:
            x, y = wcs_for.all_world2pix(sc.ra.deg, sc.dec.deg, 0)
            return np.vstack([x, y]).T

        # Build initial marker patches from frame 0 positions
        circles: List[Circle] = []
        texts: List[plt.Text] = []

        if N > 0:
            if has_per_frame:
                sc_init = _as_skycoord(markers_world_per_frame[0])
            else:
                sc_init = sc0
            xy = world_to_pix(sc_init, ref_wcs)
            for i in range(N):
                r = float(rpix[i]) if rpix is not None else 10.0
                c = Circle((xy[i, 0], xy[i, 1]), radius=r, **marker_style)
                ax.add_patch(c)
                circles.append(c)
                if marker_labels:
                    texts.append(
                        ax.text(xy[i, 0], xy[i, 1], marker_labels[i], **label_style)
                    )

        # --- animation update ---
        def update(idx):
            frame_arr, frame_wcs = self._load_frame(self.files[idx])
            if needs_reproj:
                frame_arr, _ = reproject_interp(
                    (frame_arr, frame_wcs), ref_wcs, shape_out=ref_shape
                )

            # intensity scaling
            if normalize == "per_frame":
                vmin_i, vmax_i = np.percentile(frame_arr, percentiles)
                im.set_clim(vmin=vmin_i, vmax=vmax_i)
                if cbar is not None:
                    cbar.update_normal(im)

            im.set_array(frame_arr)
            ax.set_title(
                title_fmt.format(max=float(np.max(frame_arr)), idx=idx), fontsize=10
            )

            # update source positions for this frame (fixed N)
            if N > 0:
                sc = (
                    _as_skycoord(markers_world_per_frame[idx]) if has_per_frame else sc0
                )
                xy = world_to_pix(sc, ref_wcs)
                for i in range(N):
                    circles[i].center = (xy[i, 0], xy[i, 1])
                if marker_labels:
                    for i in range(N):
                        texts[i].set_position((xy[i, 0], xy[i, 1]))

            return [im] + circles + texts

        anim = FuncAnimation(
            fig, update, frames=self.n, interval=interval_ms, blit=True
        )

        # Optional saving (fixed canvas; avoid tight bbox)
        fps = 1000.0 / max(1, interval_ms)
        if save_gif:
            writer = PillowWriter(fps=fps)
            anim.save(output_gif, writer=writer)
            print(f"GIF saved as {output_gif}")
        if save_mp4:
            try:
                anim.save(output_mp4, writer="ffmpeg", dpi=100, fps=fps)
                print(f"MP4 saved as {output_mp4}")
            except Exception as e:
                print(f"MP4 save failed (need ffmpeg?): {e}")

        plt.close(fig)  # show only HTML, not the static PNG

        return anim

    def show(self, **kwargs) -> FuncAnimation:

        anim = self.create_animation(**kwargs)
        display(HTML(anim.to_jshtml()))

        return anim
