#! /usr/bin/env python

"""Script to plot MWA beam patterns from HDF5 beam files.

This example loads an MWA beam file (HDF5 format) via pyuvbeam and produces
beam pattern plots at a specified frequency for a given feed (X or Y).

You can run the following examples in a notebook, served on http://127.0.0.1:8888/lab?token=...

    docker run --rm -it -p 8888:8888 \
        -v $PWD/karabo/examples:/home/jovyan/Karabo-Pipeline/karabo/examples \
        -w /home/jovyan/Karabo-Pipeline/karabo/examples \
        ghcr.io/d3v-null/sp5505-karabo-pipeline:sha-0ee53c2-pretest

or in an interactive terminal:

    docker run --rm -it \
        -v $PWD/karabo/examples:/home/jovyan/Karabo-Pipeline/karabo/examples \
        -w /home/jovyan/Karabo-Pipeline/karabo/examples \
        ghcr.io/d3v-null/sp5505-karabo-pipeline:sha-0ee53c2-pretest \
        bash -l

or individually in a non-interactive terminal:

    docker run --rm \
        -v $PWD/karabo/examples:/home/jovyan/Karabo-Pipeline/karabo/examples \
        -w /home/jovyan/Karabo-Pipeline/karabo/examples \
        ghcr.io/d3v-null/sp5505-karabo-pipeline:sha-0ee53c2-pretest \
        python mwa_pyuvbeam.py \
        mwa_full_embedded_element_pattern.h5 \
        --freq-mhz 150 --pol X --projection SIN --debug \
        --out mwa_beam_150mhz_x_sin.png

Example:
    # download beam file
    wget -O karabo/examples/mwa_full_embedded_element_pattern.h5 http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5

    # Plot X feed beam at 150 MHz
    python karabo/examples/mwa_pyuvbeam.py \
        karabo/examples/mwa_full_embedded_element_pattern.h5 \
        --freq-mhz 150 --pol X --out mwa_beam_150mhz_x.png

    # Plot Y feed beam at 180 MHz
    python karabo/examples/mwa_pyuvbeam.py \
        karabo/examples/mwa_full_embedded_element_pattern.h5 \
        --freq-mhz 180 --pol Y --out mwa_beam_180mhz_y.png

    # Show interactively with debug info
    python karabo/examples/mwa_pyuvbeam.py \
        karabo/examples/mwa_full_embedded_element_pattern.h5 \
        --freq-mhz 150 --pol X --debug --show

    # Reproject to gnomonic (TAN) projection
    python karabo/examples/mwa_pyuvbeam.py \
        karabo/examples/mwa_full_embedded_element_pattern.h5 \
        --freq-mhz 150 --pol X --projection TAN --proj-size 60 \
        --out mwa_beam_150mhz_x_tan.png
"""

import argparse
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pyuvdata import UVBeam
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator


def _read_beam(path: str) -> UVBeam:
    """Read a beam file using pyuvbeam.

    Args:
        path: Path to beam file (HDF5, FITS, etc.)

    Returns:
        UVBeam object
    """
    beam = UVBeam()
    lower = path.lower()
    if lower.endswith(".h5") or lower.endswith(".hdf5"):
        beam.read_mwa_beam(path, run_check=False)
    elif lower.endswith(".fits"):
        beam.read_beamfits(path, run_check=False)
    else:
        # Fallback to generic reader
        beam.read(path, run_check=False)
    return beam


def _find_freq_index(beam: UVBeam, freq_mhz: float) -> Tuple[int, float]:
    """Find the closest frequency index in the beam.

    Args:
        beam: UVBeam object
        freq_mhz: Target frequency in MHz

    Returns:
        Tuple of (frequency index, actual frequency in MHz)
    """
    freq_hz = freq_mhz * 1e6
    freqs_hz = beam.freq_array[0]  # Shape is (Nspws, Nfreqs)
    idx = int(np.argmin(np.abs(freqs_hz - freq_hz)))
    actual_freq_mhz = float(freqs_hz[idx] / 1e6)
    return idx, actual_freq_mhz


def _select_pol_index(beam: UVBeam, pol: str) -> int:
    """Select polarization index from beam.

    Args:
        beam: UVBeam object
        pol: Polarization string (e.g., 'X', 'Y')

    Returns:
        Polarization index
    """
    from pyuvdata import utils as uvutils
    pol_num = uvutils.polstr2num(pol, x_orientation=beam.x_orientation)
    try:
        return int(np.where(beam.polarization_array == pol_num)[0][0])
    except (IndexError, TypeError) as exc:
        # Build available polarizations list
        if beam.polarization_array is not None and len(beam.polarization_array) > 0:
            available_list = []
            for p in beam.polarization_array:
                try:
                    pstr = uvutils.polnum2str(p, x_orientation=beam.x_orientation)
                    available_list.append(pstr[0] if isinstance(pstr, (list, tuple)) else pstr)
                except Exception:
                    available_list.append(str(p))
            available = ",".join(available_list)
        else:
            available = "none"
        raise ValueError(
            f"Polarization {pol} not found. Available: {available}"
        ) from exc


def plot_beam(
    beam_path: str,
    *,
    freq_mhz: float,
    pol: str = "X",
    out: Optional[str] = None,
    show: bool = False,
    quantity: str = "power",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 150,
    debug: bool = False,
    pointing_za_deg: float = 0.0,
    pointing_az_deg: float = 0.0,
    projection: Optional[str] = None,
    proj_size_deg: float = 90.0,
) -> None:
    """Plot MWA beam pattern at a given frequency.

    Args:
        beam_path: Path to beam file
        freq_mhz: Frequency in MHz
        pol: Feed polarization (X or Y)
        out: Output PNG filename
        show: Show plot interactively
        quantity: What to plot ('power', 'efield', 'phase')
        vmin: Color scale minimum
        vmax: Color scale maximum
        dpi: Figure DPI
        debug: Print debug information
        pointing_za_deg: Pointing zenith angle in degrees
        pointing_az_deg: Pointing azimuth in degrees
        projection: WCS projection code (TAN, SIN, STG, etc.) for reprojection
        proj_size_deg: Size of reprojected image in degrees (default 90)
    """
    beam = _read_beam(beam_path)

    if debug:
        print("Debug info:")
        print(f"  beam_path: {beam_path}")
        print(f"  beam_type: {beam.beam_type}")
        print(f"  pixel_coordinate_system: {beam.pixel_coordinate_system}")
        print(f"  data_array shape: {beam.data_array.shape}")
        print(f"  freq_array shape: {beam.freq_array.shape}")
        print(f"  freq range [MHz]: {beam.freq_array.min() / 1e6:.2f} - {beam.freq_array.max() / 1e6:.2f}")
        print(f"  polarization_array: {beam.polarization_array}")
        print(f"  x_orientation: {beam.x_orientation}")
        if hasattr(beam, 'feed_array') and beam.feed_array is not None:
            print(f"  feed_array: {beam.feed_array}")
        print(f"  axis1_array shape: {beam.axis1_array.shape}")
        print(f"  axis2_array shape: {beam.axis2_array.shape}")

        # Show available feeds/polarizations
        if beam.polarization_array is None or len(beam.polarization_array) == 0:
            # Feed-based beam (typical for MWA)
            n_feeds = beam.data_array.shape[2] if beam.data_array.ndim >= 3 else 1
            print(f"  *** Feed-based beam with {n_feeds} feeds ***")
            print(f"  Available feeds: {', '.join(['X', 'Y'][:n_feeds])}")
            print("  Use --pol X or --pol Y")
        else:
            # Polarization-based beam
            from pyuvdata import utils as uvutils
            pol_strs = []
            for p in beam.polarization_array:
                try:
                    pstr = uvutils.polnum2str(p, x_orientation=beam.x_orientation)
                    pol_strs.append(pstr[0] if isinstance(pstr, (list, tuple)) else pstr)
                except Exception:
                    pol_strs.append(f"{p}")
            print("  *** Polarization-based beam ***")
            print(f"  Available polarizations: {', '.join(pol_strs)}")
            print(f"  Use --pol {pol_strs[0] if pol_strs else 'XX'}")

    # Find frequency index
    freq_idx, actual_freq_mhz = _find_freq_index(beam, freq_mhz)
    if debug:
        print(f"  requested freq: {freq_mhz} MHz, using: {actual_freq_mhz} MHz (index {freq_idx})")

    # Select polarization or feed
    # MWA beams may have feeds instead of polarizations
    if beam.polarization_array is None or len(beam.polarization_array) == 0:
        # Try using feeds
        if debug:
            print("  No polarization_array, attempting to use feed-based selection")
        # For MWA, feeds are typically X=0, Y=1
        feed_map = {"X": 0, "Y": 1}
        if pol.upper() in feed_map:
            pol_idx = feed_map[pol.upper()]
            if debug:
                print(f"  pol: {pol} -> feed index {pol_idx}")
        else:
            pol_idx = 0  # Default to first feed
            if debug:
                print(f"  pol: {pol} not in feed map, using index 0")
    else:
        pol_idx = _select_pol_index(beam, pol)
        if debug:
            print(f"  pol: {pol} -> index {pol_idx}")

    # Convert to power beam if needed
    if beam.beam_type == "efield":
        if debug:
            print("  converting efield beam to power...")
        beam.efield_to_power(inplace=True, calc_cross_pols=False)

    # Extract beam data: shape is (Naxes_vec, Nspws, Nfeeds/Npols, Nfreqs, Naxis1, Naxis2)
    # For power beam: (1, Nspws, Npols, Nfreqs, Naxis1, Naxis2)
    data = beam.data_array[0, 0, pol_idx, freq_idx, :, :]  # (Naxis1, Naxis2)

    # Get coordinate axes
    if beam.pixel_coordinate_system == "az_za":
        x_vals = np.rad2deg(beam.axis2_array)  # azimuth
        y_vals = np.rad2deg(beam.axis1_array)  # zenith angle
        x_label = "Azimuth [deg]"
        y_label = "Zenith Angle [deg]"
    elif beam.pixel_coordinate_system == "orthoslant_zenith":
        # Orthoslant projection
        x_vals = beam.axis2_array
        y_vals = beam.axis1_array
        x_label = "East-West [sin(θ)]"
        y_label = "North-South [sin(θ)]"
    else:
        x_vals = beam.axis2_array
        y_vals = beam.axis1_array
        x_label = "Axis2 [" + beam.pixel_coordinate_system + "]"
        y_label = "Axis1 [" + beam.pixel_coordinate_system + "]"

    if debug:
        print(f"  data shape: {data.shape}")
        print(f"  data min/max: {np.nanmin(data):.6e} / {np.nanmax(data):.6e}")
        print(f"  x_vals (axis2) range: {x_vals.min():.3f} to {x_vals.max():.3f}")
        print(f"  y_vals (axis1) range: {y_vals.min():.3f} to {y_vals.max():.3f}")

    # Apply quantity transformation
    if quantity == "power":
        z = data
        cmap = "rainbow"
        use_log = True
        cbar_label = "Power [linear]"
    elif quantity == "power_db":
        z = 10 * np.log10(np.maximum(data, 1e-20))
        cmap = "rainbow"
        use_log = False
        cbar_label = "Power [dB]"
    elif quantity == "efield":
        # This only works if we didn't convert to power
        z = np.abs(data)
        cmap = "viridis"
        use_log = False
        cbar_label = "E-field magnitude"
    elif quantity == "phase":
        z = np.angle(data, deg=True)
        cmap = "twilight"
        use_log = False
        cbar_label = "Phase [deg]"
    else:
        raise ValueError(f"Unknown quantity: {quantity}")

    # Mask invalid values
    z = np.where(np.isfinite(z), z, np.nan)

    # Setup plot with optional WCS projection
    plt.style.use("dark_background")

    if projection is not None and beam.pixel_coordinate_system == "az_za":
        # Manual reprojection from polar (za, az) to Cartesian with projection
        if debug:
            print(f"  Reprojecting from polar to Cartesian {projection} centered on zenith...")

        # Input data is on polar grid: z[za_idx, az_idx]
        # where za = x_vals (0-90°), az = y_vals (0-360°)

        # Create interpolator for polar data (wrap azimuth for continuity)
        interp = RegularGridInterpolator(
            (x_vals, y_vals), z,
            bounds_error=False, fill_value=np.nan,
            method='linear'
        )

        # Output grid: Cartesian with desired projection
        npix = int(2 * proj_size_deg / 0.5)  # 0.5° pixel scale
        pixel_scale = 2 * proj_size_deg / npix

        # Create coordinate grids for output (in degrees from zenith)
        extent_deg = proj_size_deg
        x_cart = np.linspace(-extent_deg, extent_deg, npix)
        y_cart = np.linspace(-extent_deg, extent_deg, npix)
        xx, yy = np.meshgrid(x_cart, y_cart)

        # Convert Cartesian projection coordinates to (za, az)
        # Coordinates xx, yy are in degrees on the projection plane (±extent_deg)
        # Normalize to get dimensionless projection coordinates
        r_deg = np.sqrt(xx**2 + yy**2)

        # Apply inverse projection to get zenith angle (in degrees)
        proj_upper = projection.upper()
        if proj_upper == "SIN":
            # Orthographic: r_norm = sin(za) where r_norm = r_deg / extent_deg
            # So: za = arcsin(r_norm)
            # Valid only where r_norm <= 1 (i.e., r_deg <= extent_deg)
            r_norm = r_deg / extent_deg
            za_cart = np.where(r_norm <= 1.0,
                              np.degrees(np.arcsin(r_norm)),
                              np.nan)
        elif proj_upper == "TAN":
            # Gnomonic: r_norm = tan(za) where r_norm = r_deg / extent_deg
            # So: za = arctan(r_norm)
            r_norm = r_deg / extent_deg
            za_cart = np.degrees(np.arctan(r_norm))
        elif proj_upper == "ARC":
            # Azimuthal equidistant: r_deg = za_deg
            za_cart = r_deg
        elif proj_upper == "ZEA":
            # Zenithal equal area: r_norm = 2*sin(za/2)
            # So: za = 2*arcsin(r_norm/2)
            # Valid where r_norm/2 <= 1 (i.e., r_norm <= 2, r_deg <= 2*extent_deg)
            r_norm = r_deg / extent_deg
            za_cart = np.where(r_norm <= 2.0,
                              2.0 * np.degrees(np.arcsin(r_norm / 2.0)),
                              np.nan)
        elif proj_upper == "STG":
            # Stereographic: r_norm = 2*tan(za/2)
            # So: za = 2*arctan(r_norm/2)
            r_norm = r_deg / extent_deg
            za_cart = 2.0 * np.degrees(np.arctan(r_norm / 2.0))
        else:
            # Default to linear (ARC)
            za_cart = r_deg

        # Azimuth: tan(az) = x/y (using math convention where y is north)
        az_cart = np.degrees(np.arctan2(xx, yy)) % 360

        if debug:
            print(f"  Output: ({npix}x{npix}), pixel scale {pixel_scale:.3f}°/pix")
            print(f"  Cartesian extent: ±{extent_deg}° from zenith")

        # Interpolate from polar to Cartesian
        points = np.column_stack([za_cart.ravel(), az_cart.ravel()])
        z_proj = interp(points).reshape((npix, npix))

        # Mask pixels beyond horizon (za > 90°)
        z_proj[za_cart > 90] = np.nan

        if debug:
            n_valid = np.sum(np.isfinite(z_proj))
            print(f"  Valid pixels: {n_valid}/{npix*npix} ({100*n_valid/(npix*npix):.1f}%)")

        # Use regular matplotlib (data is already in Cartesian)
        fig, ax = plt.subplots(figsize=(9, 8), dpi=dpi)
        z = z_proj

        # Set extent for Cartesian coordinates centered on zenith
        extent = [-extent_deg, extent_deg, -extent_deg, extent_deg]

        # Store extent for later use in imshow
        cart_extent = extent

        x_label = f"E-W offset [deg] — {projection}"
        y_label = f"N-S offset [deg] — {projection}"
    else:
        # No projection - standard plotting
        fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)

    # Determine color scale
    vmin_plot = vmin
    vmax_plot = vmax
    if vmin_plot is None and vmax_plot is None:
        zmin = np.nanmin(z)
        zmax = np.nanmax(z)
        if np.isfinite(zmin) and np.isfinite(zmax) and zmin == zmax:
            eps = max(1e-12, abs(zmax) * 1e-6)
            vmin_plot = zmin - eps
            vmax_plot = zmax + eps

    # Plot with explicit extent
    if projection is not None and beam.pixel_coordinate_system == "az_za":
        # Use Cartesian extent from reprojection
        extent = cart_extent
    else:
        # Use original coordinate extent
        extent = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]

    im = ax.imshow(
        z.T,  # Transpose to match axis convention
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=LogNorm(vmin=vmin_plot, vmax=vmax_plot) if use_log else None,
        vmin=None if use_log else vmin_plot,
        vmax=None if use_log else vmax_plot,
    )

    # Add pointing marker if not zenith
    if pointing_za_deg != 0 or pointing_az_deg != 0:
        if beam.pixel_coordinate_system == "az_za":
            ax.plot(pointing_az_deg, pointing_za_deg, 'w+', markersize=15,
                    markeredgewidth=2, label="Pointing")
            ax.legend(loc='upper right')

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"MWA Beam — {actual_freq_mhz:.2f} MHz — {pol} — {quantity}")

    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()

    if out is not None:
        plt.savefig(out, dpi=dpi)
        print(f"Saved beam plot: {out}")
        plt.close(fig)
    if show and out is None:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot MWA beam pattern from HDF5 beam file"
    )
    parser.add_argument("beam_path", help="Path to beam file (HDF5, FITS)")
    parser.add_argument(
        "--freq-mhz",
        type=float,
        required=True,
        help="Frequency in MHz"
    )
    parser.add_argument(
        "--pol",
        default="X",
        help="Feed polarization: X or Y"
    )
    parser.add_argument(
        "--quantity",
        default="power",
        choices=["power", "power_db", "efield", "phase"],
        help="Quantity to display",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image filename (PNG)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively"
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Color scale minimum"
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Color scale maximum"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information"
    )
    parser.add_argument(
        "--pointing-za",
        type=float,
        default=0.0,
        help="Pointing zenith angle in degrees (for marker)"
    )
    parser.add_argument(
        "--pointing-az",
        type=float,
        default=0.0,
        help="Pointing azimuth in degrees (for marker)"
    )
    parser.add_argument(
        "--projection",
        type=str,
        default=None,
        help="WCS projection code for reprojection (TAN, SIN, STG, ARC, etc.)"
    )
    parser.add_argument(
        "--proj-size",
        type=float,
        default=90.0,
        help="Size of reprojected image in degrees (default: 90)"
    )

    args = parser.parse_args()

    plot_beam(
        args.beam_path,
        freq_mhz=float(args.freq_mhz),
        pol=str(args.pol),
        out=args.out,
        show=bool(args.show),
        quantity=str(args.quantity),
        vmin=args.vmin,
        vmax=args.vmax,
        dpi=int(args.dpi),
        debug=bool(args.debug),
        pointing_za_deg=float(args.pointing_za),
        pointing_az_deg=float(args.pointing_az),
        projection=args.projection,
        proj_size_deg=float(args.proj_size),
    )


if __name__ == "__main__":
    main()
