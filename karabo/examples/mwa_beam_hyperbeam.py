#! /usr/bin/env python

"""Script to plot MWA beam patterns using mwa_hyperbeam.

This example uses the mwa_hyperbeam library to calculate
MWA beam patterns at a specified frequency for a given feed (X or Y).

You can run the following examples in a notebook, served on http://127.0.0.1:8888/lab?token=...

    docker run --rm -it -p 8888:8888 \
        -v $PWD/karabo/examples:/home/jovyan/Karabo-Pipeline/karabo/examples \
        -w /home/jovyan/Karabo-Pipeline/karabo/examples \
        docker.io/d3vnull0/sp5505:latest

or in an interactive terminal:

    docker run --rm -it \
        -v $PWD/karabo/examples:/home/jovyan/Karabo-Pipeline/karabo/examples \
        -w /home/jovyan/Karabo-Pipeline/karabo/examples \
        docker.io/d3vnull0/sp5505:latest \
        bash -l

or individually in a non-interactive terminal:

    docker run --rm \
        -v $PWD/karabo/examples:/home/jovyan/Karabo-Pipeline/karabo/examples \
        -w /home/jovyan/Karabo-Pipeline/karabo/examples \
        docker.io/d3vnull0/sp5505:latest \
        python mwa_beam_hyperbeam.py \
        mwa_full_embedded_element_pattern.h5 \
        --freq-mhz 181 --pol X --projection SIN \
        --out mwa_hyperbeam_181MHz_x_sin.png

Example:
    # download beam file
    wget -O karabo/examples/mwa_full_embedded_element_pattern.h5 \
        http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5

    # Plot X feed beam at 181 MHz
    python karabo/examples/mwa_beam_hyperbeam.py \
        karabo/examples/mwa_full_embedded_element_pattern.h5 \
        --freq-mhz 181 --pol X --out mwa_hyperbeam_181MHz_x.png

    # Plot Y feed beam at 181 MHz
    python karabo/examples/mwa_beam_hyperbeam.py \
        karabo/examples/mwa_full_embedded_element_pattern.h5 \
        --freq-mhz 181 --pol Y --out mwa_hyperbeam_181MHz_y.png

    # Reproject to SIN projection
    python karabo/examples/mwa_beam_hyperbeam.py \
        karabo/examples/mwa_full_embedded_element_pattern.h5 \
        --freq-mhz 181 --pol X --projection SIN \
        --out mwa_hyperbeam_181MHz_x_sin.png
"""

import argparse
from typing import Optional, Tuple

import os
import sys
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

try:
    from mwa_hyperbeam import FEEBeam
except ImportError:
    raise ImportError(
        "mwa_hyperbeam not installed. Install with: pip install mwa_hyperbeam"
    )


def calculate_beam_grid(
    beam: FEEBeam,
    freq_hz: float,
    delays: Optional[np.ndarray] = None,
    amps: Optional[np.ndarray] = None,
    za_grid: Optional[np.ndarray] = None,
    az_grid: Optional[np.ndarray] = None,
    za_max_deg: float = 90.0,
    za_step_deg: float = 0.2,
    az_step_deg: float = 0.2,
    norm_to_zenith: bool = False,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate beam pattern on a regular az/za grid.

    Args:
        beam: FEEBeam object
        freq_hz: Frequency in Hz
        delays: Dipole delays (16 elements), None for zenith pointing
        amps: Dipole amplitudes (16 elements), None for unity
        za_grid: Optional zenith angle grid (degrees)
        az_grid: Optional azimuth grid (degrees)
        za_max_deg: Maximum zenith angle (degrees)
        za_step_deg: Zenith angle step (degrees)
        az_step_deg: Azimuth step (degrees)
        norm_to_zenith: Normalize beam to zenith (za=0, az=0)
        debug: Print debug information

    Returns:
        Tuple of (za_vals, az_vals, jones_xx, jones_yy)
        where jones_xx and jones_yy are complex 2D arrays of shape (n_za, n_az)
    """
    # Default delays and amps (zenith pointing, unity gain)
    # Note: amps must be 16 (bowties)
    if delays is None:
        delays = np.zeros(16, dtype=np.uint32)
    if amps is None:
        # Use 16 elements for bowties (MWA standard)
        amps = np.ones(16, dtype=np.float64)

    # Create grid if not provided
    if za_grid is None:
        za_grid = np.arange(0, za_max_deg + za_step_deg / 2, za_step_deg)
    if az_grid is None:
        az_grid = np.arange(0, 360, az_step_deg)

    n_za = len(za_grid)
    n_az = len(az_grid)

    if debug:
        print(f"Calculating beam on {n_za}x{n_az} grid...")
        print(
            f"  za: {za_grid[0]:.1f}° to {za_grid[-1]:.1f}° (step {za_step_deg:.2f}°)"
        )
        print(
            f"  az: {az_grid[0]:.1f}° to {az_grid[-1]:.1f}° (step {az_step_deg:.2f}°)"
        )
        print(f"  freq: {freq_hz / 1e6:.2f} MHz")
        print(f"  delays: {delays}")
        print(f"  amps: {amps if len(amps) <= 16 else f'{amps[:8]}...'}")

    # Create meshgrid for vectorized calculation
    az_2d, za_2d = np.meshgrid(np.radians(az_grid), np.radians(za_grid), indexing="xy")

    # Flatten for calc_jones_array (expects 1D arrays)
    az_flat = az_2d.flatten()
    za_flat = za_2d.flatten()

    if debug:
        print(f"  Calling calc_jones_array with {len(az_flat)} points...")

    # calc_jones_array signature: calc_jones_array(az_rad, za_rad, freq_hz, delays, amps, norm_to_zenith)
    # Returns array of shape (N, 4) where N = len(az_rad)
    # Each row is a flattened Jones matrix: [J_xx, J_xy, J_yx, J_yy]
    jones_array = beam.calc_jones_array(
        az_flat, za_flat, freq_hz, delays, amps, norm_to_zenith
    )

    if debug:
        print(f"  jones_array shape: {jones_array.shape}")
        print(f"  jones_array dtype: {jones_array.dtype}")
        print(f"  First Jones matrix: {jones_array[0]}")
        print(f"  Power at zenith: {np.abs(jones_array[0, 0]) ** 2:.6e}")

    # Extract XX and YY components and reshape to grid
    # Flattened format: [J_xx, J_xy, J_yx, J_yy]
    jones_xx = jones_array[:, 0].reshape(n_za, n_az)  # XX component (index 0)
    jones_yy = jones_array[:, 3].reshape(n_za, n_az)  # YY component (index 3)

    if debug:
        print(
            f"  jones_xx range: {np.abs(jones_xx).min():.3e} to {np.abs(jones_xx).max():.3e}"
        )
        print(
            f"  jones_yy range: {np.abs(jones_yy).min():.3e} to {np.abs(jones_yy).max():.3e}"
        )

    # WARNING: Check if beam is all zeros (possible hyperbeam/beam file incompatibility)
    if np.all(np.abs(jones_xx) == 0) and np.all(np.abs(jones_yy) == 0):
        print("WARNING: All Jones matrix values are zero!")
        print(
            "This may indicate an incompatibility between the beam file and hyperbeam version."
        )
        print("Please verify:")
        print("  1. The beam file is in the correct format for hyperbeam")
        print("  2. The hyperbeam version supports this beam file")
        print("  3. The beam file contains valid data at the requested frequency")

    return za_grid, az_grid, jones_xx, jones_yy


def plot_beam(
    beam_path: str,
    *,
    freq_mhz: float,
    pol: str = "X",
    delays: Optional[np.ndarray] = None,
    amps: Optional[np.ndarray] = None,
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
    za_step_deg: float = 0.2,
    az_step_deg: float = 0.2,
) -> None:
    """Plot MWA beam pattern using mwa_hyperbeam.

    Args:
        beam_path: Path to HDF5 beam file
        freq_mhz: Frequency in MHz
        pol: Feed polarization (X or Y)
        delays: Dipole delays (16 elements), None for zenith
        amps: Dipole amplitudes (16 elements), None for unity
        out: Output PNG filename
        show: Show plot interactively
        quantity: What to plot ('power', 'power_db', 'efield', 'phase')
        vmin: Color scale minimum
        vmax: Color scale maximum
        dpi: Figure DPI
        debug: Print debug information
        pointing_za_deg: Pointing zenith angle in degrees
        pointing_az_deg: Pointing azimuth in degrees
        projection: WCS projection code (TAN, SIN, STG, etc.) for reprojection
        proj_size_deg: Size of reprojected image in degrees (default 90)
        za_step_deg: Zenith angle grid step in degrees
        az_step_deg: Azimuth grid step in degrees
    """
    if debug:
        print("Debug info:")
        print(f"  beam_path: {beam_path}")
        print(f"  freq: {freq_mhz} MHz")
        print(f"  pol: {pol}")

    # Load beam
    if debug:
        print(f"  Loading FEEBeam from: {beam_path}")

    beam = FEEBeam(beam_path)

    # Get available frequencies and find closest
    available_freqs = beam.get_fee_beam_freqs()
    freq_idx = int(np.argmin(np.abs(available_freqs - freq_mhz * 1e6)))
    freq_hz = available_freqs[freq_idx]

    if debug:
        print("  FEEBeam loaded successfully")
        print(
            f"  Available freqs: {len(available_freqs)} from {available_freqs[0] / 1e6:.2f} to {available_freqs[-1] / 1e6:.2f} MHz"
        )
        print(
            f"  Requested: {freq_mhz} MHz, using: {freq_hz / 1e6:.2f} MHz (index {freq_idx})"
        )

    # Calculate beam on grid
    za_vals, az_vals, jones_xx, jones_yy = calculate_beam_grid(
        beam,
        freq_hz,
        delays=delays,
        amps=amps,
        za_step_deg=za_step_deg,
        az_step_deg=az_step_deg,
        debug=debug,
    )

    # Select polarization
    pol_upper = pol.upper()
    if pol_upper == "X":
        jones = jones_xx
    elif pol_upper == "Y":
        jones = jones_yy
    else:
        raise ValueError(f"Invalid pol: {pol}. Must be X or Y")

    if debug:
        print(f"  Using {pol_upper} polarization")
        print(f"  Jones shape: {jones.shape}")

    # Convert to power
    data = np.abs(jones) ** 2

    if debug:
        print(f"  Power range: {data.min():.3e} to {data.max():.3e}")

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
        z = np.abs(jones)
        cmap = "viridis"
        use_log = False
        cbar_label = "E-field magnitude"
    elif quantity == "phase":
        z = np.angle(jones, deg=True)
        cmap = "twilight"
        use_log = False
        cbar_label = "Phase [deg]"
    else:
        raise ValueError(f"Unknown quantity: {quantity}")

    # Mask invalid values
    z = np.where(np.isfinite(z), z, np.nan)

    # Setup plot
    plt.style.use("dark_background")

    if projection is not None:
        # Manual reprojection from polar (za, az) to Cartesian with projection
        if debug:
            print(
                f"  Reprojecting from polar to Cartesian {projection} centered on zenith..."
            )

        # Input data is on polar grid: z[za_idx, az_idx]
        # Create interpolator for polar data
        interp = RegularGridInterpolator(
            (za_vals, az_vals),
            z,
            bounds_error=False,
            fill_value=np.nan,
            method="linear",
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
        r_deg = np.sqrt(xx**2 + yy**2)

        # Apply inverse projection to get zenith angle (in degrees)
        proj_upper = projection.upper()
        if proj_upper == "SIN":
            # Orthographic: r_norm = sin(za)
            r_norm = r_deg / extent_deg
            za_cart = np.where(r_norm <= 1.0, np.degrees(np.arcsin(r_norm)), np.nan)
        elif proj_upper == "TAN":
            # Gnomonic: r_norm = tan(za)
            r_norm = r_deg / extent_deg
            za_cart = np.degrees(np.arctan(r_norm))
        elif proj_upper == "ARC":
            # Azimuthal equidistant: r_deg = za_deg
            za_cart = r_deg
        elif proj_upper == "ZEA":
            # Zenithal equal area: r_norm = 2*sin(za/2)
            r_norm = r_deg / extent_deg
            za_cart = np.where(
                r_norm <= 2.0, 2.0 * np.degrees(np.arcsin(r_norm / 2.0)), np.nan
            )
        elif proj_upper == "STG":
            # Stereographic: r_norm = 2*tan(za/2)
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
            print(
                f"  Valid pixels: {n_valid}/{npix * npix} ({100 * n_valid / (npix * npix):.1f}%)"
            )

        fig, ax = plt.subplots(figsize=(9, 8), dpi=dpi)
        z = z_proj
        extent = [-extent_deg, extent_deg, -extent_deg, extent_deg]
        x_label = f"E-W offset [deg] — {projection}"
        y_label = f"N-S offset [deg] — {projection}"
    else:
        # No projection - standard plotting
        fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
        extent = [az_vals.min(), az_vals.max(), za_vals.min(), za_vals.max()]
        x_label = "Azimuth [deg]"
        y_label = "Zenith Angle [deg]"

    # Determine color scale
    vmin_plot = vmin
    vmax_plot = vmax
    if vmin_plot is None and vmax_plot is None:
        zmin = np.nanmin(z)
        zmax = np.nanmax(z)
        if debug:
            print(f"  Color scale: min={zmin:.3e}, max={zmax:.3e}")
        if not np.isfinite(zmin) or not np.isfinite(zmax):
            if debug:
                print("  WARNING: No valid data in plot!")
            vmin_plot = 1e-20  # Use small value for log scale
            vmax_plot = 1e-10
        elif zmin == zmax:
            if zmin == 0:
                # All zeros - set a valid range for visualization
                vmin_plot = 1e-20
                vmax_plot = 1e-10
            else:
                eps = max(1e-12, abs(zmax) * 1e-6)
                vmin_plot = zmin - eps
                vmax_plot = zmax + eps

    # Plot
    # Ensure azimuth wraps smoothly at 0/360 to avoid discontinuity
    if projection is None:
        # append a 360° column duplicated from 0° to close the seam
        if az_vals[-1] < 360.0:
            az_vals = np.append(az_vals, 360.0)
            z = np.concatenate([z, z[:, :1]], axis=1)
            extent = [az_vals.min(), az_vals.max(), za_vals.min(), za_vals.max()]

    im = ax.imshow(
        z,  # no transpose: z[za_idx, az_idx] matches extent=[az_min, az_max, za_min, za_max]
        aspect="auto",
        origin="upper",  # show za increasing downward (0 at top, 90 at bottom)
        extent=extent,
        cmap=cmap,
        norm=LogNorm(vmin=vmin_plot, vmax=vmax_plot) if use_log else None,
        vmin=None if use_log else vmin_plot,
        vmax=None if use_log else vmax_plot,
    )

    # Add pointing marker if not zenith
    if pointing_za_deg != 0 or pointing_az_deg != 0:
        if projection is None:
            ax.plot(
                pointing_az_deg,
                pointing_za_deg,
                "w+",
                markersize=15,
                markeredgewidth=2,
                label="Pointing",
            )
            ax.legend(loc="upper right")

    # Set labels and title
    # Correct axis labels: x=Azimuth, y=Zenith Angle
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"MWA Beam (hyperbeam) — {freq_mhz:.2f} MHz — {pol} — {quantity}")

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
        description="Plot MWA beam pattern using mwa_hyperbeam"
    )
    parser.add_argument("beam_path", help="Path to HDF5 beam file")
    parser.add_argument(
        "--freq-mhz", type=float, required=True, help="Frequency in MHz"
    )
    parser.add_argument("--pol", default="X", help="Feed polarization: X or Y")
    parser.add_argument(
        "--quantity",
        default="power",
        choices=["power", "power_db", "efield", "phase"],
        help="Quantity to display",
    )
    parser.add_argument("--out", default=None, help="Output image filename (PNG)")
    parser.add_argument(
        "--show", action="store_true", help="Show the plot interactively"
    )
    parser.add_argument("--vmin", type=float, default=None, help="Color scale minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument(
        "--pointing-za",
        type=float,
        default=0.0,
        help="Pointing zenith angle in degrees (for marker)",
    )
    parser.add_argument(
        "--pointing-az",
        type=float,
        default=0.0,
        help="Pointing azimuth in degrees (for marker)",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default=None,
        help="WCS projection code for reprojection (TAN, SIN, STG, ARC, etc.)",
    )
    parser.add_argument(
        "--proj-size",
        type=float,
        default=90.0,
        help="Size of reprojected image in degrees (default: 90)",
    )
    parser.add_argument(
        "--za-step",
        type=float,
        default=1.0,
        help="Zenith angle grid step in degrees (default: 1.0)",
    )
    parser.add_argument(
        "--az-step",
        type=float,
        default=1.0,
        help="Azimuth grid step in degrees (default: 1.0)",
    )

    args = parser.parse_args()

    # Optional: specify delays for beam pointing
    # delays = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint32)
    delays = None
    amps = None

    plot_beam(
        args.beam_path,
        freq_mhz=float(args.freq_mhz),
        pol=str(args.pol),
        delays=delays,
        amps=amps,
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
        za_step_deg=float(args.za_step),
        az_step_deg=float(args.az_step),
    )


if __name__ == "__main__":
    main()
