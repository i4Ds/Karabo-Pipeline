#! /usr/bin/env python

"""Script to plot a time-frequency (waterfall) for a specified baseline.

This example loads an MWA UVFITS (or MS) file via pyuvdata and produces a
waterfall plot for a chosen antenna pair and polarisation.

Example:
    wget https://projects.pawsey.org.au/high0.uvfits/hyp_1184702048_ionosub_ssins_30l_src8k_300it_8s_80kHz_i1000.uvfits

    # inspect bl 23x26 xx
    python karabo/examples/mwa_waterfall_baseline.py \
        hyp_1184702048_ionosub_ssins_30l_src8k_300it_8s_80kHz_i1000.uvfits \
        --antpair 23,26 --pol XX --out mwa_23x26_xx.png

    # inspect bl 23x26 xx fft
    python karabo/examples/mwa_waterfall_baseline.py \
        hyp_1184702048_ionosub_ssins_30l_src8k_300it_8s_80kHz_i1000.uvfits \
        --fft --all-baselines --max-baseline 500 --bins 0 \
        --out mwa_all.png

    # inspect all baselines fft
    python karabo/examples/mwa_waterfall_baseline.py \
        hyp_1184702048_ionosub_ssins_30l_src8k_300it_8s_80kHz_i1000.uvfits \
        --antpair 23,26 --pol XX --fft --out mwa_23x26_fft_xx.png
"""

import argparse
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pyuvdata import UVData
from pyuvdata import utils as uvutils


def _read_uv(path: str) -> UVData:
    uv = UVData()
    lower = path.lower()
    if lower.endswith(".uvfits"):
        uv.read_uvfits(path)
    elif lower.endswith(".ms"):
        uv.read_ms(path)
    else:
        # Fallback to generic reader if extension is not standard
        uv.read(path)
    return uv


def _get_antnums_from_names(uv: UVData, name_a: str, name_b: str) -> Tuple[int, int]:
    names = np.array(uv.antenna_names)
    nums = np.array(uv.antenna_numbers)
    try:
        a = int(nums[names == name_a][0])
        b = int(nums[names == name_b][0])
    except IndexError as exc:
        raise ValueError(f"Antenna names not found: {name_a}, {name_b}") from exc
    return a, b


def _select_pol_index(uv: UVData, pol: Optional[str]) -> int:
    if pol is None:
        return 0
    # Convert to pyuvdata numeric code; allow common inputs (e.g., XX, YY, XY, YX, I, Q, U, V)
    pol_num = uvutils.polstr2num(pol)
    # Find index in UVData.polarization_array
    try:
        return int(np.where(uv.polarization_array == pol_num)[0][0])
    except IndexError as exc:
        available = ",".join(uvutils.polnum2str(uv.polarization_array))
        raise ValueError(
            f"Polarisation {pol} not present. Available: {available}"
        ) from exc


def _extract_waterfall(
    uv: UVData,
    ant_a: int,
    ant_b: int,
    pol_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (times_mjd, freqs_hz, data_2d) for the given baseline and pol.

    data_2d has shape (Ntimes, Nfreqs) and preserves time ordering.
    """
    # Identify baseline rows (both directions allowed)
    ant1 = uv.ant_1_array
    ant2 = uv.ant_2_array
    mask = ((ant1 == ant_a) & (ant2 == ant_b)) | ((ant1 == ant_b) & (ant2 == ant_a))
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"Baseline ({ant_a},{ant_b}) not found in dataset.")

    # Determine data_array dimensionality across pyuvdata versions
    data = uv.data_array
    # data shape can be (Nblt, Nspw, Nfreq, Npol) or (Nblt, Nfreq, Npol)
    if data.ndim == 4:
        # Assume single spw; squeeze that axis
        data = np.squeeze(data[:, 0, :, :])  # -> (Nblt, Nfreq, Npol)
    # Select only matching blt indices along axis 0 without squeezing when size==1
    data = np.take(data, idx, axis=0)
    # Ensure 3D output even for a single row selection
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim != 3:
        raise RuntimeError(f"Unexpected data_array shape after selection: {data.shape}")

    # Prepare matching flag array selection
    flags = uv.flag_array
    if flags.ndim == 4:
        flags = np.squeeze(flags[:, 0, :, :])  # -> (Nblt, Nfreq, Npol)
    flags = np.take(flags, idx, axis=0)
    if flags.ndim == 2:
        flags = flags[np.newaxis, :, :]
    elif flags.ndim != 3:
        raise RuntimeError(
            f"Unexpected flag_array shape after selection: {flags.shape}"
        )

    # Select polarisation
    data_pol = data[:, :, pol_index]  # (Nblt, Nfreq)
    flags_pol = flags[:, :, pol_index]  # (Nblt, Nfreq)

    # Sort by time and build 2D array
    times = np.take(uv.time_array, idx, axis=0)
    order = np.argsort(times)
    times_sorted = times[order]
    wf = data_pol[order, :]
    flags_sorted = flags_pol[order, :]

    # Frequencies: pyuvdata stores freq_array with shape (Nspw, Nfreq)
    if uv.freq_array.ndim == 2:
        freqs = uv.freq_array[0]
    else:
        freqs = uv.freq_array
    return times_sorted, freqs, wf, flags_sorted


def _compute_quantity(wf: np.ndarray, quantity: str) -> np.ndarray:
    q = quantity.lower()
    if q in ("amp", "amplitude"):
        return np.abs(wf)
    if q == "real":
        return np.real(wf)
    if q == "imag":
        return np.imag(wf)
    if q == "phase":
        return np.angle(wf, deg=True)
    raise ValueError(f"Unknown quantity: {quantity}")


def _log10_safe(z: np.ndarray) -> np.ndarray:
    """Return log10 of z for z>0, NaN elsewhere (keeps flagged NaNs)."""
    z = z.astype(float, copy=False)
    out = np.full_like(z, np.nan, dtype=float)
    mask = np.isfinite(z) & (z > 0)
    if np.any(mask):
        out[mask] = np.log10(z[mask])
    return out


def _all_baselines_binned_matrix(
    uv: UVData,
    pol_index: int,
    *,
    quantity: str,
    fft: bool,
    bins: int,
    debug: bool,
    max_baseline_m: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, int]:
    """Build a 2D matrix: baseline-length bins (rows) vs freq/delay (cols).

    Returns:
        z: (Nbins, Ncols) data matrix (float)
        x_vals: x-axis values (MHz if not FFT, or microseconds if FFT)
        y_edges: baseline length bin edges (m)
        x_label, y_label: axis labels
        num_pairs: number of unique baseline pairs included
    """
    ant1 = uv.ant_1_array
    ant2 = uv.ant_2_array
    bl_keys = np.stack([np.minimum(ant1, ant2), np.maximum(ant1, ant2)], axis=1)
    uniq_pairs, first_idx = np.unique(bl_keys, axis=0, return_index=True)
    # Sort by first occurrence to keep stable ordering
    order = np.argsort(first_idx)
    uniq_pairs = uniq_pairs[order]

    # Group indices per pair
    pair_to_indices: list[np.ndarray] = []
    pair_lengths: list[float] = []
    for p in uniq_pairs:
        mask = (ant1 == p[0]) & (ant2 == p[1]) | (ant1 == p[1]) & (ant2 == p[0])
        idx = np.where(mask)[0]
        pair_to_indices.append(idx)
        # Representative length: median |uvw| over this pair's rows
        uvw = uv.uvw_array[idx]
        blens = np.sqrt((uvw**2).sum(axis=1))
        pair_lengths.append(float(np.nanmedian(blens)))

    num_pairs = len(pair_to_indices)
    lengths = np.array(pair_lengths, dtype=float)
    if num_pairs == 0:
        raise ValueError("No baselines found")

    # Optional baseline length filtering
    if max_baseline_m is not None:
        keep_mask = lengths <= float(max_baseline_m)
        if not np.any(keep_mask):
            raise ValueError(
                f"No baselines ≤ {max_baseline_m} m found (min={float(np.nanmin(lengths)):.2f}, "
                f"max={float(np.nanmax(lengths)):.2f})"
            )
        lengths = lengths[keep_mask]
        pair_to_indices = [pair_to_indices[i] for i, k in enumerate(keep_mask) if k]
        num_pairs = len(pair_to_indices)

    # Frequency axis reference
    if uv.freq_array.ndim == 2:
        freqs_hz = uv.freq_array[0]
    else:
        freqs_hz = uv.freq_array
    nfreq = freqs_hz.shape[0]

    # Build per-pair, time-averaged complex vector per frequency
    per_pair_vectors: list[np.ndarray] = []
    for idx in pair_to_indices:
        # Extract complex data and flags for rows in this pair
        data = uv.data_array
        flags = uv.flag_array
        if data.ndim == 4:
            data = np.squeeze(data[:, 0, :, :])
            flags = np.squeeze(flags[:, 0, :, :])
        dsel = data[idx][:, :, pol_index]  # (rows, nfreq)
        fsel = flags[idx][:, :, pol_index]  # (rows, nfreq)
        # Time-average ignoring flagged samples (complex average)
        # Set flagged to NaN, then nanmean along axis 0
        dmask = np.where(fsel, np.nan + 0j, dsel.astype(np.complex128))
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(dmask, axis=0)
        per_pair_vectors.append(avg)  # (nfreq,)

    # Define baseline length bins (auto sqrt if bins <= 0)
    if bins is None or bins <= 0:
        bins = int(max(1, round(float(np.sqrt(float(len(lengths)))))))
    y_edges = np.linspace(
        float(np.nanmin(lengths)), float(np.nanmax(lengths)), bins + 1
    )

    # X axis and transform
    if fft:
        # Build delay grid 0..4 microseconds (seconds in tau_s)
        num_tau = nfreq // 2 + 1
        tau_s = np.linspace(0.0, 4e-6, num=num_tau, endpoint=True)
        y_vals = tau_s * 1e6  # microseconds
        y_label = "Delay [µs]"
        x_label = "Baseline length [m]"
        # NUDFT per bin using true freqs and complex, time-averaged vectors
        z = np.full((bins, y_vals.shape[0]), np.nan, dtype=float)
        for b in range(bins):
            in_bin = (lengths >= y_edges[b]) & (lengths < y_edges[b + 1])
            if not np.any(in_bin):
                continue
            vecs = [per_pair_vectors[i] for i, ok in enumerate(in_bin) if ok]
            if len(vecs) == 0:
                continue
            stack = np.vstack(vecs)  # (n_pairs_in_bin, nfreq)
            # Complex average across baselines, ignore NaNs per frequency
            with np.errstate(invalid="ignore"):
                avg_complex = np.nanmean(stack, axis=0)
            # Only keep frequencies with finite avg value
            mask_ok = np.isfinite(avg_complex)
            if not np.any(mask_ok):
                continue
            fi = freqs_hz[mask_ok]
            vi = avg_complex[mask_ok]
            phase = -2j * np.pi * fi[:, None] * tau_s[None, :]
            z[b, :] = np.abs(np.sum(vi[:, None] * np.exp(phase), axis=0))
        # Mask DC bin
        if z.shape[1] > 0:
            z[:, 0] = np.nan
    else:
        # Frequency axis in MHz
        y_vals = freqs_hz / 1e6
        y_label = "Frequency [MHz]"
        x_label = "Baseline length [m]"
        z = np.full((bins, nfreq), np.nan, dtype=float)
        for b in range(bins):
            in_bin = (lengths >= y_edges[b]) & (lengths < y_edges[b + 1])
            if not np.any(in_bin):
                continue
            vecs = [per_pair_vectors[i] for i, ok in enumerate(in_bin) if ok]
            if len(vecs) == 0:
                continue
            stack = np.vstack(vecs)  # (n_pairs_in_bin, nfreq)
            with np.errstate(invalid="ignore"):
                avg_complex = np.nanmean(stack, axis=0)
            # Apply quantity mapping
            z[b, :] = _compute_quantity(avg_complex, quantity)

    if debug:
        print(f"  all-baselines mode: pairs={num_pairs}, bins={bins}")
        print(
            "  baseline length range [m]: "
            f"{float(np.nanmin(lengths))} .. {float(np.nanmax(lengths))}"
        )
        counts = [
            int(((lengths >= y_edges[i]) & (lengths < y_edges[i + 1])).sum())
            for i in range(bins)
        ]
        print(f"  counts per bin: {counts}")

    return z, y_vals, y_edges, x_label, y_label, num_pairs


def plot_waterfall(
    uv_path: str,
    *,
    antpair: Optional[Tuple[int, int]] = None,
    antnames: Optional[Tuple[str, str]] = None,
    pol: Optional[str] = None,
    quantity: str = "amp",
    out: Optional[str] = None,
    show: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 150,
    debug: bool = False,
    fft: bool = False,
    all_baselines: bool = False,
    bins: int = 20,
    max_baseline: Optional[float] = None,
) -> None:
    uv = _read_uv(uv_path)

    if all_baselines:
        ant_a, ant_b = None, None  # not used
    elif antnames is not None:
        ant_a, ant_b = _get_antnums_from_names(uv, antnames[0], antnames[1])
    elif antpair is not None:
        ant_a, ant_b = antpair
    else:
        raise ValueError(
            "You must provide either --antpair/--antnames or use --all-baselines."
        )

    plt.style.use("dark_background")

    pol_index = _select_pol_index(uv, pol)
    if all_baselines:
        # Build binned matrix over all baselines
        z_matrix, x_vals, y_edges, x_label, y_label, num_pairs = (
            _all_baselines_binned_matrix(
                uv,
                pol_index,
                quantity=quantity,
                fft=fft,
                bins=bins,
                debug=debug,
                max_baseline_m=max_baseline,
            )
        )
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        vmin_plot = vmin
        vmax_plot = vmax
        use_log_norm = quantity.lower() in ("amp", "amplitude")
        if vmin_plot is None and vmax_plot is None:
            zmin = np.nanmin(z_matrix)
            zmax = np.nanmax(z_matrix)
            if np.isfinite(zmin) and np.isfinite(zmax) and zmin == zmax:
                eps = max(1e-12, abs(zmax) * 1e-6)
                vmin_plot = zmin - eps
                vmax_plot = zmax + eps
        # We want baseline length on X (use y_edges) and delay/frequency on Y (use x_vals)
        x_min = float(np.nanmin(y_edges))
        x_max = float(np.nanmax(y_edges))
        y_min = float(np.nanmin(x_vals))
        y_max = float(np.nanmax(x_vals))
        im = ax.imshow(
            z_matrix.T,
            aspect="auto",
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            norm=LogNorm(vmin=vmin_plot, vmax=vmax_plot) if use_log_norm else None,
            vmin=None if use_log_norm else vmin_plot,
            vmax=None if use_log_norm else vmax_plot,
            cmap="rainbow" if quantity != "phase" else "twilight",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        title_pol = (
            pol
            if pol is not None
            else uvutils.polnum2str([uv.polarization_array[0]])[0]
        )
        fft_tag = " — FFT(freq)" if fft else ""
        max_tag = (
            f" — ≤{int(max_baseline)} m"
            if isinstance(max_baseline, (int, float))
            else ""
        )
        ax.set_title(
            f"All baselines (pairs={num_pairs}{max_tag}) — {title_pol} — {quantity}{fft_tag}"
        )
        cbar_label_map = {
            "amp": "|V|",
            "amplitude": "|V|",
            "real": "Re(V)",
            "imag": "Im(V)",
            "phase": "Phase [deg]",
        }
        cbar_label = cbar_label_map.get(quantity.lower(), quantity)
        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.tight_layout()

        if out is not None:
            plt.savefig(out, dpi=dpi)
            print(f"Saved waterfall plot: {out}")
            plt.close(fig)
        if show and out is None:
            plt.show()
        return

    times_mjd, freqs_hz, wf, flags = _extract_waterfall(uv, ant_a, ant_b, pol_index)
    z = _compute_quantity(wf, quantity)
    # Mask flagged samples (True means flagged/bad) -> set to NaN so not shown
    if flags is not None:
        z = np.where(flags, np.nan, z)
    use_log_norm = quantity.lower() in ("amp", "amplitude")

    if debug:
        np.set_printoptions(precision=6, suppress=True)
        print("Debug info:")
        print(f"  uv_path: {uv_path}")
        print(f"  antpair: {antpair} antnames: {antnames}")
        print(f"  pol: {pol} -> index {pol_index}")
        print(f"  wf shape: {wf.shape}, dtype: {wf.dtype}")
        print(
            f"  times_mjd shape: {times_mjd.shape}, min/max: {times_mjd.min()}/{times_mjd.max()}"
        )
        print(
            f"  freqs_hz shape: {freqs_hz.shape}, min/max: {freqs_hz.min()}/{freqs_hz.max()}"
        )
        zmin = np.nanmin(z) if z.size else np.nan
        zmax = np.nanmax(z) if z.size else np.nan
        zmean = float(np.nanmean(z)) if z.size else float("nan")
        znan = int(np.isnan(z).sum()) if z.size else 0
        print(
            f"  z shape: {z.shape}, dtype: {z.dtype}, min/max/mean: {zmin}/{zmax}/{zmean}, nans: {znan}"
        )
        if flags is not None:
            ftrue = int(flags.sum())
            fcnt = int(flags.size)
            print(
                f"  flags: true={ftrue} of {fcnt} ({(ftrue / max(1, fcnt)) * 100:.2f}%)"
            )
        # print a small sample of values
        sample_rows = min(3, z.shape[0]) if z.ndim == 2 else 0
        sample_cols = min(8, z.shape[1]) if z.ndim == 2 else 0
        if sample_rows and sample_cols:
            print("  z[0:sample_rows, 0:sample_cols]:")
            print(z[0:sample_rows, 0:sample_cols])
        # print selected baseline blt indices and UVW
        ant1 = uv.ant_1_array
        ant2 = uv.ant_2_array
        mask = ((ant1 == ant_a) & (ant2 == ant_b)) | ((ant1 == ant_b) & (ant2 == ant_a))
        idx_all = np.where(mask)[0]
        ord_all = np.argsort(uv.time_array[idx_all])
        idx_sorted = idx_all[ord_all]
        print(
            f"  selected blt indices (count={idx_sorted.size}): {idx_sorted.tolist()}"
        )
        max_lines = 10
        for j, blt in enumerate(idx_sorted[:max_lines]):
            u, v, w = uv.uvw_array[blt]
            t = uv.time_array[blt]
            a1 = int(ant1[blt])
            a2 = int(ant2[blt])
            blen = float(np.sqrt(u * u + v * v + w * w))
            print(
                f"    blt={int(blt)} (ant1={a1}, ant2={a2}) time_mjd={t:.6f} "
                f"uvw_m=({u:.6f}, {v:.6f}, {w:.6f}) |b|={blen:.6f}"
            )
        if idx_sorted.size > max_lines:
            print(f"  ... {idx_sorted.size - max_lines} more rows omitted ...")
        # print overview of unique baselines in the dataset with a sample UVW
        pairs_all = np.stack([ant1, ant2], axis=1)
        uniq_pairs, uniq_idx = np.unique(pairs_all, axis=0, return_index=True)
        order_pairs = np.argsort(uniq_idx)
        uniq_pairs = uniq_pairs[order_pairs]
        uniq_idx = uniq_idx[order_pairs]
        print(f"  unique baselines in dataset: {uniq_pairs.shape[0]}")
        max_pairs = 10
        for k in range(min(max_pairs, uniq_pairs.shape[0])):
            blt0 = int(uniq_idx[k])
            a1 = int(uniq_pairs[k, 0])
            a2 = int(uniq_pairs[k, 1])
            u, v, w = uv.uvw_array[blt0]
            t = uv.time_array[blt0]
            blen = float(np.sqrt(u * u + v * v + w * w))
            print(
                f"    pair[{k}] (ant1={a1}, ant2={a2}) sample blt={blt0} "
                f"time_mjd={t:.6f} uvw_m=({u:.6f}, {v:.6f}, {w:.6f}) |b|={blen:.6f}"
            )
        if uniq_pairs.shape[0] > max_pairs:
            print(f"  ... {uniq_pairs.shape[0] - max_pairs} more baselines omitted ...")

        # show a handful of shortest and longest baselines (by sample length)
        uvw_samples = uv.uvw_array[uniq_idx]
        blens = np.sqrt(
            uvw_samples[:, 0] ** 2 + uvw_samples[:, 1] ** 2 + uvw_samples[:, 2] ** 2
        )
        order_by_len = np.argsort(blens)
        nshow = min(5, uniq_pairs.shape[0])
        print("  shortest baselines (by sample |b|):")
        for k in range(nshow):
            ii = int(order_by_len[k])
            blt0 = int(uniq_idx[ii])
            a1 = int(uniq_pairs[ii, 0])
            a2 = int(uniq_pairs[ii, 1])
            u, v, w = uv.uvw_array[blt0]
            t = uv.time_array[blt0]
            blen = float(blens[ii])
            print(
                f"    (ant1={a1}, ant2={a2}) blt={blt0} time_mjd={t:.6f} "
                f"uvw_m=({u:.6f}, {v:.6f}, {w:.6f}) |b|={blen:.6f}"
            )
        print("  longest baselines (by sample |b|):")
        for k in range(nshow):
            ii = int(order_by_len[-(k + 1)])
            blt0 = int(uniq_idx[ii])
            a1 = int(uniq_pairs[ii, 0])
            a2 = int(uniq_pairs[ii, 1])
            u, v, w = uv.uvw_array[blt0]
            t = uv.time_array[blt0]
            blen = float(blens[ii])
            print(
                f"    (ant1={a1}, ant2={a2}) blt={blt0} time_mjd={t:.6f} "
                f"uvw_m=({u:.6f}, {v:.6f}, {w:.6f}) |b|={blen:.6f}"
            )

    # Build plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    # Avoid vmin==vmax which makes a blank plot
    vmin_plot = vmin
    vmax_plot = vmax
    if vmin_plot is None and vmax_plot is None:
        zmin = np.nanmin(z)
        zmax = np.nanmax(z)
        if np.isfinite(zmin) and np.isfinite(zmax) and zmin == zmax:
            eps = max(1e-12, abs(zmax) * 1e-6)
            vmin_plot = zmin - eps
            vmax_plot = zmax + eps

        # Compute plotting extent with safeguards for single-sample axes
        fx_min_mhz = float(freqs_hz.min() / 1e6)
        fx_max_mhz = float(freqs_hz.max() / 1e6)
        ty_min_mjd = float(times_mjd.min())
        ty_max_mjd = float(times_mjd.max())
        if fx_min_mhz == fx_max_mhz:
            eps_fx = max(1e-6, abs(fx_min_mhz) * 1e-9)
            fx_min_mhz -= eps_fx
            fx_max_mhz += eps_fx
            if debug:
                print(
                    f"  expanded freq extent to [{fx_min_mhz}, {fx_max_mhz}] MHz due to single channel"
                )
        if ty_min_mjd == ty_max_mjd:
            eps_ty = 1e-6  # ~0.0864 seconds in MJD
            ty_min_mjd -= eps_ty
            ty_max_mjd += eps_ty
            if debug:
                print(
                    f"  expanded time extent to [{ty_min_mjd}, {ty_max_mjd}] MJD due to single time sample"
                )

        # Prepare axes: time on X, frequency/delay on Y
        if fft:
            # Build non-negative delay axis (0 .. 4 microseconds)
            nfreq = z.shape[1]
            df = float(np.median(np.diff(freqs_hz)))
            if df > 0 and np.isfinite(df):
                # Decide FFT type: regular rFFT or NUDFT excluding flagged channels
                # Use NUDFT when requested to avoid flagged channels contributing power
                # seconds delay samples (non-negative up to 4 us)
                num_tau = nfreq // 2 + 1
                tau_s = np.linspace(0.0, 4e-6, num=num_tau, endpoint=True)
                # NUDFT per time row using only unflagged channels on true freqs
                z_out = np.zeros((z.shape[0], tau_s.shape[0]), dtype=np.float64)
                for i in range(z.shape[0]):
                    mask_ok = np.isfinite(wf[i, :]) & (~flags[i, :])
                    if not np.any(mask_ok):
                        continue
                    fi = freqs_hz[mask_ok]
                    vi = wf[i, mask_ok]
                    # compute magnitude of sum_j v_j * exp(-i 2pi f_j tau)
                    phase = -2j * np.pi * fi[:, None] * tau_s[None, :]
                    z_out[i, :] = np.abs(np.sum(vi[:, None] * np.exp(phase), axis=0))
                # Mask DC bin (tau==0)
                if tau_s.shape[0] > 0:
                    z_out[:, 0] = np.nan
                z = z_out
                # Plot in microseconds
                y_label = "Delay [µs]"
                y_min = 0.0
                y_max = 4.0
                tau_s = tau_s * 1e6
            else:
                # Fallback if frequency spacing invalid: index-based axis
                z_filled = np.where(np.isfinite(z), z, 0.0)
                z = np.abs(np.fft.rfft(z_filled, axis=1))
                # Mask DC
                if z.shape[1] > 0:
                    z[:, 0] = np.nan
                tau_s = np.arange(z.shape[1], dtype=float)
                y_label = "FFT(freq) index"
                y_min = float(tau_s.min())
                y_max = float(tau_s.max())
            z_plot = z.T
        else:
            y_min = fx_min_mhz
            y_max = fx_max_mhz
            y_label = "Frequency [MHz]"
            z_plot = z.T

        im = ax.imshow(
            z_plot,
            aspect="auto",
            origin="lower",
            extent=[ty_min_mjd, ty_max_mjd, y_min, y_max],
            norm=LogNorm(vmin=vmin_plot, vmax=vmax_plot) if use_log_norm else None,
            vmin=None if use_log_norm else vmin_plot,
            vmax=None if use_log_norm else vmax_plot,
            cmap="rainbow" if quantity != "phase" else "twilight",
        )
        ax.set_xlabel("Time [MJD]")
        ax.set_ylabel(y_label)
        title_pol = (
            pol
            if pol is not None
            else uvutils.polnum2str([uv.polarization_array[0]])[0]
        )
        # Compute representative |uvw| (median over selected rows)
        ant1_all = uv.ant_1_array
        ant2_all = uv.ant_2_array
        mask_sel = ((ant1_all == ant_a) & (ant2_all == ant_b)) | (
            (ant1_all == ant_b) & (ant2_all == ant_a)
        )
        idx_sel = np.where(mask_sel)[0]
        ord_sel = np.argsort(uv.time_array[idx_sel])
        uvw_sel = uv.uvw_array[idx_sel[ord_sel]]
        blen_rep = (
            float(np.nanmedian(np.sqrt((uvw_sel**2).sum(axis=1))))
            if uvw_sel.size
            else float("nan")
        )
        fft_tag = " — FFT(freq)" if fft else ""
        ax.set_title(
            f"Baseline ({ant_a},{ant_b}) — |uvw|≈{blen_rep:.2f} m — {title_pol} — {quantity}{fft_tag}"
        )

    cbar_label = {
        "amp": "|V|",
        "amplitude": "|V|",
        "real": "Re(V)",
        "imag": "Im(V)",
        "phase": "Phase [deg]",
    }.get(quantity.lower(), quantity)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()

    if out is not None:
        plt.savefig(out, dpi=dpi)
        print(f"Saved waterfall plot: {out}")
        plt.close(fig)
    if show and out is None:
        plt.show()


def _tuple_from_csv(x: str) -> Tuple[str, str]:
    parts: Sequence[str] = tuple(p.strip() for p in x.split(","))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Expected two comma-separated values, e.g., '56,23'"
        )
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a baseline waterfall from UVFITS/MS"
    )
    parser.add_argument("uv_path", help="Path to UVFITS or Measurement Set")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--antpair", type=_tuple_from_csv, help="Antenna numbers as 'antA,antB'"
    )
    g.add_argument(
        "--antnames", type=_tuple_from_csv, help="Antenna names as 'nameA,nameB'"
    )
    g.add_argument(
        "--all-baselines",
        action="store_true",
        help="Use all baselines binned by length (time-averaged)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of baseline-length bins for --all-baselines",
    )
    parser.add_argument(
        "--pol", default=None, help="Polarisation (e.g. XX, YY, XY, YX, I, Q, U, V)"
    )
    parser.add_argument(
        "--quantity",
        default="amp",
        choices=["amp", "amplitude", "real", "imag", "phase"],
        help="Quantity to display",
    )
    parser.add_argument("--out", default=None, help="Output image filename (PNG)")
    parser.add_argument(
        "--show", action="store_true", help="Show the plot interactively"
    )
    parser.add_argument("--vmin", type=float, default=None, help="Color scale minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument(
        "--debug", action="store_true", help="Print arrays and stats for debugging"
    )
    parser.add_argument(
        "--fft", action="store_true", help="Apply FFT along frequency axis (per time)"
    )
    parser.add_argument(
        "--max-baseline",
        type=float,
        default=None,
        help="Maximum baseline length in meters to include (all-baselines mode)",
    )

    args = parser.parse_args()

    antpair: Optional[Tuple[int, int]] = None
    antnames: Optional[Tuple[str, str]] = None
    if args.antpair is not None:
        antpair = (int(args.antpair[0]), int(args.antpair[1]))
    if args.antnames is not None:
        antnames = (str(args.antnames[0]), str(args.antnames[1]))

    plot_waterfall(
        args.uv_path,
        antpair=antpair,
        antnames=antnames,
        pol=args.pol,
        quantity=args.quantity,
        out=args.out,
        show=bool(args.show),
        vmin=args.vmin,
        vmax=args.vmax,
        dpi=int(args.dpi),
        debug=bool(args.debug),
        fft=bool(args.fft),
        all_baselines=bool(args.all_baselines),
        bins=int(args.bins),
        max_baseline=args.max_baseline,
    )


if __name__ == "__main__":
    main()
