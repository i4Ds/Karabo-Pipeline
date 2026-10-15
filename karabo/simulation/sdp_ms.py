"""SDP 1.4.1 MS I/O, with narrow corrections at the Karabo boundary.

SDP owns the MS/data-model conversion. The corrections below preserve spectral
weights, row flags and correlation semantics that its minimal MS reader omits.
Input Measurement Sets are never modified. TIME is always an interval midpoint.
"""

from pathlib import Path
from typing import Union

import numpy as np
from casacore.tables import table
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility import Visibility
from ska_sdp_datamodels.visibility import (
    create_visibility_from_ms as _create_visibility_from_ms,
)
from ska_sdp_datamodels.visibility import (
    export_visibility_to_ms as _export_visibility_to_ms,
)


def export_sdp_visibility(ms_path: Union[str, Path], vis: Visibility) -> None:
    """Export one simulated field/window using SDP, retaining sample midpoints.

    SDP samples and UVW are evaluated at ``vis.time``; adding half an integration
    would misdate those samples. The existing transit/hour-angle schedule is not
    changed here. Flags and channel-dependent weights go through the official
    exporter. Its ancillary column defaults are completed below.
    """
    path = str(ms_path)
    _export_visibility_to_ms(path, [vis])
    with table(path, readonly=False, ack=False) as ms:
        # SDP writes the true weights to WEIGHT_SPECTRUM, but unit WEIGHT and
        # placeholder SIGMA. Keep the channel-averaged row columns consistent.
        weight = ms.getcol("WEIGHT_SPECTRUM").mean(axis=1)
        sigma = np.full_like(weight, np.inf)
        np.divide(1.0, np.sqrt(weight), out=sigma, where=weight > 0)
        ms.putcol("WEIGHT", weight)
        ms.putcol("SIGMA", sigma)
        times, intervals = ms.getcol("TIME"), ms.getcol("INTERVAL")
        start = float(np.min(times - intervals / 2))
        end = float(np.max(times + intervals / 2))
    with table(path + "/OBSERVATION", readonly=False, ack=False) as obs:
        obs.putcell("TIME_RANGE", 0, [start, end])
    # The upstream writer zeroes offsets even when the configuration has them.
    with table(path + "/ANTENNA", readonly=False, ack=False) as antennas:
        antennas.putcol("OFFSET", vis.configuration.offset.data)
    # SDP 1.4.1 multiplies these already-second-valued epochs by 86400. These
    # simulated directions have NUM_POLY=0; the origin is the observation centre.
    for subtable in ("FIELD", "SOURCE"):
        with table(path + "/" + subtable, readonly=False, ack=False) as sub:
            sub.putcol("TIME", np.full(sub.nrows(), (start + end) / 2))


def import_sdp_visibility(ms_path: Union[str, Path]) -> Visibility:
    """Read a single field/DD with official SDP I/O and correct MS semantics.

    Preserve per-channel weights and FLAG_ROW, and normalize linearFITS to the
    linear frame understood by processing-functions 1.0.0. Reversed polarized
    baselines need cross-hand exchange in addition to SDP's conjugation/UVW flip.
    Multiple populated blocks and ambiguous duplicate samples remain unsupported.
    """
    path = str(ms_path)
    with table(path, ack=False) as ms:
        pairs = np.unique(
            np.column_stack((ms.getcol("FIELD_ID"), ms.getcol("DATA_DESC_ID"))),
            axis=0,
        )
        if len(pairs) != 1:
            raise NotImplementedError(
                "SdpImager currently supports a single visibility per call."
            )
        times = ms.getcol("TIME")
        time_reference = ms.getcolkeywords("TIME").get("MEASINFO", {}).get("Ref", "UTC")
        if time_reference != "UTC":
            raise NotImplementedError("SDP MS import requires UTC TIME epochs.")
        if np.any(np.diff(times) < 0):
            raise ValueError("MS rows must be time-sorted for SDP imaging.")
        a1, a2 = ms.getcol("ANTENNA1"), ms.getcol("ANTENNA2")
        flags = ms.getcol("FLAG") | ms.getcol("FLAG_ROW")[:, None, None]
        # Some external MSs declare WEIGHT_SPECTRUM but leave rows undefined.
        has_spectrum = "WEIGHT_SPECTRUM" in ms.colnames()
        defined = np.array(
            [
                has_spectrum and ms.iscelldefined("WEIGHT_SPECTRUM", row)
                for row in range(ms.nrows())
            ]
        )
        if np.all(defined):
            weights = ms.getcol("WEIGHT_SPECTRUM")
        else:
            weights = np.broadcast_to(
                ms.getcol("WEIGHT")[:, None, :], flags.shape
            ).copy()
            for row in np.flatnonzero(defined):
                weights[row] = ms.getcell("WEIGHT_SPECTRUM", int(row))
        if weights.shape != flags.shape:
            raise ValueError("MS WEIGHT_SPECTRUM shape does not match FLAG/DATA.")

    with table(path + "/ANTENNA", ack=False) as antennas:
        names = np.asarray(antennas.getcol("NAME"))
    valid = names != ""
    if not np.any(valid):
        valid[:] = True
    antenna_map = np.full(len(names), -1, dtype=int)
    antenna_map[valid] = np.arange(np.count_nonzero(valid))
    if (
        np.any(a1 < 0)
        or np.any(a2 < 0)
        or np.any(a1 >= len(names))
        or np.any(a2 >= len(names))
    ):
        raise ValueError("MS references an invalid antenna index.")
    a1, a2 = antenna_map[a1], antenna_map[a2]
    if np.any(a1 < 0) or np.any(a2 < 0):
        raise ValueError("MS references unnamed antennas that SDP would discard.")

    block = _create_visibility_from_ms(path)[0]
    ti = np.searchsorted(block.time.data, times)
    if np.any(ti >= len(block.time)) or not np.array_equal(block.time.data[ti], times):
        raise ValueError("SDP merged distinct MS sample times; cannot import safely.")
    bi = block.indexes["baselines"].get_indexer(
        list(zip(np.minimum(a1, a2), np.maximum(a1, a2)))
    )
    if np.any(bi < 0) or len(np.unique(ti * len(block.baselines) + bi)) != len(ti):
        raise ValueError("MS contains duplicate or unsupported time/baseline samples.")

    # Missing baseline slots have zero weight; mark them flagged as well.
    block["flags"].data[:] = 1
    block["flags"].data[ti, bi] = flags
    block["weight"].data[ti, bi] = weights
    frame = block.visibility_acc.polarisation_frame.type
    reversed_rows = a1 > a2
    swaps = {
        "linear": [0, 2, 1, 3],
        "circular": [0, 2, 1, 3],
        "linearFITS": [0, 1, 3, 2],
    }
    if frame in swaps and np.any(reversed_rows):
        rt, rb = ti[reversed_rows], bi[reversed_rows]
        for name in ("vis", "flags", "weight"):
            block[name].data[rt, rb] = block[name].data[rt, rb][..., swaps[frame]]

    if frame == "linearFITS":
        # CORR_TYPE is XX,YY,XY,YX; processing-functions expects XX,XY,YX,YY.
        block = block.isel(polarisation=[0, 2, 3, 1]).assign_coords(
            polarisation=PolarisationFrame("linear").names
        )
        block.attrs["_polarisation_frame"] = "linear"
    # Materialize imaging weights only after all permutations/corrections.
    block.visibility_acc.imaging_weight = block.weight.data.copy()
    return block
