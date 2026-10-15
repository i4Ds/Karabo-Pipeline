"""MS migration regressions on the real SDP/casacore implementations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from casacore.tables import makearrcoldesc, maketabdesc, table
from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.science_data_model import PolarisationFrame, ReceptorFrame
from ska_sdp_datamodels.visibility import Visibility as SdpVisibility
from ska_sdp_datamodels.visibility import (
    create_visibility_from_ms,
    export_visibility_to_ms,
)

from karabo.imaging.backends.sdp_backend import SdpImager
from karabo.simulation import sdp_ms
from karabo.simulation.visibility import Visibility


def _visibility(frame="linear"):
    config = Configuration.constructor(
        name="audit",
        location=EarthLocation.from_geocentric(5100000, 2000000, -3200000, unit="m"),
        names=["a", "b"],
        xyz=np.array(
            [[5100000.0, 2000000.0, -3200000.0], [5101000.0, 2000200.0, -3200000.0]]
        ),
        mount=["alt-az", "equatorial"],
        frame="ITRF",
        receptor_frame=ReceptorFrame("linear"),
        diameter=[15.0, 12.0],
        offset=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        stations=["station-a", "station-b"],
    )
    pol = PolarisationFrame(frame)
    shape = (2, 3, 2, pol.npol)
    values = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1
    flags = np.zeros(shape, dtype=int)
    flags[0, 1, 0, 0] = 1
    return SdpVisibility.constructor(
        uvw=np.arange(18.0, dtype=float).reshape(2, 3, 3),
        baselines=pd.MultiIndex.from_tuples(
            [(0, 0), (0, 1), (1, 1)], names=["antenna1", "antenna2"]
        ),
        time=np.array([5e9, 5e9 + 20]),
        frequency=np.array([1e8, 1.01e8]),
        channel_bandwidth=np.array([1e6, 1e6]),
        vis=values + 1j * values / 10,
        flags=flags,
        weight=values / 4,
        integration_time=np.array([10.0, 12.0]),
        configuration=config,
        phasecentre=SkyCoord(15.0, -30.0, unit="deg"),
        polarisation_frame=pol,
        source="audit",
    )


@pytest.mark.parametrize(
    "frame",
    [
        "stokesI",
        "linear",
        "linearFITS",
        "circular",
        "linearnp",
        "circularnp",
        "stokesIQUV",
        "stokesIQ",
        "stokesIV",
    ],
)
def test_ms_roundtrip(frame, tmp_path, monkeypatch):
    original = _visibility(frame)
    before = original.copy(deep=True)
    path = tmp_path / "roundtrip.ms"
    calls = []

    def exporter(*args, **kwargs):
        calls.append("export")
        return export_visibility_to_ms(*args, **kwargs)

    def importer(*args, **kwargs):
        calls.append("import")
        return create_visibility_from_ms(*args, **kwargs)

    monkeypatch.setattr(sdp_ms, "_export_visibility_to_ms", exporter)
    monkeypatch.setattr(sdp_ms, "_create_visibility_from_ms", importer)
    sdp_ms.export_sdp_visibility(path, original)
    actual = sdp_ms.import_sdp_visibility(path)
    assert calls == ["export", "import"]
    order = (
        [0, 2, 3, 1] if frame == "linearFITS" else list(range(original.vis.shape[-1]))
    )
    for name in ("vis", "flags", "weight"):
        np.testing.assert_allclose(
            actual[name].data, original[name].data[..., order], rtol=1e-7, atol=1e-7
        )
        np.testing.assert_array_equal(original[name].data, before[name].data)
    for name in ("time", "uvw", "frequency", "channel_bandwidth", "integration_time"):
        np.testing.assert_allclose(
            actual[name].data, original[name].data, rtol=0, atol=1e-6
        )
    np.testing.assert_array_equal(actual.datetime, original.datetime)
    np.testing.assert_array_equal(actual.visibility_acc.imaging_weight, actual.weight)
    for name in ("names", "xyz", "mount", "diameter", "offset", "stations"):
        np.testing.assert_array_equal(
            actual.configuration[name], original.configuration[name]
        )
    assert actual.source == original.source
    assert actual.phasecentre.separation(original.phasecentre).arcsec < 1e-8
    assert actual.visibility_acc.polarisation_frame.type == (
        "linear" if frame == "linearFITS" else frame
    )
    with table(str(path), ack=False) as ms:
        times = np.repeat(original.time.data, 3)
        np.testing.assert_array_equal(ms.getcol("TIME"), times)
        np.testing.assert_array_equal(ms.getcol("TIME_CENTROID"), times)
        np.testing.assert_allclose(ms.getcol("UVW"), -original.uvw.data.reshape(-1, 3))
        np.testing.assert_allclose(
            ms.getcol("WEIGHT"), ms.getcol("WEIGHT_SPECTRUM").mean(axis=1)
        )
        np.testing.assert_allclose(ms.getcol("SIGMA"), 1 / np.sqrt(ms.getcol("WEIGHT")))
    with table(str(path / "OBSERVATION"), ack=False) as obs:
        np.testing.assert_array_equal(obs.getcell("TIME_RANGE", 0), [5e9 - 5, 5e9 + 26])
    for subtable in ("FIELD", "SOURCE"):
        with table(str(path / subtable), ack=False) as sub:
            np.testing.assert_array_equal(sub.getcol("TIME"), [5e9 + 10.5])


def _permute_correlations(path, order):
    with table(str(path / "POLARIZATION"), readonly=False, ack=False) as pol:
        pol.putcell("CORR_TYPE", 0, pol.getcell("CORR_TYPE", 0)[order])
        pol.putcell("CORR_PRODUCT", 0, pol.getcell("CORR_PRODUCT", 0)[order])
    with table(str(path), readonly=False, ack=False) as ms:
        for column in ("DATA", "FLAG", "WEIGHT_SPECTRUM", "WEIGHT", "SIGMA"):
            ms.putcol(column, ms.getcol(column)[..., order])


@pytest.mark.parametrize("frame", ["linear", "linearFITS", "circular"])
@pytest.mark.parametrize("reverse", [False, True])
def test_metadata_order_and_reversed_crosshands(frame, reverse, tmp_path):
    original = _visibility("circular" if frame == "circular" else "linear")
    original.flags.data[1, 1, 0, 1] = 1  # Distinguish XY/RL from YX/LR flags.
    path = tmp_path / "polarized.ms"
    sdp_ms.export_sdp_visibility(path, original)
    if frame == "linearFITS":
        _permute_correlations(path, [0, 3, 1, 2])
    with table(str(path), readonly=False, ack=False) as ms:
        if reverse:
            rows = ms.getcol("ANTENNA1") != ms.getcol("ANTENNA2")
            a1, a2 = ms.getcol("ANTENNA1"), ms.getcol("ANTENNA2")
            ms.putcol("ANTENNA1", a2)
            ms.putcol("ANTENNA2", a1)
            swap = [0, 1, 3, 2] if frame == "linearFITS" else [0, 2, 1, 3]
            for name in ("DATA", "FLAG", "WEIGHT_SPECTRUM", "WEIGHT", "SIGMA"):
                data = ms.getcol(name)
                data[rows] = data[rows][..., swap]
                if name == "DATA":
                    data[rows] = data[rows].conj()
                ms.putcol(name, data)
            uvw = ms.getcol("UVW")
            uvw[rows] *= -1
            ms.putcol("UVW", uvw)
        before = {
            name: ms.getcol(name)
            for name in ("DATA", "FLAG", "UVW", "TIME", "WEIGHT_SPECTRUM")
        }
    actual = sdp_ms.import_sdp_visibility(path)
    for name in ("vis", "flags", "weight", "uvw"):
        np.testing.assert_allclose(actual[name], original[name], rtol=1e-7, atol=1e-7)
    # This is the path used by the imager, including autocorrelation flagging.
    stokes = SdpImager()._load_visibility(Visibility(str(path)))
    np.testing.assert_allclose(
        stokes.vis.data[1, 1, :, 0],
        (original.vis.data[1, 1, :, 0] + original.vis.data[1, 1, :, 3]) / 2,
        rtol=1e-7,
    )
    assert np.all(stokes.flags.data[:, [0, 2]] == 1)
    assert stokes.flags.data[0, 1, 0, 0] == 1
    with table(str(path), ack=False) as ms:
        for name, expected in before.items():
            np.testing.assert_array_equal(ms.getcol(name), expected)


@pytest.mark.parametrize("spectral_weights", [True, False])
def test_external_row_flags_and_weights(tmp_path, spectral_weights):
    path = tmp_path / "external.ms"
    original = _visibility("stokesI")
    sdp_ms.export_sdp_visibility(path, original)
    with table(str(path), readonly=False, ack=False) as ms:
        row_flags = ms.getcol("FLAG_ROW")
        row_flags[4] = True
        ms.putcol("FLAG_ROW", row_flags)
        if not spectral_weights:
            expected = np.broadcast_to(
                ms.getcol("WEIGHT")[:, None, :], (6, 2, 1)
            ).copy()
            ms.removecols("WEIGHT_SPECTRUM")
        else:
            expected = ms.getcol("WEIGHT_SPECTRUM")
    actual = sdp_ms.import_sdp_visibility(path)
    np.testing.assert_array_equal(actual.weight.data.reshape(6, 2, 1), expected)
    assert np.all(actual.flags.data[1, 1] == 1)


@pytest.mark.parametrize("defined_rows", [[], [1, 4]])
def test_partly_undefined_spectral_weights(tmp_path, defined_rows):
    path = tmp_path / "undefined.ms"
    sdp_ms.export_sdp_visibility(path, _visibility("stokesI"))
    with table(str(path), readonly=False, ack=False) as ms:
        spectrum = ms.getcol("WEIGHT_SPECTRUM")
        expected = np.broadcast_to(
            ms.getcol("WEIGHT")[:, None, :], spectrum.shape
        ).copy()
        ms.removecols("WEIGHT_SPECTRUM")
        ms.addcols(maketabdesc(makearrcoldesc("WEIGHT_SPECTRUM", 0.0, ndim=2)))
        for row in defined_rows:
            ms.putcell("WEIGHT_SPECTRUM", row, spectrum[row])
            expected[row] = spectrum[row]
    actual = sdp_ms.import_sdp_visibility(path)
    np.testing.assert_array_equal(actual.weight.data.reshape(6, 2, 1), expected)


def test_unnamed_antennas_and_repeated_read(tmp_path):
    path = tmp_path / "unnamed.ms"
    original = _visibility("stokesI")
    sdp_ms.export_sdp_visibility(path, original)
    with table(str(path / "ANTENNA"), readonly=False, ack=False) as antennas:
        antennas.putcol("NAME", ["", ""])
    for _ in range(3):
        loaded = sdp_ms.import_sdp_visibility(path)
        np.testing.assert_array_equal(loaded.configuration.names, ["", ""])
        np.testing.assert_array_equal(
            loaded.configuration.xyz, original.configuration.xyz
        )
        np.testing.assert_allclose(loaded.vis, original.vis, rtol=1e-7)
    with table(str(path), readonly=False, ack=False) as ms:
        ms.lock(write=True, nattempts=1)  # Raises on failure; returns None on success.
        assert ms.haslock(write=True)
        ms.unlock()


def test_non_utc_epochs_are_not_silently_relabelled(tmp_path):
    path = tmp_path / "tai.ms"
    sdp_ms.export_sdp_visibility(path, _visibility("stokesI"))
    with table(str(path), readonly=False, ack=False) as ms:
        info = ms.getcolkeyword("TIME", "MEASINFO")
        info["Ref"] = "TAI"
        ms.putcolkeyword("TIME", "MEASINFO", info)
    with pytest.raises(NotImplementedError, match="UTC"):
        sdp_ms.import_sdp_visibility(path)


@pytest.mark.parametrize("case", ["multi", "empty", "unsorted", "duplicate"])
def test_unsupported_ms_fails_explicitly(tmp_path, case):
    path = tmp_path / "unsupported.ms"
    sdp_ms.export_sdp_visibility(path, _visibility())
    with table(str(path), readonly=False, ack=False) as ms:
        if case == "multi":
            field = ms.getcol("FIELD_ID")
            field[-1] = 1  # Reject before upstream encounters sparse field/DD pairs.
            ms.putcol("FIELD_ID", field)
        elif case == "empty":
            path = tmp_path / "empty.ms"
            with ms.query("FIELD_ID < 0") as empty:
                empty.copy(str(path), deep=True).close()
        elif case == "unsorted":
            ms.putcol("TIME", ms.getcol("TIME")[::-1])
        else:
            ms.putcell("ANTENNA1", 1, 0)
            ms.putcell("ANTENNA2", 1, 0)
    with pytest.raises(
        (ValueError, NotImplementedError),
        match="single visibility|time-sorted|duplicate",
    ):
        sdp_ms.import_sdp_visibility(path)


@pytest.mark.parametrize("frame", ["stokesI", "linear"])
def test_against_archived_legacy_io(frame, tmp_path):
    # Captured by running the removed implementation, not generated by the new
    # wrapper. See the adjacent JSON provenance; no vendored reader is retained.
    golden = json.loads(
        (Path(__file__).parent / "data" / "legacy_sdp_ms.json").read_text()
    )[frame]
    original = _visibility(frame)
    path = tmp_path / "new.ms"
    sdp_ms.export_sdp_visibility(path, original)
    new = sdp_ms.import_sdp_visibility(path)
    for name in ("uvw", "frequency", "channel_bandwidth", "integration_time"):
        np.testing.assert_allclose(new[name], golden[name], rtol=0, atol=1e-6)
    expected_vis = np.array(golden["vis_real"]) + 1j * np.array(golden["vis_imag"])
    np.testing.assert_array_equal(new.vis, expected_vis)
    # Legacy writer + reader shifts cancelled; the official pair keeps the same
    # physical sample epochs, but writes correct midpoints rather than starts+dt/2.
    np.testing.assert_array_equal(new.time, golden["time"])
    with table(str(path), ack=False) as ms:
        np.testing.assert_array_equal(
            ms.getcol("TIME") + ms.getcol("INTERVAL") / 2, golden["ms_time"]
        )
    assert np.count_nonzero(golden["flags"]) == 0
    np.testing.assert_array_equal(new.flags, original.flags)
    assert np.all(np.array(golden["weight"]) == 1)
    np.testing.assert_array_equal(new.weight, original.weight)
    # Reading an existing (legacy-produced) MS must trust its recorded midpoints,
    # not guess provenance and silently subtract half an interval.
    with table(str(path), readonly=False, ack=False) as ms:
        ms.putcol("TIME", np.array(golden["ms_time"]))
        ms.putcol("TIME_CENTROID", np.array(golden["ms_time"]))
    reread = sdp_ms.import_sdp_visibility(path)
    np.testing.assert_array_equal(
        reread.time, original.time + original.integration_time / 2
    )
    np.testing.assert_array_equal(
        reread.datetime, Time(reread.time.data / 86400, format="mjd").datetime64
    )
