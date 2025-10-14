from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend
from karabo.util.file_handler import FileHandler


def test_sdp_simulation_matches_rascil(monkeypatch, tmp_path):
    # Redirect temp roots
    monkeypatch.setattr(FileHandler, "root_stm", str(tmp_path), raising=False)
    monkeypatch.setattr(FileHandler, "root_ltm", str(tmp_path), raising=False)

    # Capture in-memory visibilities instead of writing MS
    exported = []

    def fake_export(ms_path, vis_list, source_name=None):
        Path(ms_path).mkdir(parents=True, exist_ok=True)
        # store deep copies to avoid later mutation
        exported.append([vis.copy(deep=True) for vis in vis_list])

    monkeypatch.setattr(
        "karabo.util.ska_sdp_datamodels.visibility.vis_io_ms.export_visibility_to_ms",
        fake_export,
        raising=False,
    )
    monkeypatch.setattr(
        "karabo.simulation.interferometer.export_visibility_to_ms",
        fake_export,
        raising=False,
    )

    # Minimal sky: single 1 Jy point at phase centre
    sky = SkyModel()
    arr = np.zeros((1, SkyModel.SOURCES_COLS))
    arr[0, 0] = 15.0  # RA deg
    arr[0, 1] = -30.0  # DEC deg
    arr[0, 2] = 1.0  # Stokes I Jy
    sky.add_point_sources(arr)

    # Telescope via RASCIL/SDP config (same config object underneath)
    tel = Telescope.constructor("MID", backend=SimulatorBackend.RASCIL)

    # Obs: 1 time, 1 chan, centred at phase centre
    sim = InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        time_average_sec=1.0,
        ignore_w_components=True,
        use_gpus=False,
        use_dask=False,
    )
    obs = Observation(
        start_frequency_hz=1.0e9,
        start_date_and_time=datetime(2020, 1, 1, 0, 0, 0),
        phase_centre_ra_deg=15.0,
        phase_centre_dec_deg=-30.0,
        number_of_time_steps=1,
        frequency_increment_hz=1.0e6,
        number_of_channels=1,
        length=timedelta(seconds=1),
    )

    rascil_ms = tmp_path / "rascil.ms"
    sdp_ms = tmp_path / "sdp.ms"

    v_rascil = sim.run_simulation(
        tel, sky, obs, backend=SimulatorBackend.RASCIL, visibility_path=str(rascil_ms)
    )
    v_sdp = sim.run_simulation(
        tel, sky, obs, backend=SimulatorBackend.SDP, visibility_path=str(sdp_ms)
    )

    # Basic assertions on API
    assert v_rascil.format == "MS"
    assert v_sdp.format == "MS"
    assert v_rascil.path == str(rascil_ms)
    assert v_sdp.path == str(sdp_ms)

    # We expect two export calls, each with a single xarray.Dataset
    assert len(exported) == 2
    assert all(len(entry) == 1 for entry in exported)
    ds_r = exported[0][0]
    ds_s = exported[1][0]

    # Helper to compare datasets with tolerance and ignore attr noise
    def _assert_ds_close(a: xr.Dataset, b: xr.Dataset, rtol=1e-6, atol=1e-9):
        # Same variable names and shapes
        assert set(a.data_vars) == set(b.data_vars)
        for name in a.data_vars:
            va, vb = a[name], b[name]
            assert va.shape == vb.shape
            if np.issubdtype(va.dtype, np.floating):
                np.testing.assert_allclose(va.values, vb.values, rtol=rtol, atol=atol)
            else:
                assert np.array_equal(va.values, vb.values)
        # Coordinates (times, frequency, baselines) with tolerance
        for cname in set(a.coords) & set(b.coords):
            ca, cb = a[cname], b[cname]
            assert ca.shape == cb.shape
            if np.issubdtype(ca.dtype, np.floating):
                np.testing.assert_allclose(ca.values, cb.values, rtol=rtol, atol=atol)
            else:
                assert np.array_equal(ca.values, cb.values)

    _assert_ds_close(ds_r, ds_s)
