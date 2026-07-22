from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend
from karabo.util.file_handler import FileHandler


def test_sdp_simulation_exports_finite_visibility(monkeypatch, tmp_path):
    monkeypatch.setattr(FileHandler, "root_stm", str(tmp_path), raising=False)
    monkeypatch.setattr(FileHandler, "root_ltm", str(tmp_path), raising=False)

    exported = []

    def fake_export(ms_path, vis_list, source_name=None):
        Path(ms_path).mkdir(parents=True, exist_ok=True)
        exported.append([vis.copy(deep=True) for vis in vis_list])

    monkeypatch.setattr(
        "karabo.simulation.interferometer.export_visibility_to_ms",
        fake_export,
        raising=False,
    )

    sky = SkyModel()
    source = np.zeros((1, SkyModel.SOURCES_COLS))
    source[0, 0] = 15.0
    source[0, 1] = -30.0
    source[0, 2] = 1.0
    sky.add_point_sources(source)

    telescope = Telescope.constructor("MID", backend=SimulatorBackend.SDP)
    simulation = InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        time_average_sec=1.0,
        ignore_w_components=True,
        use_gpus=False,
        use_dask=False,
    )
    observation = Observation(
        start_frequency_hz=1.0e9,
        start_date_and_time=datetime(2020, 1, 1, 0, 0, 0),
        phase_centre_ra_deg=15.0,
        phase_centre_dec_deg=-30.0,
        number_of_time_steps=1,
        frequency_increment_hz=1.0e6,
        number_of_channels=1,
        length=timedelta(seconds=1),
    )

    ms_path = tmp_path / "sdp.ms"
    visibility = simulation.run_simulation(
        telescope,
        sky,
        observation,
        backend=SimulatorBackend.SDP,
        visibility_path=str(ms_path),
    )

    assert visibility.format == "MS"
    assert visibility.path == str(ms_path)
    assert len(exported) == 1
    assert len(exported[0]) == 1

    dataset = exported[0][0]
    assert "vis" in dataset.data_vars
    assert dataset["vis"].size > 0
    assert np.isfinite(dataset["vis"].values).all()
    assert np.any(np.abs(dataset["vis"].values) > 0.0)
