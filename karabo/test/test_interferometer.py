from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from karabo.imaging.imager_factory import ImagingBackend, get_imager
from karabo.imaging.imager_interface import ImageSpec
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend
from karabo.util.file_handler import FileHandler


def _create_point_source_sky(ra_deg: float, dec_deg: float) -> SkyModel:
    sky = SkyModel()
    source = np.zeros((1, SkyModel.SOURCES_COLS))
    source[0, 0] = ra_deg
    source[0, 1] = dec_deg
    source[0, 2] = 1.0
    sky.add_point_sources(source)
    return sky


def _create_minimal_observation(ra_deg: float, dec_deg: float) -> Observation:
    return Observation(
        start_frequency_hz=1.0e9,
        start_date_and_time=datetime(2020, 1, 1, 0, 0, 0),
        phase_centre_ra_deg=ra_deg,
        phase_centre_dec_deg=dec_deg,
        number_of_time_steps=1,
        frequency_increment_hz=1.0e6,
        number_of_channels=1,
        length=timedelta(seconds=1),
    )


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

    sky = _create_point_source_sky(15.0, -30.0)

    telescope = Telescope.constructor("MID", backend=SimulatorBackend.SDP)
    simulation = InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        time_average_sec=1.0,
        ignore_w_components=True,
        use_gpus=False,
        use_dask=False,
    )
    observation = _create_minimal_observation(15.0, -30.0)

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


@pytest.mark.parametrize(
    "backend,telescope_name",
    [
        (SimulatorBackend.OSKAR, "SKA1MID"),
        (SimulatorBackend.SDP, "ASKAP"),
    ],
)
def test_supported_backends_write_imageable_measurement_sets(
    tmp_path, backend: SimulatorBackend, telescope_name: str
):
    sky = _create_point_source_sky(15.0, -30.0)
    telescope = Telescope.constructor(telescope_name, backend=backend)
    simulation = InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        time_average_sec=1.0,
        ignore_w_components=True,
        use_gpus=False,
        use_dask=False,
    )
    observation = _create_minimal_observation(15.0, -30.0)
    ms_path = tmp_path / f"{backend.name.lower()}.ms"

    visibility = simulation.run_simulation(
        telescope,
        sky,
        observation,
        backend=backend,
        visibility_path=str(ms_path),
    )

    assert visibility.format == "MS"
    assert visibility.path == str(ms_path)
    assert (ms_path / "table.dat").is_file()
    assert (ms_path / "ANTENNA" / "table.dat").is_file()

    imager = get_imager(ImagingBackend.SDP)
    dirty, psf = imager.invert(
        visibility,
        ImageSpec(
            npix=128,
            cellsize_arcsec=20.0,
            phase_centre_deg=(15.0, -30.0),
        ),
    )
    restored = imager.restore(dirty, psf)

    image_centre = (64, 64)
    for image in (dirty, psf, restored):
        data = image.get_squeezed_data()
        assert data.shape == (128, 128)
        assert np.isfinite(data).all()
        assert np.unravel_index(np.nanargmax(data), data.shape) == image_centre
        assert np.isclose(np.nanmax(data), 1.0, rtol=0.02)
        assert np.isclose(image.header["CRVAL1"], 15.0)
        assert np.isclose(image.header["CRVAL2"], -30.0)
        assert image.header["CTYPE1"].startswith("RA")
        assert image.header["CTYPE2"].startswith("DEC")

    model = imager.last_model_image
    residual = imager.last_residual_image
    assert np.isfinite(model.data).all()
    assert np.isfinite(residual.data).all()
    assert np.nanmax(np.abs(residual.data)) < 0.02 * np.nanmax(np.abs(dirty.data))
