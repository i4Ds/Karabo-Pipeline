import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from casacore.tables import table

from karabo.imaging.imager_factory import (
    ImagingBackend,
    SdpImagerConfig,
    WscleanBackendConfig,
    get_imager,
)
from karabo.imaging.imager_interface import ImageSpec
from karabo.simulation import sdp_ms
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

    def fake_export(ms_path, vis):
        Path(ms_path).mkdir(parents=True, exist_ok=True)
        exported.append([vis.copy(deep=True)])

    monkeypatch.setattr(
        "karabo.simulation.interferometer.export_sdp_visibility",
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


@pytest.mark.parametrize(
    "imaging_backend", [ImagingBackend.SDP, ImagingBackend.WSCLEAN]
)
def test_sdp_ms_preserves_generated_timing_and_metadata(
    tmp_path, monkeypatch, imaging_backend
):
    """Real DFT/export/import: retain the sample epochs associated with the UVW.

    The requested UTC start selects a nearby transit in the existing simulator;
    changing that scheduling model is not part of the MS migration.
    """
    if imaging_backend is ImagingBackend.WSCLEAN and shutil.which("wsclean") is None:
        pytest.skip("WSClean executable is required for this integration test")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    captured = []
    official_export = sdp_ms._export_visibility_to_ms

    def record_export(path, vis_list):
        captured.append(vis_list[0].copy(deep=True))
        return official_export(path, vis_list)

    monkeypatch.setattr(sdp_ms, "_export_visibility_to_ms", record_export)
    telescope = Telescope.constructor("ASKAP", backend=SimulatorBackend.SDP)
    sky = _create_point_source_sky(15.0, -30.0)
    obs = _create_minimal_observation(15.0, -30.0)
    obs.number_of_time_steps = 3
    obs.length = timedelta(seconds=30)
    obs.number_of_channels = 2
    simulation = InterferometerSimulation(
        channel_bandwidth_hz=1e6,
        time_average_sec=10,
        ignore_w_components=False,
        use_gpus=False,
        use_dask=False,
    )
    path = tmp_path / "timing.ms"
    result = simulation.run_simulation(
        telescope, sky, obs, backend=SimulatorBackend.SDP, visibility_path=str(path)
    )
    assert len(captured) == 1
    generated = captured[0]
    loaded = sdp_ms.import_sdp_visibility(path)
    for name in (
        "time",
        "datetime",
        "integration_time",
        "frequency",
        "channel_bandwidth",
        "flags",
    ):
        np.testing.assert_array_equal(loaded[name], generated[name])
    for name in ("vis", "uvw", "weight"):
        np.testing.assert_allclose(loaded[name], generated[name], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(
        loaded.configuration.names, generated.configuration.names
    )
    with table(str(path), ack=False) as ms:
        np.testing.assert_array_equal(np.unique(ms.getcol("TIME")), generated.time.data)
        np.testing.assert_array_equal(ms.getcol("TIME_CENTROID"), ms.getcol("TIME"))
    # A continuum point source remains at the phase centre with unit Jy peak.
    imager = get_imager(
        imaging_backend,
        config=(
            SdpImagerConfig(combine_across_frequencies=False)
            if imaging_backend is ImagingBackend.SDP
            else WscleanBackendConfig(weighting="natural")
        ),
    )
    dirty, psf = imager.invert(
        result,
        ImageSpec(npix=128, cellsize_arcsec=20.0, phase_centre_deg=(15.0, -30.0)),
    )
    restored = imager.restore(dirty, psf)
    for image in (dirty, psf, restored):
        assert np.isfinite(image.data).all()
        for plane in image.data[:, 0]:
            assert np.unravel_index(np.argmax(plane), plane.shape) == (64, 64)
            assert np.max(plane) == pytest.approx(1.0, rel=0.02)
