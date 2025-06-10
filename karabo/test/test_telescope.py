import os
import pathlib as pl
import tempfile
from unittest import mock

import numpy as np
import pytest
from astropy import constants as const
from ska_sdp_datamodels.configuration.config_model import Configuration

from karabo.simulation.telescope import Telescope
from karabo.simulation.telescope_versions import (
    ACAVersions,
    ALMAVersions,
    CARMAVersions,
    NGVLAVersions,
    PDBIVersions,
    SMAVersions,
    VLAVersions,
)
from karabo.simulator_backend import SimulatorBackend


@pytest.mark.parametrize("filename", ["test_telescope.tm"])
def test_write_and_read_tm_file(filename):
    BACKEND = SimulatorBackend.OSKAR
    tel = Telescope.constructor("EXAMPLE", backend=BACKEND)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_filename = os.path.join(tmpdir, filename)
        tel.write_to_disk(tmp_filename)
        assert pl.Path(tmp_filename).resolve().exists()
        new_tel = Telescope.read_OSKAR_tm_file(tmp_filename)
        assert len(new_tel.stations) == 30


def test_convert_to_oskar():
    tel = Telescope.constructor("EXAMPLE")
    oskar_tel = tel.get_OSKAR_telescope()
    assert oskar_tel.get_num_stations() == 30


def test_invalid_OSKAR_telescope():
    with pytest.raises(TypeError):
        Telescope.constructor("FAKETELESCOPE")


def test_OSKAR_telescope_with_missing_version():
    # ALMA requires a version
    with pytest.raises(AssertionError):
        Telescope.constructor("ALMA", version=None)


def test_OSKAR_telescope_with_invalid_version():
    # Use NGVLA version for ALMA telescope
    with pytest.raises(AssertionError):
        Telescope.constructor("ALMA", version=NGVLAVersions.CORE_rev_B)


def test_OSKAR_telescope_with_version_but_version_not_required():
    # MeerKAT does not require a version
    with pytest.raises(AssertionError):
        Telescope.constructor("MeerKAT", version="Not None version")


def test_OSKAR_telescope_plot_file_created():
    with tempfile.TemporaryDirectory() as tmpfile:
        temp_plot_file_name = os.path.join(tmpfile, "test-plot.png")
        tel = Telescope.constructor("MeerKAT")
        tel.plot_telescope(temp_plot_file_name)
        assert os.path.exists(temp_plot_file_name)
        # It is tedious to check a specific file size. Even
        # small changes to the code creating the image will make
        # this test fail. Thus, I check only if the file size
        # is > 0.
        assert os.path.getsize(temp_plot_file_name) > 0


def test_create_alma_telescope():
    tel = Telescope.constructor("ALMA", ALMAVersions.CYCLE_1_1)
    assert len(tel.stations) == 32


def test_create_meerkat_telescope():
    tel = Telescope.constructor("MeerKAT")
    assert len(tel.stations) == 64


@pytest.mark.parametrize("version", ALMAVersions)
def test_read_all_ALMA_versions(version):
    try:
        _ = Telescope.constructor("ALMA", version)
    except FileNotFoundError:
        pytest.fail(f"Cannot create ALMA telescope with version {version}")


@pytest.mark.parametrize("version", ACAVersions)
def test_read_all_ACA_versions(version):
    try:
        _ = Telescope.constructor("ACA", version)
    except FileNotFoundError:
        pytest.fail(f"Cannot create ALMA telescope with version {version}")


@pytest.mark.parametrize("version", CARMAVersions)
def test_read_all_CARMA_versions(version):
    try:
        _ = Telescope.constructor("CARMA", version)
    except FileNotFoundError:
        pytest.fail(f"Cannot create CARMA telescope with version {version}")


@pytest.mark.parametrize("version", NGVLAVersions)
def test_read_all_NG_VLA_versions(version):
    try:
        _ = Telescope.constructor("NGVLA", version)
    except FileNotFoundError:
        pytest.fail(f"Cannot create NGVLA telescope with version {version}")


@pytest.mark.parametrize("version", PDBIVersions)
def test_read_all_PDBI_versions(version):
    try:
        _ = Telescope.constructor("PDBI", version)
    except FileNotFoundError:
        pytest.fail(f"Cannot create PDBI telescope with version {version}")


@pytest.mark.parametrize("version", SMAVersions)
def test_read_all_SMA_versions(version):
    try:
        _ = Telescope.constructor("SMA", version)
    except FileNotFoundError:
        pytest.fail(f"Cannot create SMA telescope with version {version}")


@pytest.mark.parametrize("version", VLAVersions)
def rest_read_all_VLA_versions(version):
    try:
        _ = Telescope.constructor("VLA", version)
    except FileNotFoundError:
        pytest.fail(f"Cannot create VLA telescope with version {version}")


def test_read_SKA_LOW():
    tel = Telescope.constructor("SKA1LOW")
    assert len(tel.stations) == 512


def test_read_SKA_MID():
    tel = Telescope.constructor("SKA1MID")
    assert len(tel.stations) == 197


def test_read_VLBA():
    tel = Telescope.constructor("VLBA")
    assert len(tel.stations) == 10


def test_read_WSRT():
    tel = Telescope.constructor("WSRT")
    assert len(tel.stations) == 14


def test_RASCIL_telescope():
    tel = Telescope.constructor("MID", backend=SimulatorBackend.RASCIL)
    assert tel.backend is SimulatorBackend.RASCIL
    info = tel.get_backend_specific_information()
    assert isinstance(info, Configuration)


# Interesting and funny article on asserting with mocks:
# https://engineeringblog.yelp.com/2015/02/assert_called_once-threat-or-menace.html
@mock.patch("logging.warning", autospec=True)
def test_RASCIL_telescope_with_version_triggers_logging(mock_logging_warning):
    Telescope.constructor(
        "MID", backend=SimulatorBackend.RASCIL, version="Not None version"
    )
    assert mock_logging_warning.call_count == 1


def test_invalid_RASCIL_telescope():
    with pytest.raises(AssertionError):
        Telescope.constructor("FAKETELESCOPE", backend=SimulatorBackend.RASCIL)


def test_invalid_backend():
    with pytest.raises(AssertionError):
        Telescope.constructor("FAKETELESCOPE", backend="FAKEBACKEND")


def test_get_OSKAR_backend_information():
    tel = Telescope.constructor("MeerKAT", backend=SimulatorBackend.OSKAR)
    info = tel.get_backend_specific_information()
    assert isinstance(info, str)


def test_get_RASCIL_backend_information():
    from ska_sdp_datamodels.configuration.config_model import Configuration

    tel = Telescope.constructor("MID", backend=SimulatorBackend.RASCIL)
    info = tel.get_backend_specific_information()
    assert isinstance(info, Configuration)


def test_RASCIL_telescope_plot_file_created():
    with tempfile.TemporaryDirectory() as tmpfile:
        temp_plot_file_name = os.path.join(tmpfile, "test-plot.png")
        tel = Telescope.constructor("MID", backend=SimulatorBackend.RASCIL)
        tel.plot_telescope(temp_plot_file_name)
        assert os.path.exists(temp_plot_file_name)
        # It is tedious to check a specific file size. Even
        # small changes to the code creating the image will make
        # this test fail. Thus, I check only if the file size
        # is > 0.
        assert os.path.getsize(temp_plot_file_name) > 0


# There is an if statement in Telescope::plot_telescope for the
# RASCIL backend. Let's test it
def test_RASCIL_telescope_no_plot_file_created():
    with tempfile.TemporaryDirectory() as tmpfile:
        temp_plot_file_name = os.path.join(tmpfile, "test-plot.png")
        tel = Telescope.constructor("MID", backend=SimulatorBackend.RASCIL)
        tel.plot_telescope()
        assert not os.path.exists(temp_plot_file_name)


def test_get_invalid_backend_information():
    tel = Telescope.constructor("MeerKAT", backend=SimulatorBackend.OSKAR)
    # Modify backend
    tel.backend = "FAKEBACKEND"
    with pytest.raises(ValueError):
        tel.get_backend_specific_information()


@mock.patch("logging.warning", autospec=True)
def test_plot_invalid_backend(mock_logging_warning):
    tel = Telescope.constructor("MeerKAT", backend=SimulatorBackend.OSKAR)
    # Modify backend
    tel.backend = "FAKEBACKEND"
    # Attempt plotting, which triggers logging but no plot
    tel.plot_telescope()
    assert mock_logging_warning.call_count == 1


def test_ang_res():
    """
    At 1m wavelength, a 1km baseline resolves 1/1000rad => 206asec
    """
    wavelength = 1  # 1m
    freq = const.c.value / wavelength
    b = 1000  # 1km
    exp_ang_res_radians = wavelength / b
    exp_ang_res_arcsec = (exp_ang_res_radians * 180 * 3600) / np.pi
    ang_res_arcsec = Telescope.ang_res(freq, b)
    assert np.isclose(
        ang_res_arcsec, exp_ang_res_arcsec, rtol=1e-5
    ), f"Expected {exp_ang_res_arcsec}, got {ang_res_arcsec}"