import os
import tempfile
from unittest import mock

import pytest

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


def test_read_tm_file():
    tel = Telescope.constructor("EXAMPLE")
    with tempfile.TemporaryDirectory() as tmpdir:
        tel.plot_telescope(os.path.join(tmpdir, "oskar_tel.png"))
        assert len(tel.stations) == 30


def test_deprecated_read_from_file():
    tel = Telescope.constructor("EXAMPLE")
    with pytest.raises(DeprecationWarning):
        tel.read_from_file("fakefilename")


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


def test_read_alma_file():
    tel = Telescope.constructor("ALMA", ALMAVersions.CYCLE_1_1)
    tel.plot_telescope()
    assert len(tel.stations) == 32


def test_read_meerkat_file():
    tel = Telescope.constructor("MeerKAT")
    tel.plot_telescope()
    assert len(tel.stations) == 64


@pytest.mark.parametrize("version", ALMAVersions)
def test_read_all_ALMA_versions(version):
    tel = Telescope.constructor("ALMA", version)
    tel.plot_telescope()


@pytest.mark.parametrize("version", ACAVersions)
def test_read_all_ACA_versions(version):
    tel = Telescope.constructor("ACA", version)
    tel.plot_telescope()


@pytest.mark.parametrize("version", CARMAVersions)
def test_read_all_CARMA_versions(version):
    tel = Telescope.constructor("CARMA", version)
    tel.plot_telescope()


@pytest.mark.parametrize("version", NGVLAVersions)
def test_read_all_NG_VLA_versions(version):
    tel = Telescope.constructor("NGVLA", version)
    tel.plot_telescope()


@pytest.mark.parametrize("version", PDBIVersions)
def test_read_all_PDBI_versions(version):
    tel = Telescope.constructor("PDBI", version)
    tel.plot_telescope()


@pytest.mark.parametrize("version", SMAVersions)
def test_read_all_SMA_versions(version):
    tel = Telescope.constructor("SMA", version)
    tel.plot_telescope()


@pytest.mark.parametrize("version", VLAVersions)
def rest_read_all_VLA_versions(version):
    tel = Telescope.constructor("VLA", version)
    tel.plot_telescope()


def test_read_SKA_LOW():
    tel = Telescope.constructor("SKA1LOW")
    tel.plot_telescope()


def test_read_SKA_MID():
    tel = Telescope.constructor("SKA1MID")
    tel.plot_telescope()


def test_read_VLBA():
    tel = Telescope.constructor("VLBA")
    tel.plot_telescope()


def test_read_WSRT():
    tel = Telescope.constructor("WSRT")
    tel.plot_telescope()


def test_RASCIL_telescope():
    tel = Telescope.constructor("MID", backend=SimulatorBackend.RASCIL)
    assert tel.backend is SimulatorBackend.RASCIL

    tel.plot_telescope()


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
