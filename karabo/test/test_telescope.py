import os
import tempfile

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


def test_convert_to_oskar():
    tel = Telescope.get_OSKAR_Example_Telescope()
    oskar_tel = tel.get_OSKAR_telescope()
    assert oskar_tel.get_num_stations() == 30


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


def test_invalid_RASCIL_telescope():
    with pytest.raises(
        ValueError,
        match="Requested telescope FAKEEXAMPLE is not supported by this backend",
    ):
        Telescope.constructor("FAKEEXAMPLE", backend=SimulatorBackend.RASCIL)
