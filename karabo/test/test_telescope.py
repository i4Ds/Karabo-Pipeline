import os

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


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    # make dir for result files
    if not os.path.exists("result/tel"):
        os.makedirs("result/tel")


def test_plot():
    tel = Telescope(0, -50)
    tel.plot_telescope("result/tel/simple_tel.png")


def test_read_tm_file():
    tel = Telescope.get_OSKAR_Example_Telescope()
    tel.plot_telescope("result/tel/oskar_tel.png")
    assert len(tel.stations) == 30


def test_convert_to_oskar():
    tel = Telescope.get_OSKAR_Example_Telescope()
    oskar_tel = tel.get_OSKAR_telescope()
    assert oskar_tel.get_num_stations() == 30


def test_read_alma_file():
    tel = Telescope.get_ALMA_Telescope(ALMAVersions.CYCLE_1_1)
    tel.plot_telescope("result/tel/alma_tel.png")
    assert len(tel.stations) == 32


def test_read_meerkat_file():
    tel = Telescope.get_MEERKAT_Telescope()
    tel.plot_telescope("result/tel/meerkat_tel.png")
    assert len(tel.stations) == 64


@pytest.mark.parametrize("version", ALMAVersions)
def test_read_all_ALMA_versions(version):
    tel = Telescope.get_ALMA_Telescope(version)
    tel.plot_telescope(f"result/tel/{version}.png")


@pytest.mark.parametrize("version", ACAVersions)
def test_read_all_ACA_versions(version):
    tel = Telescope.get_ACA_Telescope(version)
    tel.plot_telescope(f"result/tel/{version}.png")


@pytest.mark.parametrize("version", CARMAVersions)
def test_read_all_CARMA_versions(version):
    tel = Telescope.get_CARMA_Telescope(version)
    tel.plot_telescope(f"result/tel/{version}.png")


@pytest.mark.parametrize("version", NGVLAVersions)
def test_read_all_NG_VLA_versions(version):
    tel = Telescope.get_NG_VLA_Telescope(version)
    tel.plot_telescope(f"result/tel/{version}.png")


@pytest.mark.parametrize("version", PDBIVersions)
def test_read_all_PDBI_versions(version):
    tel = Telescope.get_PDBI_Telescope(version)
    tel.plot_telescope(f"result/tel/{version}.png")


@pytest.mark.parametrize("version", SMAVersions)
def test_read_all_SMA_versions(version):
    tel = Telescope.get_SMA_Telescope(version)
    tel.plot_telescope(f"result/tel/{version}.png")


@pytest.mark.parametrize("version", VLAVersions)
def rest_read_all_VLA_versions(version):
    tel = Telescope.get_VLA_Telescope(version)
    tel.plot_telescope(f"result/tel/{version}.png")


def test_read_SKA_LOW():
    tel = Telescope.get_SKA1_LOW_Telescope()
    tel.plot_telescope("result/tel/ska_low.png")


def test_read_SKA_MID():
    tel = Telescope.get_SKA1_LOW_Telescope()
    tel.plot_telescope("result/tel/ska_mid.png")


def test_read_VLBA():
    tel = Telescope.get_VLBA_Telescope()
    tel.plot_telescope("result/tel/vlba.png")


def test_read_WSRT():
    Telescope.get_WSRT_Telescope()
