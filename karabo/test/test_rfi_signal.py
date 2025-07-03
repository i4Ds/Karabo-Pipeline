import math
import os
import subprocess
from datetime import datetime, timedelta

import numpy as np
import pytest

from karabo.simulation.observation import Observation
from karabo.simulation.signal.rfi_signal import RFISignal
from karabo.simulation.telescope import SimulatorBackend, Telescope

# we need a credentials file to run the simulation.
# Its content is not important for this test.
credentials_filename = "test_credentials.yaml"
properties_filename = "test_properties.yaml"


def write_dummy_credentials_file(filename):
    """
    Write a dummy credentials file for testing purposes. We mock the call to
    the tab_sim command, so the content of this file is not important. But it
    is required to run the simulation.
    """
    f = open(filename, "w")
    f.write("credentials: test_credentials")
    f.close()


@pytest.fixture(scope="module")
def setup_observation() -> Observation:
    """
    Fixture to create a mock Observation object.

    Returns:
        A default observation that can be used in different modules
    """
    observation_length = timedelta(minutes=10)  # 14400 = 4hours
    integration_time = timedelta(seconds=8)

    freqs = np.arange(800e6, 1500e6, 10e6)
    freq = freqs[0]
    num_chan = 10

    return Observation(
        phase_centre_ra_deg=20,
        phase_centre_dec_deg=-30,
        start_date_and_time=datetime(2000, 3, 20, 12, 6, 39),
        length=observation_length,
        number_of_time_steps=int(
            observation_length.total_seconds() / integration_time.total_seconds()
        ),
        start_frequency_hz=freq,
        number_of_channels=num_chan,
        frequency_increment_hz=(freqs.max() - freqs.min()) / num_chan,
    )


@pytest.fixture(scope="module")
def setup_telescope() -> Telescope:
    """
    Fixture to create a mock Telescope object.
    We use the SKA-Mid telescope and the OSKAR backend as a default.
    Returns:
        A default telescope that can be used in different modules
    """

    return Telescope.constructor("SKA1MID", backend=SimulatorBackend.OSKAR)


def test_no_args_constructor():
    rfi_signal = RFISignal()
    # Check that the RFISignal object is created with default values
    assert rfi_signal is not None
    assert rfi_signal.G0_mean == 1.0
    assert rfi_signal.G0_std == 0.0
    assert rfi_signal.Gt_std_amp == 0.0
    assert rfi_signal.Gt_std_phase == 0.0
    assert rfi_signal.Gt_corr_amp == 0.0
    assert rfi_signal.Gt_corr_phase == 0.0
    assert rfi_signal.random_seed == 999


def test_class_has_attribs():
    rfi_signal = RFISignal()

    assert hasattr(rfi_signal, "G0_mean")
    assert hasattr(rfi_signal, "G0_std")
    assert hasattr(rfi_signal, "Gt_std_amp")
    assert hasattr(rfi_signal, "Gt_std_phase")
    assert hasattr(rfi_signal, "Gt_corr_amp")
    assert hasattr(rfi_signal, "Gt_corr_phase")
    assert hasattr(rfi_signal, "random_seed")


def test_cannot_open_credentials_file(mocker, setup_observation, setup_telescope):
    rfi_signal = RFISignal()

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value = subprocess.CompletedProcess(
        args=["sim_vis"], returncode=0, stdout="OK", stderr=""
    )

    with pytest.raises(FileNotFoundError):
        rfi_signal.run_simulation(
            setup_observation,
            setup_telescope,
            credentials_filename="non_existent_file.yaml",
        )


def test_property_filename_given(mocker, setup_observation, setup_telescope):
    """
    If we give a property filename, the RFISignal class should write the
    simulation properties to that file.
    """
    rfi_signal = RFISignal()

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value = subprocess.CompletedProcess(
        args=["sim_vis"], returncode=0, stdout="OK", stderr=""
    )

    write_dummy_credentials_file(credentials_filename)

    rfi_signal.run_simulation(
        setup_observation,
        setup_telescope,
        credentials_filename=credentials_filename,
        property_filename=properties_filename,
    )
    # Check that the properties file is created
    assert os.path.isfile(properties_filename)

    # os.remove(properties_filename)
    os.remove(credentials_filename)


def test_mandatory_properties_written(mocker, setup_observation, setup_telescope):
    """
    Test that the mandatory properties are written to the properties file.
    """
    rfi_signal = RFISignal()

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value = subprocess.CompletedProcess(
        args=["sim_vis"], returncode=0, stdout="OK", stderr=""
    )

    write_dummy_credentials_file(credentials_filename)

    rfi_signal.run_simulation(
        setup_observation,
        setup_telescope,
        credentials_filename=credentials_filename,
        property_filename=properties_filename,
    )

    mandatory_attributes = [
        "gains",
        "diagnostics",
        "dask",
        "observation",
        "telescope",
    ]

    # Check that the properties file contains the mandatory properties
    with open(properties_filename) as testfile:
        content = testfile.read()
        for attr in mandatory_attributes:
            assert attr in content

    os.remove(properties_filename)
    os.remove(credentials_filename)


def test_telescope_property_written_properly(
    mocker, setup_observation, setup_telescope
):
    """
    This test checks that the latitude and longitude of the telescope
    are written correctly to the properties file. Sometimes latitude and
    longitude get confused.
    """
    rfi_signal = RFISignal()

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value = subprocess.CompletedProcess(
        args=["sim_vis"], returncode=0, stdout="OK", stderr=""
    )

    write_dummy_credentials_file(credentials_filename)

    rfi_signal.run_simulation(
        setup_observation,
        setup_telescope,
        credentials_filename=credentials_filename,
        property_filename=properties_filename,
    )

    lat_to_check = setup_telescope.centre_latitude
    lon_to_check = setup_telescope.centre_longitude
    with open(properties_filename) as testfile:
        for line in testfile.readlines():
            if "latitude" in line:
                lat = float(line.split(":")[1].strip())
                assert math.isclose(lat, lat_to_check, rel_tol=1e-3)
            if "longitude" in line:
                lon = float(line.split(":")[1].strip())
                assert math.isclose(lon, lon_to_check, rel_tol=1e-3)

    # os.remove(properties_filename)
    os.remove(credentials_filename)


def test_provide_cache_dir():
    rfi_signal = RFISignal()

    assert rfi_signal.tmp_dir is not None
    assert os.path.isdir(rfi_signal.tmp_dir)

    assert rfi_signal.cache_dir is not None
    assert os.path.isdir(rfi_signal.cache_dir)


def test_run_simulation(mocker, setup_observation, setup_telescope):
    """
    This test checks that the latitude and longitude of the telescope
    are written correctly to the properties file. Sometimes latitude and
    longitude get confused.
    """
    rfi_signal = RFISignal()

    write_dummy_credentials_file(credentials_filename)

    rfi_signal.run_simulation(
        setup_observation,
        setup_telescope,
        credentials_filename=credentials_filename,
        property_filename=properties_filename,
    )

    os.remove(properties_filename)
    os.remove(credentials_filename)
