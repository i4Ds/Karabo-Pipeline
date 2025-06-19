import os
import subprocess
from datetime import datetime, timedelta

import numpy as np
import pytest

from karabo.simulation.observation import Observation
from karabo.simulation.signal.rfi_signal import RFISignal
from karabo.simulation.telescope import SimulatorBackend, Telescope


@pytest.fixture
def setup_observation() -> Observation:
    """
    Fixture to create a mock Observation object.

    Returns:
        A default observation that can be used in different modules
    """
    observation_length = timedelta(hours=4)  # 14400 = 4hours
    integration_time = timedelta(hours=0.5)

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


@pytest.fixture
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


def test_can_set_all_attribs():
    rfi_signal = RFISignal()

    rfi_signal.G0_mean = 1.0
    rfi_signal.G0_std = 0.1
    rfi_signal.Gt_std_amp = 0.2
    rfi_signal.Gt_std_phase = 0.3
    rfi_signal.Gt_corr_amp = 0.4
    rfi_signal.Gt_corr_phase = 0.5
    rfi_signal.random_seed = 42

    assert rfi_signal.G0_mean == 1.0
    assert rfi_signal.G0_std == 0.1
    assert rfi_signal.Gt_std_amp == 0.2
    assert rfi_signal.Gt_std_phase == 0.3
    assert rfi_signal.Gt_corr_amp == 0.4
    assert rfi_signal.Gt_corr_phase == 0.5
    assert rfi_signal.random_seed == 42


def test_cannot_open_credentials_file_(mocker, setup_observation, setup_telescope):
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


def test_property_filename_given_(mocker, setup_observation, setup_telescope):
    """
    If we give a property filename, the RFISignal class should write the
    simulation properties to that file.
    """
    rfi_signal = RFISignal()

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value = subprocess.CompletedProcess(
        args=["sim_vis"], returncode=0, stdout="OK", stderr=""
    )

    # we need a credentials file to run the simulation.
    # Its content is not important for this test.
    credentials_filename = "test_credentials.yaml"
    propertys_filename = "test_properties.yaml"

    f = open(credentials_filename, "w")
    f.write("credentials: test_credentials")
    f.close()

    rfi_signal.run_simulation(
        setup_observation,
        setup_telescope,
        credentials_filename=credentials_filename,
        property_filename=propertys_filename,
    )
    # Check that the properties file is created
    assert os.path.isfile(propertys_filename)

    # Check that the properties file contains the expected content
    with open(propertys_filename) as testfile:
        assert "gains" in testfile.read()

    os.remove(propertys_filename)
    os.remove(credentials_filename)
