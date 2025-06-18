from datetime import datetime, timedelta

import numpy as np
import pytest

from karabo.simulation.observation import Observation
from karabo.simulation.signal.rfi_signal import RFISignal
from karabo.simulation.sky_model import SkyModel


@pytest.fixture
def setup_observation():
    """Fixture to create a mock Observation object."""
    """
        Returns a default observation that can be used in different modules
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
def sky_model():
    """Fixture to create a mock SkyModel object."""
    return SkyModel.sky_test()


def test_no_args_constructor():
    rfi_signal = RFISignal()
    # Check that the RFISignal object is created with default values
    assert rfi_signal is not None
    rfi_signal.G0_mean = 1.0
    rfi_signal.G0_std = 0.0
    rfi_signal.Gt_std_amp = 0.0
    rfi_signal.Gt_std_phase = 0.0
    rfi_signal.Gt_corr_amp = 0.0
    rfi_signal.Gt_corr_phase = 0.0
    rfi_signal.random_seed = 999


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


def test_plot_methods_not_implemented(setup_observation, sky_model):
    rfi_signal = RFISignal()

    # Check that the plot methods raise NotImplementedError
    try:
        rfi_signal.plot_uv_coverage()
    except NotImplementedError as e:
        assert str(e).startswith("plot_uv_coverage")

    try:
        rfi_signal.plot_rfi_separation()
    except NotImplementedError as e:
        assert str(e).startswith("plot_rfi_separation")

    try:
        rfi_signal.plot_source_altitude()
    except NotImplementedError as e:
        assert str(e).startswith("plot_source_altitude")

    try:
        rfi_signal.run_simulation(setup_observation, sky_model)
    except NotImplementedError as e:
        assert str(e).startswith("simulate method")
