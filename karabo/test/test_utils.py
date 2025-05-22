import importlib
import logging
import os

import pytest
import rascil

import karabo
import karabo.util.rascil_util
from karabo.test.conftest import RUN_GPU_TESTS
from karabo.util.gpu_util import get_gpu_memory, is_cuda_available
from karabo.util.helpers import Environment


def test_is_cuda_available():
    assert isinstance(is_cuda_available(), bool)


CUDA_AVAILABLE = is_cuda_available()


@pytest.mark.skipif(
    CUDA_AVAILABLE,
    reason="get-gpu-memory throws a RuntimeError only if cuda is not available",
)
def test_gpu_memory_error():
    with pytest.raises(RuntimeError):
        get_gpu_memory()


@pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="get-gpu-memory works only if cuda is available"
)
def test_get_gpu_memory():
    memory = get_gpu_memory()
    assert isinstance(memory, int)
    assert memory > 0


@pytest.mark.skipif(not RUN_GPU_TESTS, reason="GPU tests are disabled")
def test_is_cuda_available_true():
    assert CUDA_AVAILABLE


def test_version():
    assert karabo.__version__ != "0+unknown"


def test_suppress_rascil_warning(caplog: pytest.LogCaptureFixture):
    path_to_rascil_module = karabo.util.rascil_util.DATA_DIR_WARNING_PATH_TO_MODULE
    # Make sure RASCIL module of concern is where we expect it.
    # Otherwise the suppression won't work since the logger uses the file path
    # of the module as its name.
    assert os.path.isfile(path_to_rascil_module)

    logger_name = path_to_rascil_module
    logger = logging.getLogger(logger_name)
    # Remove filter that was already added at this point by karabo.__init__.py
    logger.filters.clear()
    with caplog.at_level(level=logging.WARNING, logger=logger_name):
        # Load rascil, warning should be logged
        importlib.reload(rascil)
    warning_message = karabo.util.rascil_util.DATA_DIR_WARNING_MESSAGE
    assert any(warning_message in record.message for record in caplog.records)
    caplog.clear()
    with caplog.at_level(level=logging.WARNING, logger=logger_name):
        # Load karabo, installing the filter in __init__.py that suppresses the warning
        importlib.reload(karabo)
        # Load rascil, warning should not be logged
        importlib.reload(rascil)
    assert not any(warning_message in record.message for record in caplog.records)


def test_environment():
    envs = {
        "STRING": "String",
        "NONE": "None",
        "TRUE": "True",
        "FALSE": "False",
        "ZERO": "0",
        "POS_INT": "42",
        "NEG_INT": "-13",
        "POS_FLOAT": "3.1415926535",
        "NEG_FLOAT": "-2.7182818",
        "SCI_FLOAT": "-2.5e-3",
        "EMPTY": "",
    }
    for k, v in envs.items():
        os.environ[k] = v

    assert isinstance(Environment.get("STRING", str), str)
    assert isinstance(Environment.get("EMPTY", str), str)
    assert isinstance(Environment.get("TRUE", str), str)
    assert isinstance(Environment.get("ZERO", str), str)
    assert isinstance(Environment.get("POS_INT", str), str)
    assert isinstance(Environment.get("SCI_FLOAT", str), str)
    assert Environment.get("TRUE", bool) is True
    assert Environment.get("FALSE", bool) is False
    assert isinstance(Environment.get("ZERO", float), float)
    assert isinstance(Environment.get("ZERO", int), int)
    assert isinstance(Environment.get("POS_INT", int), int)
    assert isinstance(Environment.get("NEG_INT", int), int)
    assert isinstance(Environment.get("POS_FLOAT", float), float)
    assert isinstance(Environment.get("NEG_FLOAT", float), float)
    assert isinstance(Environment.get("SCI_FLOAT", float), float)
    assert Environment.get("NONE", str, allow_none_parsing=True) is None
    assert Environment.get("NONE", bool, allow_none_parsing=True) is None
    assert Environment.get("NONE", float, allow_none_parsing=True) is None
    assert Environment.get("DOES_NOT_EXIST", float, 0.3) == 0.3
    assert Environment.get("NEG_FLOAT", float, 0.3) == -2.7182818
    assert Environment.get("DOES_NOT_EXIST", bool, None) is None
    with pytest.raises(KeyError):
        Environment.get("DOES_NOT_EXIST", bool)
    with pytest.raises(ValueError):
        Environment.get("STRING", bool)
    with pytest.raises(ValueError):
        Environment.get("NONE", float)
    with pytest.raises(ValueError):
        Environment.get("TRUE", float)
    with pytest.raises(ValueError):
        Environment.get("FALSE", int)
    with pytest.raises(ValueError):
        Environment.get("ZERO", bool)
    with pytest.raises(ValueError):
        Environment.get("POS_FLOAT", int)
    with pytest.raises(ValueError):
        Environment.get("SCI_FLOAT", int)
