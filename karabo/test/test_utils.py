import logging
import os

import pytest

import karabo
from karabo.test.conftest import RUN_GPU_TESTS
from karabo.util.gpu_util import get_gpu_memory, is_cuda_available


def test_is_cuda_available():
    assert isinstance(is_cuda_available(), bool)


CUDA_AVAILABLE = is_cuda_available()


@pytest.mark.skipif(
    CUDA_AVAILABLE,
    reason="get-gpu-memory thorows a RuntimeError only if cuda is not available",
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


def test_rascil_logger_path_valid():
    logger_path = karabo.logger_name
    assert os.path.exists(logger_path)
    assert os.path.isfile(logger_path)


def test_rascil_warning_supressed(caplog):
    logger = karabo.logger
    message_to_suppress = (
        "The RASCIL data directory is not available - continuing but any "
        "simulations will fail"
    )
    with caplog.at_level(logging.WARNING):
        logger.warning(message_to_suppress)
    assert message_to_suppress not in caplog.text
