import logging
import os

import pytest

import karabo
import karabo.util.rascil_util
from karabo.test.conftest import RUN_GPU_TESTS
from karabo.util.gpu_util import get_gpu_memory, is_cuda_available


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


def test_suppress_rascil_warning(caplog):
    path_to_rascil_module = karabo.util.rascil_util.DATA_DIR_WARNING_PATH_TO_MODULE
    # Make sure RASCIL module of concern is where we expect it
    assert os.path.isfile(path_to_rascil_module)

    logger = logging.getLogger(path_to_rascil_module)
    with caplog.at_level(logging.WARNING):
        logger.warning(karabo.util.rascil_util.DATA_DIR_WARNING_MESSAGE)
    assert karabo.util.rascil_util.DATA_DIR_WARNING_MESSAGE not in caplog.text
