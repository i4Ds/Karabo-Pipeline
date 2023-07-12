import os

import pytest

from karabo.util.gpu_util import get_gpu_memory, is_cuda_available
from karabo.version import __version__

RUN_GPU_TESTS = os.environ.get("RUN_GPU_TESTS", "false").lower() == "true"


@pytest.mark.skipif(not RUN_GPU_TESTS, reason="GPU tests are disabled")
def test_get_gpu_memory():
    memory = get_gpu_memory()
    assert isinstance(memory, int)
    assert memory > 0


@pytest.mark.skipif(RUN_GPU_TESTS, reason="Does not fail when GPU is available")
def test_gpu_memory_error():
    with pytest.raises(RuntimeError):
        get_gpu_memory()


def test_is_cuda_available():
    assert isinstance(is_cuda_available(), bool)


@pytest.mark.skipif(not RUN_GPU_TESTS, reason="GPU tests are disabled")
def test_is_cuda_available_true():
    assert is_cuda_available()


def test_version():
    assert isinstance(__version__, str)
