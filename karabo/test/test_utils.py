import collections
import sys
from importlib import metadata

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


def test_pkg_dist():
    def packages_distributions():
        """
        Return a mapping of top-level packages to their distributions.
        Note: copied from https://github.com/python/importlib_metadata/pull/287
        """
        pkg_to_dist = collections.defaultdict(list)
        for dist in metadata.distributions():
            for pkg in (dist.read_text("top_level.txt") or "").split():
                try:
                    pkg_to_dist[pkg].append(dist.metadata["Name"])
                except KeyError:
                    print(f"pkg-dist: {pkg=}", file=sys.stdout)
                    print(f"pkg-dist: {pkg=}", file=sys.stderr)
                    raise
        return dict(pkg_to_dist)

    _ = packages_distributions()
