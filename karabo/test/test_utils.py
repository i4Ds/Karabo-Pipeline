import os
import unittest

from karabo.util.gpu_util import get_gpu_memory, is_cuda_available

RUN_GPU_TESTS = os.environ.get("RUN_GPU_TESTS", "false").lower() == "true"


class TestGpuUtils(unittest.TestCase):
    @unittest.skipIf(not RUN_GPU_TESTS, "GPU tests are disabled")
    def test_get_gpu_memory(self):
        memory = get_gpu_memory()
        assert isinstance(memory, int)
        assert memory > 0

    @unittest.skipIf(RUN_GPU_TESTS, "Does does not fail when GPU is available")
    def test_gpu_memory_error(self):
        with self.assertRaises(RuntimeError):
            get_gpu_memory()

    def test_is_cuda_available(self):
        assert isinstance(is_cuda_available(), bool)

    @unittest.skipIf(not RUN_GPU_TESTS, "GPU tests are disabled")
    def test_is_cuda_available_true(self):
        assert is_cuda_available()
