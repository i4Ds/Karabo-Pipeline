import os
import shutil
import subprocess
import sys


def is_cuda_available() -> bool:
    """
    Checks if CUDA-compatible GPU is available on the system by invoking nvidia-smi.

    Returns:
        bool: True if a CUDA-compatible GPU is found, otherwise False.
    """
    # "2> /dev/null" suppresses stderr if the command is not found and returns ""
    output = os.popen("nvidia-smi 2> /dev/null").read()
    if "GPU" in output and "CUDA" in output:
        return True
    elif (
        "nvidia-smi: not found" in output
        or "NVIDIA-SMI has failed because it couldn't communicate" in output
        or "" == output
    ):
        return False
    else:
        print("Unexpected output from nvidia-smi:", f'"{output}"')
        return False


def is_oskar_opencl_gpu_available(double_precision: bool = False) -> bool:
    """
    Checks if OSKAR can use an OpenCL GPU, e.g. the GPU of an Apple Silicon Mac.

    Asks `oskar_system_info`, which only lists OpenCL devices if OSKAR was built
    with OpenCL support.

    Args:
        double_precision: Require a device that supports double precision.
            Apple GPUs do not, so OSKAR runs double precision on the CPU there.

    Returns:
        bool: True if OSKAR finds a usable OpenCL GPU, otherwise False.
    """
    exe = os.path.join(sys.prefix, "bin", "oskar_system_info")
    if not os.path.exists(exe):
        exe = shutil.which("oskar_system_info") or ""
    if not exe:
        return False
    output = subprocess.run([exe], capture_output=True, text=True).stdout
    if "OpenCL devices (" not in output:
        return False
    opencl = output.split("OpenCL devices (", 1)[1]
    if opencl.startswith("0)"):
        return False
    if double_precision:
        return "Supports double precision         : true" in opencl
    return True


def get_gpu_memory() -> int:
    """
    Retrieves the available GPU memory in MiB by invoking nvidia-smi.

    Returns:
        int: Available GPU memory in MiB.

    Raises:
        RuntimeError: If unexpected output is encountered when running nvidia-smi.
    """
    # "2> /dev/null" suppresses stderr if the command is not found and returns ""
    output = os.popen("nvidia-smi 2> /dev/null").read()
    if "GPU" in output and "CUDA" in output:
        return int(output.split("MiB")[1].split(" ")[-1])
    else:
        raise RuntimeError("Unexpected output from nvidia-smi:", f'"{output}"')


if __name__ == "__main__":
    print(is_cuda_available())
