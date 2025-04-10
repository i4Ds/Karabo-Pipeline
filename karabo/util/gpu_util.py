import os


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
