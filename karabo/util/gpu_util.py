import os


def is_cuda_available() -> bool:
    # Check available GPU by invoking nvidia-smi
    # "2> /dev/null" surpresses stderr if command not found and returns ""
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
    # Check available GPU by invoking nvidia-smi
    # "2> /dev/null" surpresses stderr if command not found and returns ""
    output = os.popen("nvidia-smi 2> /dev/null").read()
    if "GPU" in output and "CUDA" in output:
        return int(output.split("MiB")[1].split(" ")[-1])
    else:
        raise RuntimeError("Unexpected output from nvidia-smi:", f'"{output}"')


if __name__ == "__main__":
    print(is_cuda_available())
