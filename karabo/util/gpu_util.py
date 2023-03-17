import os


def is_cuda_available() -> bool:
    # Check available GPU by invoking nvidia-smi
    try:
        output = os.popen("nvidia-smi").read()
        if "GPU" in output and "CUDA" in output:
            return True
        elif (
            "nvidia-smi: not found" in output
            or "NVIDIA-SMI has failed because it couldn't communicate" in output
        ):
            return False
        else:
            print("Unexpected output from nvidia-smi: ", output)
            return False
    except Exception:
        return False


if __name__ == "__main__":
    print(is_cuda_available())
