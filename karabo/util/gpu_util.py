import os


def is_cuda_available():
    # Check available GPU by invoking nvidia-smi
    try:
        output = os.popen("nvidia-smi").read()
        if "GPU" in output and "CUDA" in output:
            return True
        else:
            return False
    except Exception:
        return False


if __name__ == "__main__":
    print(is_cuda_available())
